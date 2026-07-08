import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# ==========================================================================
# train_util.py : boucle d'entraînement (TrainLoop) pour un modèle de
# diffusion (UNet). Ce module n'est PAS utilisé par les scripts main_ddpir*
# de ce dépôt (qui ne font que de l'inférence avec un checkpoint gelé),
# mais c'est le point d'entrée à utiliser si l'on veut :
#   - entraîner un modèle de diffusion from scratch, ou
#   - fine-tuner un checkpoint pré-entraîné (ex: 256x256_diffusion_uncond.pt)
#     sur un nouveau dataset, par exemple des images satellites RGB 256x256.
#
# Pour fine-tuner, il faut écrire un petit script (absent de ce dépôt, voir
# le style de guided-diffusion/scripts/image_train.py) qui :
#   1. construit `model` et `diffusion` via script_util.create_model_and_diffusion
#      avec EXACTEMENT la même architecture que le checkpoint de départ,
#   2. charge les poids du checkpoint dans `model` (model.load_state_dict),
#   3. construit `data` via image_datasets.load_data sur le dataset satellite,
#   4. instancie TrainLoop(...) avec les paramètres ci-dessous et appelle
#      .run_loop().
# ==========================================================================

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def _is_main_process():
    # Vrai s'il n'y a pas de process group torch.distributed initialise
    # (entrainement mono-processus, cf. dist_util.setup_dist) ou si on est
    # le rang 0. Permet a TrainLoop de fonctionner aussi bien en mono-GPU/
    # CPU (pas de process group) qu'en multi-GPU/multi-noeud (process group
    # reel via MPI).
    return not dist.is_initialized() or dist.get_rank() == 0


class TrainLoop:
    """
    Boucle d'entraînement générique pour un modèle de diffusion (UNet + EMA
    + AdamW + mixed precision optionnelle + DDP optionnel).

    Paramètres d'entraînement et conseils pour un fine-tuning sur un
    dataset satellite RGB 256x256 :

    :param model: le UNetModel (voir unet.py) à entraîner. Pour un
        fine-tuning, ce modèle doit avoir la même architecture que le
        checkpoint chargé (num_channels, num_res_blocks, attention_resolutions,
        channel_mult...) et ses poids doivent déjà être chargés AVANT de
        créer le TrainLoop (voir _load_and_sync_parameters qui ne fait que
        gérer la reprise après coupure, pas le chargement du checkpoint de
        base pré-entraîné).
    :param diffusion: l'objet GaussianDiffusion (voir gaussian_diffusion.py)
        qui définit le schedule de bruit et calcule la loss. Doit être créé
        avec la même config (noise_schedule, learn_sigma, predict_xstart...)
        que celle utilisée pour pré-entraîner le checkpoint, sinon les poids
        et le schedule de bruit seront incohérents.
    :param data: générateur infini (batch, cond) produit par
        image_datasets.load_data. `batch` est un tenseur NCHW normalisé en
        [-1, 1]; `cond` est un dict (vide si class_cond=False, ce qui est le
        cas typique pour des images satellites sans label de classe).
    :param batch_size: taille de batch effective par device. Sur des images
        satellites 256x256, la mémoire GPU est le facteur limitant : réduire
        batch_size (ex: 4-8) si l'entraînement du modèle 256x256 (~550M
        paramètres pour 256x256_diffusion_uncond) sature le VRAM.
    :param microbatch: sous-découpage de batch_size pour limiter le pic
        mémoire (accumulation de gradient). -1 ou <=0 désactive le
        micro-batching (microbatch = batch_size). Utile pour fine-tuner un
        gros modèle avec un batch_size logique élevé mais peu de VRAM,
        par ex. batch_size=32, microbatch=4.
    :param lr: taux d'apprentissage de AdamW. Pour un entraînement from
        scratch, ~1e-4 est typique. Pour un FINE-TUNING d'un checkpoint déjà
        pré-entraîné, utiliser un lr nettement plus petit (ex: 1e-5 à 5e-5)
        pour éviter de détruire les poids pré-appris avant que le modèle
        n'ait vu assez d'images satellites.
    :param ema_rate: taux (ou liste de taux séparés par des virgules, ex.
        "0.9999,0.9995") de moyenne mobile exponentielle des poids, utilisée
        pour l'inférence finale (voir save()). Avec un petit dataset de
        fine-tuning, un ema_rate plus faible (ex. 0.999) permet à la moyenne
        de suivre plus vite les nouveaux poids adaptés au domaine satellite.
    :param log_interval: fréquence (en steps) d'écriture des logs (loss,
        lg_loss_scale, samples vus...).
    :param save_interval: fréquence (en steps) de sauvegarde des checkpoints
        (model + EMA + optimiseur). À réduire pour un fine-tuning court afin
        de ne pas perdre un run interrompu.
    :param resume_checkpoint: chemin vers un checkpoint "modelNNNNNN.pt" à
        reprendre. C'est ICI qu'on doit pointer vers le modèle pré-entraîné
        (ex: "models/256x256_diffusion_uncond.pt") pour démarrer un
        fine-tuning au lieu d'un entraînement from scratch. Le nom de
        fichier doit contenir le numéro de step ("modelNNNNNN.pt") pour que
        parse_resume_step_from_filename puisse retrouver le step de départ
        et charger l'EMA/l'optimiseur correspondants; sinon renommer le
        checkpoint en "model000000.pt" et resume_step repartira de 0.
    :param use_fp16: active l'entraînement en précision mixte (fp16) via
        MixedPrecisionTrainer. Recommandé pour fine-tuner un grand modèle
        (256x256_diffusion_uncond) avec un GPU à mémoire limitée : réduit le
        VRAM utilisé et accélère l'entraînement sur GPU récents.
    :param fp16_scale_growth: vitesse de croissance du facteur d'échelle du
        gradient en fp16 (rarement à modifier).
    :param schedule_sampler: stratégie d'échantillonnage des timesteps t
        pour chaque exemple d'entraînement (voir resample.py). Par défaut
        UniformSampler (tirage uniforme dans [0, num_timesteps)). Un
        LossAwareSampler (ex: échantillonnage par importance) peut accélérer
        la convergence sur un petit dataset spécialisé comme le satellite.
    :param weight_decay: régularisation L2 de AdamW. 0.0 par défaut; garder
        faible (0.0 à 1e-4) pour un fine-tuning afin de ne pas trop
        contraindre les poids pré-entraînés.
    :param lr_anneal_steps: si > 0, nombre total de steps sur lesquels le lr
        décroît linéairement jusqu'à 0 (voir _anneal_lr) et après lequel
        run_loop s'arrête. Pour un fine-tuning avec un budget de steps
        connu (ex: dataset satellite restreint), fixer cette valeur permet
        un lr-decay propre en fin d'entraînement plutôt qu'un lr constant.
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * (
            dist.get_world_size() if dist.is_initialized() else 1
        )

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # DDP (data parallelisme multi-GPU) n'a de sens que s'il existe un
        # vrai process group multi-processus (dist.is_initialized() et
        # world_size > 1) : sur un entrainement mono-GPU/CPU (pas de
        # process group, cf. dist_util.setup_dist), on utilise directement
        # self.model sans DDP - c'est le cas typique d'un fine-tuning sur
        # une seule machine.
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        if th.cuda.is_available() and distributed:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if distributed and not th.cuda.is_available():
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if _is_main_process():
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if _is_main_process():
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        # Boucle principale : tourne indéfiniment (lr_anneal_steps=0) ou
        # jusqu'à ce que step + resume_step atteigne lr_anneal_steps. Pour un
        # fine-tuning avec un dataset satellite limité, préférer fixer
        # lr_anneal_steps à un budget de steps raisonnable (ex: quelques
        # milliers, à ajuster selon la taille du dataset et save_interval)
        # plutôt que de laisser tourner indéfiniment et risquer l'overfitting.
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        # Découpe le batch en micro-batchs (voir le paramètre `microbatch`)
        # pour limiter la mémoire GPU pic, puis tire un timestep aléatoire t
        # par exemple via schedule_sampler et calcule la loss de diffusion
        # (diffusion.training_losses, voir gaussian_diffusion.py) sur chaque
        # micro-batch avant d'accumuler les gradients.
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if _is_main_process():
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if _is_main_process():
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        if dist.is_initialized():
            dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
