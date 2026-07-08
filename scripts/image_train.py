"""
Script d'entraînement / fine-tuning d'un modèle de diffusion (UNet) avec
guided_diffusion.

Ce dépôt (DiffPIR) n'utilise guided_diffusion que pour l'INFÉRENCE
(main_ddpir*.py chargent un checkpoint gelé). Ce script est le point
d'entrée manquant pour :
  - entraîner un modèle de diffusion from scratch sur un nouveau dataset, ou
  - FINE-TUNER un checkpoint pré-entraîné (ex: model_zoo/256x256_diffusion_uncond.pt)
    sur un nouveau dataset, typiquement des images satellites RGB 256x256.

Il assemble les briques documentées dans guided_diffusion/ :
  - image_datasets.load_data      : dataloader infini (images -> [-1, 1])
  - script_util.create_model_and_diffusion : construit UNetModel + GaussianDiffusion
  - resample.create_named_schedule_sampler : stratégie d'échantillonnage des timesteps
  - train_util.TrainLoop          : boucle d'entraînement (AdamW + EMA + fp16 optionnel)

--------------------------------------------------------------------------
Exemple : FINE-TUNER 256x256_diffusion_uncond sur un dataset satellite RGB
--------------------------------------------------------------------------

    python scripts/image_train.py \
        --data_dir /chemin/vers/dataset_satellite \
        --resume_checkpoint model_zoo/256x256_diffusion_uncond.pt \
        --image_size 256 --num_channels 256 --num_res_blocks 2 \
        --attention_resolutions 8,16,32 --learn_sigma True \
        --class_cond False \
        --lr 2e-5 --batch_size 4 --microbatch 2 \
        --use_fp16 True --use_checkpoint True \
        --log_interval 50 --save_interval 2000

Points clés pour le fine-tuning (voir aussi les docstrings dans
guided_diffusion/*.py pour le détail de chaque paramètre) :

  - `--resume_checkpoint` doit pointer vers le .pt pré-entraîné. Comme son
    nom ne contient pas "modelNNNNNN", TrainLoop repart avec resume_step=0
    (nouvel entraînement) tout en chargeant les poids pré-entraînés — c'est
    le comportement voulu pour un fine-tuning (par opposition à une reprise
    après coupure, où le checkpoint s'appelle "model012345.pt").
  - L'architecture (`num_channels`, `num_res_blocks`, `attention_resolutions`,
    `learn_sigma`, `class_cond`, `channel_mult`, `noise_schedule`,
    `diffusion_steps`) DOIT correspondre exactement à celle utilisée pour
    pré-entraîner le checkpoint choisi, sinon le chargement des poids
    échoue. Pour "256x256_diffusion_uncond" : num_channels=256,
    num_res_blocks=2, attention_resolutions="8,16,32", learn_sigma=True,
    class_cond=False. Pour "diffusion_ffhq_10m" (modèle plus léger) :
    num_channels=128, num_res_blocks=1, attention_resolutions="16" (voir
    main_ddpir.py, model_config).
  - `--lr` : utiliser une valeur nettement plus petite qu'un entraînement
    from scratch (1e-5 à 5e-5) pour ne pas détruire les poids pré-appris.
  - `--batch_size` / `--microbatch` : ajuster selon la VRAM disponible;
    activer `--use_fp16 True` et `--use_checkpoint True` pour un gros
    modèle (256x256_diffusion_uncond, ~550M paramètres) sur GPU limité.
  - `--class_cond` : False si le dataset satellite n'est pas étiqueté par
    classe (cas général).

Ce script utilise MPI (via mpi4py, déjà requis par image_datasets.py et
dist_util.py). Sur une seule machine/un seul GPU, il peut être lancé
directement avec `python scripts/image_train.py ...` (MPI.COMM_WORLD a
alors une taille de 1); pour du multi-GPU/multi-noeud, lancer via
`mpiexec -n <N> python scripts/image_train.py ...`.
"""

import argparse
import sys
import os

# Permet de lancer `python scripts/image_train.py` depuis la racine du
# dépôt sans avoir à installer guided_diffusion comme package.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creation du modele et du schedule de diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creation du dataloader (dataset satellite)...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("debut de l'entrainement / fine-tuning...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    # Defaults orientes "fine-tuning satellite RGB 256x256" : lr faible,
    # petit batch_size (a adapter a la VRAM disponible), resume_checkpoint
    # vide par defaut (entrainement from scratch) a surcharger en ligne de
    # commande avec le chemin du checkpoint pre-entraine (ex:
    # model_zoo/256x256_diffusion_uncond.pt) pour faire du fine-tuning.
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=2e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 desactive le micro-batching (microbatch = batch_size)
        ema_rate="0.9999",  # peut aussi etre une liste separee par des virgules, ex: "0.9999,0.999"
        log_interval=50,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    # image_size=256, class_cond=False et learn_sigma=False par defaut ici;
    # a surcharger en ligne de commande pour matcher le checkpoint fine-tune
    # (ex: --learn_sigma True --num_channels 256 --num_res_blocks 2
    # --attention_resolutions 8,16,32 pour 256x256_diffusion_uncond).
    defaults.update(model_and_diffusion_defaults())
    defaults["image_size"] = 256
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
