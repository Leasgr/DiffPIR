# main_ddpir.py
#
# Point d'entrée principal de DiffPIR (version configurable via fichier YAML).
# Supporte les trois tâches : super-résolution ('sr'), défloutage ('deblur'), inpainting ('inpaint').
# Utilisation : python main_ddpir.py --opt options/config.yaml
#
# La configuration est chargée depuis un fichier YAML passé via --opt.
# Voir parse_args_and_config() pour la liste complète des paramètres YAML attendus.
#
# Différence avec les scripts autonomes (main_ddpir_deblur.py, etc.) :
#   - Utilise un DataLoader PyTorch (batch processing) pour efficacité
#   - Toutes les tâches sont dans un seul script, sélectionnées par config.task
#   - Les hyperparamètres sont dans un fichier YAML, pas codés en dur

import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_resizer import Resizer
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from utils.utils_inpaint import mask_generator
from scipy import ndimage

from functools import partial

import yaml
import argparse
import shutil
import random

from torch.utils.data import Dataset, DataLoader

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

class CustomDataset(Dataset):
    '''Dataset PyTorch qui charge et dégrade les images à la volée.
    Prend en charge les trois tâches selon config.task.
    Retourne (img_H, img_L, img_name, k, mask) pour chaque image.
    '''
    def __init__(self, img_paths, config):
        self.img_paths = img_paths
        self.config = config

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # --------------------------------
        # Chargement du noyau de dégradation
        # --------------------------------

        if self.config.task == "sr":
            kernels = hdf5storage.loadmat(os.path.join(self.config.cwd, 'kernels', 'kernels_bicubicx234.mat'))['kernels']
            k_index = self.config.sf-2 if self.config.sf < 5 else 2
            k = kernels[0, k_index].astype(np.float64)
        elif self.config.task == 'deblur':
            if self.config.use_DIY_kernel:
                np.random.seed(seed=idx*10)  # for reproducibility of blur kernel for each image
                if self.config.blur_mode == 'Gaussian':
                    kernel_std_i = self.config.kernel_std * np.abs(np.random.rand()*2+1)
                    kernel = GaussialBlurOperator(kernel_size=self.config.kernel_size, intensity=kernel_std_i, device=self.config.device)
                elif self.config.blur_mode == 'motion':
                    kernel = MotionBlurOperator(kernel_size=self.config.kernel_size, intensity=self.config.kernel_std, device=self.config.device)
                k_tensor = kernel.get_kernel().to(self.config.device, dtype=torch.float)
                k = k_tensor.clone().detach().cpu().numpy()       #[0,1]
                k = np.squeeze(k)
                k = np.squeeze(k)
            else:
                k_index = 0
                kernels = hdf5storage.loadmat(os.path.join(self.config.cwd, 'kernels', 'Levin09.mat'))['kernels']
                k = kernels[0, k_index].astype(np.float32)
        else:
            k = torch.ones((1,1,1,1)) # dummy kernel
        
        # --------------------------------
        # get img_L
        # --------------------------------

        img_name= os.path.basename(img_path)
        img_H = util.imread_uint(img_path, n_channels=self.config.n_channels)
        img_H = util.modcrop(img_H, self.config.sf)  # modcrop
        if self.config.task == "sr":
            img_H_tensor = np.transpose(img_H, (2, 0, 1))
            img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(self.config.device)
            img_H_tensor = img_H_tensor / 255 
            down_sample = Resizer(img_H_tensor.shape, 1/self.config.sf).to(self.config.device)
            if self.config.sr_mode == 'blur':
                img_L = util.imresize_np(util.uint2single(img_H), 1/self.config.sf)
            elif self.config.sr_mode == 'cubic':
                img_L = down_sample(img_H_tensor)
                img_L = img_L.cpu().numpy()       #[0,1]
                img_L = np.squeeze(img_L)
                if img_L.ndim == 3:
                    img_L = np.transpose(img_L, (1, 2, 0))
            mask = np.ones_like(img_L)
        elif self.config.task == 'deblur':
            # mode='wrap' is important for analytical solution
            img_L = ndimage.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
            img_L = util.uint2single(img_L)
            mask = np.ones_like(img_L)
        elif self.config.task == 'inpaint':
            if self.config.load_mask:
                mask = util.imread_uint(self.config.mask_path, n_channels=self.config.n_channels).astype(bool)
            else:
                mask_gen = mask_generator(mask_type=self.config.mask_type, mask_len_range=self.config.mask_len_range, mask_prob_range=self.config.mask_prob_range)
                mask = mask_gen(util.uint2tensor4(img_H)).numpy()
                mask = np.squeeze(mask)
                mask = np.transpose(mask, (1, 2, 0))
            img_L = img_H * mask  / 255.   #(256,256,3)         [0,1]

        img_L = img_L * 2 - 1
        img_L += np.random.normal(0, self.config.noise_level_img * 2, img_L.shape) # add AWGN
        img_L = img_L / 2 + 0.5

        # Return images names and kernels
        return img_H, img_L, img_name, k, mask

class Config:
    '''Convertit un dictionnaire YAML imbriqué en objet Python avec attributs accessibles.
    Ex : config['task'] → config.task, config['blur']['mode'] → config.blur.mode.
    '''
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def parse_args_and_config():
    '''
    Charge le fichier YAML passé via --opt et calcule les chemins et paramètres dérivés.

    Paramètres YAML attendus (liste complète) :
    ─────────────────────────────────────────────
    Général :
      task             : tâche à effectuer ('sr', 'deblur', 'inpaint')
      model_name       : nom du checkpoint ('diffusion_ffhq_10m' ou '256x256_diffusion_uncond')
      testset_name     : nom du dossier de test dans testsets/
      seed             : graine aléatoire pour la reproductibilité
      n_channels       : nombre de canaux (3 pour RGB)
      cwd              : répertoire racine ('' = courant)
      batch_size       : taille du batch pour le DataLoader

    Bruit :
      noise_level_img  : niveau de bruit AWGN sur l'image dégradée (en [0,255], converti en [0,1])
      noise_init_img   : niveau de bruit pour initialiser x_t ('max' = bruit pur, ou valeur float)

    Diffusion :
      num_train_timesteps : nombre de pas T de la diffusion (1000, fixe)
      beta_start          : début du schedule linéaire β (0.0001)
      beta_end            : fin du schedule linéaire β (0.02)
      iter_num            : nombre de pas de diffusion inverse (NFE, ex: 20-100)
      iter_num_U          : nombre d'itérations internes par pas (défaut 1)
      skip_type           : espacement des pas ('uniform' ou 'quad')
      skip_noise_model_t  : si True, calcule noise_model_t depuis noise_level_model

    Algorithme DiffPIR :
      generate_mode    : 'DiffPIR', 'DPS_y0', 'DPS_yt', 'repaint', 'vanilla'
      model_output_type: 'pred_xstart', 'pred_x_prev', 'epsilon', 'score'
      sub_1_analytic   : True = solveur FFT, False = gradient de premier ordre
      ddim_sample      : True = DDIM, False = DDPM
      eta              : stochasticité DDIM (0 = déterministe, 1 = DDPM)
      zeta             : mélange bruit déterministe/stochastique (0 à 1)
      lambda_          : poids du terme de données (régularisation HQS)
      guidance_scale   : force de la correction par les données (défaut 1.0)

    Métriques :
      calc_LPIPS       : calculer la métrique LPIPS (True/False)

    Super-résolution (task='sr') :
      sf               : facteur d'échelle (2, 3 ou 4)
      sr_mode          : 'blur' (dégradation classique) ou 'cubic' (bicubique)
      inIter           : itérations IBP pour sr_mode='cubic'
      gamma            : pas de l'IBP pour sr_mode='cubic'

    Défloutage (task='deblur') :
      blur_mode        : 'Gaussian' ou 'motion'
      kernel_size      : taille du noyau en pixels
      use_DIY_kernel   : True = noyau aléatoire, False = noyau fixe depuis Levin09.mat

    Inpainting (task='inpaint') :
      mask_type        : 'box', 'random', 'both', 'extreme'
      mask_len_range   : [min, max] taille du rectangle (pour 'box')
      mask_prob_range  : [min, max] probabilité de masquage (pour 'random')
      load_mask        : True = charge un masque depuis mask_path
      mask_path        : chemin vers l'image masque (si load_mask=True)

    Sauvegarde :
      save_E           : sauvegarder les images restaurées
      save_L           : sauvegarder les images dégradées
    ─────────────────────────────────────────────
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="Path to option YMAL file.")
    args = parser.parse_args()
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt

    config.noise_level_img = config.noise_level_img / 255.  # convertit de [0,255] en [0,1]
    config.noise_level_model = config.noise_level_img
    config.sigma = max(0.001, config.noise_level_img)  # évite division par zéro dans rho
    # Chemins fixes (ne pas modifier)
    config.model_zoo = os.path.join(config.cwd, 'model_zoo')
    config.testsets = os.path.join(config.cwd, 'testsets')
    config.results = os.path.join(config.cwd, 'results')
    config.result_name = f'{config.testset_name}_{config.task}_{config.generate_mode}_{config.model_name}_sigma{config.noise_level_img}_NFE{config.iter_num}_eta{config.eta}_zeta{config.zeta}_lambda{config.lambda_}'
    if config.task == "sr":
        config.result_name += f'_{config.sr_mode}{str(config.sf)}'
    elif config.task == "deblur":
        config.result_name += f'_blurmode_{config.blur_mode}'
        config.kernel_std = 3.0 if config.blur_mode == 'Gaussian' else 0.5
    elif config.task == "inpaint":
        config.result_name += f'_mask_type_{config.mask_type}'
        assert config.generate_mode in ['DiffPIR', 'repaint', 'vanilla']

    config.model_path = os.path.join(config.model_zoo, config.model_name+'.pt')
    config.L_path = os.path.join(config.testsets, config.testset_name)
    config.E_path = os.path.join(config.results, config.result_name)
    util.mkdir(config.E_path)

    # Fixe toutes les graines aléatoires pour la reproductibilité
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    return config


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = parse_args_and_config()
    config.device = device
    L_paths = util.get_image_paths(config.L_path)

    # ----------------------------------------
    # Calendrier de bruit (noise schedule) linéaire DDPM
    # β_t croit linéairement de beta_start à beta_end sur num_train_timesteps pas.
    # reduced_alpha_cumprod[t] ≈ σ équivalente à l'instant t (utilisée pour mapper
    # un niveau de bruit vers un pas de temps discret).
    # ----------------------------------------
    betas = np.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    # σ équivalente sur l'image : ratio √(1-ᾱ_t)/√ᾱ_t, croissant avec t
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)

    # noise_model_t : pas à partir duquel on désactive la correction par les données.
    # Quand σ_résiduelle < σ_image, le modèle n'a plus d'avantage à corriger.
    if config.skip_noise_model_t:
        config.noise_model_t = utils_model.find_nearest(reduced_alpha_cumprod, 2 * config.noise_level_model)
    else:
        config.noise_model_t = 0  # désactivé : correction active jusqu'au bout

    # t_start : pas de temps initial (point de départ de la diffusion inverse).
    # 'max' = départ depuis bruit pur (t=999), explorant tout l'espace latent.
    # Valeur numérique = départ depuis le niveau de bruit correspondant (plus rapide).
    if config.noise_init_img == 'max':
        config.t_start = config.num_train_timesteps - 1
    else:
        config.t_start = utils_model.find_nearest(reduced_alpha_cumprod, 2 * config.noise_init_img / 255)

    # set up logger
    logger_name = config.result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(config.E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load datasets
    # ----------------------------------------
    # Assuming you have L_paths as your list of image file paths
    dataset = CustomDataset(L_paths, config)
    # Define batch size and create a DataLoader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    # Architecture U-Net selon le checkpoint :
    #   diffusion_ffhq_10m      → modèle léger (≈10M params), adapté aux visages 256×256
    #   256x256_diffusion_uncond → modèle large (≈550M params), adapté à ImageNet
    model_config = dict(
            model_path=config.model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if config.model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=config.model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if config.generate_mode != 'DPS_y0':
        # DPS_y0 nécessite les gradients du modèle pour la guidance.
        # Pour tous les autres modes, on désactive les gradients pour économiser la mémoire.
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    # save config
    shutil.copyfile(config.opt, os.path.join(config.E_path, os.path.basename('config.yaml')))

    # ----------------------------------------
    # main function
    # ----------------------------------------

    def test_rho(config):
        '''Exécute l'inférence DiffPIR pour les hyperparamètres courants (lambda_, zeta, etc.).
        Cette fonction est appelée en boucle pour balayer différentes valeurs de lambda_.
        '''
        parameters = f'eta:{config.eta}, zeta:{config.zeta}, lambda:{config.lambda_}, guidance_scale:{config.guidance_scale}'
        parameters = parameters + f', inIter:{config.inIter}, gamma:{config.gamma}' if (config.task == "sr" and config.sr_mode == 'cubic') else parameters
        logger.info(parameters)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['psnr_y'] = []
        if config.calc_LPIPS:
            test_results['lpips'] = []
        total_num = 0
        for idx, batch in enumerate(dataloader):
            model_out_type = config.model_output_type
            batch_size = batch[0].shape[0]
            C, H, W = batch[0].shape[3], batch[0].shape[1], batch[0].shape[2]
            img_H, img_L, names, k, mask = batch
            # Conversion en numpy pour les opérations spatiales
            img_H = img_H.numpy()
            img_L = img_L.numpy()
            k = k.numpy()
            mask = mask.numpy()
            
            # --------------------------------
            # (2) Calcul des rhos et sigmas pour chaque pas de temps
            # rho[t] = λσ²/σ_k[t]² : poids du terme de données dans HQS à l'instant t.
            # Un rho grand → la solution des données domine (forte fidélité à y).
            # Un rho petit → le prior du modèle domine (liberté hallucinatoire).
            # --------------------------------

            sigmas = []    # niveaux de bruit décroissants (de t=999 à t=0)
            sigma_ks = []  # variance conditionnelle de x0 sachant x_t
            rhos = []      # paramètre de régularisation HQS à chaque pas
            for i in range(config.num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[config.num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart' and config.generate_mode == 'DiffPIR':
                    # Formule DiffPIR : σ_k = √(1-ᾱ_t)/√ᾱ_t (variance du décodeur)
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                else:
                    # Formule alternative (pred_x_prev) : σ_k = √(β_t/α_t)
                    sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                rhos.append(config.lambda_*(config.sigma**2)/(sigma_ks[i]**2))

            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(config.device), torch.tensor(sigmas).to(config.device), torch.tensor(sigma_ks).to(config.device)
            
            # --------------------------------
            # (3) Initialisation de x_t et pré-calcul FFT
            # --------------------------------
            y = util.single2tensor4_batch(img_L).to(config.device)  # observation y ∈ [0,1], (B,3,H,W)

            if config.task == "sr":
                # Opérateur de dégradation : Resizer implémente le sous-échantillonnage bicubique
                degrade_op = Resizer((batch_size, C, H, W), 1/config.sf).to(config.device)
                # Initialisation bicubique : x interpolé à la résolution HR (meilleur point de départ)
                x = F.interpolate(torch.from_numpy(img_L).permute(0, 3, 1, 2),
                                  size=(img_L.shape[1]*config.sf, img_L.shape[2]*config.sf),
                                  mode='bicubic', align_corners=False).to(config.device)
                if config.sr_mode == 'cubic':
                    up_sample = partial(F.interpolate, scale_factor=config.sf)
            elif config.task == "deblur":
                util.imsave_batch(k*255.*200, names, config.E_path, 'motion_kernel_')
                k_4d = torch.from_numpy(k).to(device)
                k_4d = k_4d.unsqueeze(1)  # (B, 1, H, W) : noyau identique pour chaque canal
                x = y
                # Opérateur de flou : convolution avec padding réflexif (évite les artefacts de bords)
                def degrade_op(x):
                    x = x / 2 + 0.5  # convertit en [0,1] pour la convolution
                    pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                    x_blurs = []
                    for i in range(x.shape[0]):
                        x_blurs.append(F.conv2d(pad_2d(x[i:i+1]), k_4d))
                    return torch.cat(x_blurs, 0)
            elif config.task == 'inpaint':
                img_L = img_L * mask
                mask = util.single2tensor4_batch(mask.astype(np.float32)).to(device)
                x = y * mask  # pixels inconnus initialisés à 0

            # Bruite x au niveau t_start : mélange x avec du bruit gaussien
            # pour cohérence avec la distribution q(x_t | x_0) à l'instant t_start.
            x = sqrt_alphas_cumprod[config.t_start] * (2*x-1) + sqrt_1m_alphas_cumprod[config.t_start] * torch.randn_like(x)

            if config.task in ['sr', 'deblur']:
                # Pré-calcul FFT des matrices nécessaires à data_solution (fait une seule fois)
                k_tensor = util.single2tensor4_batch(np.expand_dims(k, 3)).to(config.device)
                FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, config.sf)

            # --------------------------------
            # (4) Boucle principale de diffusion inverse DiffPIR
            # Alterne : (Étape 1) débruitage par le U-Net → (Étape 2) correction par données
            # --------------------------------

            # Construction de la séquence de pas de temps échantillonnés
            skip = config.num_train_timesteps // config.iter_num
            if config.skip_type == 'uniform':
                # Espacement régulier : t = 0, skip, 2*skip, ...
                seq = [i*skip for i in range(config.iter_num)]
                if skip > 1:
                    seq.append(config.num_train_timesteps-1)
            elif config.skip_type == "quad":
                # Espacement quadratique : concentre plus de pas aux niveaux de bruit élevés
                seq = np.sqrt(np.linspace(0, config.num_train_timesteps**2, config.iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::max(len(seq)//10,1)]
            if progress_seq[-1] != seq[-1]:
                progress_seq.append(seq[-1])

            # Boucle de diffusion inverse (de t_start vers t=0)
            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                # Pas de temps t_i correspondant au niveau de bruit courant
                t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
                if t_i > config.t_start:
                    continue  # saute les pas au-dessus du point de départ
                # Itérations internes (iter_num_U > 1 : stratégie RepaintPaper)
                for u in range(config.iter_num_U):
                    # --------------------------------
                    # Étape 1 : pas de diffusion inverse → estime x0
                    # --------------------------------

                    if config.task == "inpaint":
                        if config.generate_mode == 'repaint':
                            # RePaint : remplace les pixels connus par y bruité au niveau t_i
                            # Assure la cohérence des pixels observés tout au long de la trajectoire.
                            x = (sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                                    + (1-mask) * x

                        # solve equation 6b with one reverse diffusion step
                        if model_out_type == 'pred_xstart':
                            x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                    model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                        else:
                            x = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                    model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                        # x = utils_model.test_mode(model_fn, x, mode=0, refield=32, min_size=256, modulo=16, noise_level=sigmas[i].cpu().numpy()*255)
                    else:
                        ### solve equation 6b with one reverse diffusion step
                        if 'DPS' in config.generate_mode:
                            x = x.requires_grad_()
                            xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start', \
                                        model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                        else:
                            x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                    model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                        # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                        #       model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                    # --------------------------------
                    # Étape 2 : correction par les données (sous-problème HQS)
                    # --------------------------------

                    if seq[i] != seq[-1]:
                        if config.generate_mode == 'DiffPIR':
                            if config.sub_1_analytic:
                                if model_out_type == 'pred_xstart':
                                    tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                    # Active la correction seulement quand σ_résiduelle > σ_image
                                    if i < config.num_train_timesteps-config.noise_model_t:
                                        if config.task == "inpaint":
                                            # Solution close-form : (M⊙y + τ*x0) / (M + τ)
                                            x0_p = (mask * (2*y-1) + tau * x0).div(mask + tau)
                                            x0 = x0 + config.guidance_scale * (x0_p-x0)
                                        elif config.task == "deblur" or config.sr_mode == 'blur':
                                            # Déconvolution FFT (voir utils_sisr.data_solution)
                                            x0_p = x0 / 2 + 0.5
                                            x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, config.sf)
                                            x0_p = x0_p * 2 - 1
                                            x0 = x0 + config.guidance_scale * (x0_p-x0)
                                        elif config.sr_mode == 'cubic':
                                            # IBP (Iterative Back Projection) pour SR bicubique :
                                            # x0 ← x0 + γ * A^T(y - A(x0)) / (1+ρ)
                                            # inIter itérations, γ est le pas d'IBP.
                                            for _ in range(config.inIter):
                                                x0 = x0 / 2 + 0.5
                                                x0 = x0 + config.gamma * up_sample((y - degrade_op(x0))) / (1+rhos[t_i])
                                                x0 = x0 * 2 - 1
                                    else:
                                        # Niveau de bruit trop bas → repasse en mode pred_x_prev
                                        model_out_type = 'pred_x_prev'
                                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                                model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                                        # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                                        #       model_out_type=model_out_type, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                                        pass
                                elif model_out_type == 'pred_x_prev' and config.task == "inpaint":
                                    # when noise level less than given image noise, skip
                                    if i < config.num_train_timesteps-config.noise_model_t: 
                                        x = (mask * (2*y-1) + tau * x0).div(mask + tau) # y-->yt ?
                                    else:
                                        pass
                            else:
                                #TODO first order solver for inpainting
                                x0 = x0.requires_grad_()
                                # first order solver
                                measurement = y if config.task == "deblur" else 2*y-1
                                norm_grad, norm = utils_model.grad_and_value(operator=degrade_op,x=x0, x_hat=x0, measurement=measurement)
                                                    
                                x0 = x0 - norm_grad * norm / (rhos[t_i]) 
                                x0 = x0.detach_()
                                pass                          
                        elif 'DPS' in config.generate_mode:                    
                            if config.generate_mode == 'DPS_y0':
                                measurement = y if config.task == "deblur" else 2*y-1
                                norm_grad, norm = utils_model.grad_and_value(operator=degrade_op,x=x, x_hat=x0, measurement=measurement)
                                #norm_grad, norm = utils_model.grad_and_value(operator=degrade_op,x=xt, x_hat=x0, measurement=2*y-1)    # does not work
                                x = xt - norm_grad * 1. #norm / (2*rhos[t_i]) 
                                x = x.detach_()
                                pass
                            elif config.generate_mode == 'DPS_yt':
                                y_t = sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(y) # add AWGN [-1,1]
                                measurement = y_t/2 + 0.5 if config.task == "deblur" else y_t
                                #norm_grad, norm = utils_model.grad_and_value(operator=degrade_op,x=x, x_hat=xt, measurement=measurement)    # no need to use
                                norm_grad, norm = utils_model.grad_and_value(operator=degrade_op,x=xt, x_hat=xt, measurement=measurement)
                                x = xt - norm_grad * config.lambda_ * norm / (rhos[t_i]) * 0.35
                                x = x.detach_()
                                pass
                        
                    # --------------------------------
                    # Re-bruitage vers t_{i-1} : formule DDIM généralisée (Eq. 12 de l'article)
                    # x_{t-1} = √ᾱ_{t-1}*x0 + √(1-ζ)*(direction_DDIM + bruit_eta) + √ζ*bruit_pur
                    # --------------------------------
                    if ((config.task == "inpaint" or config.generate_mode == 'DiffPIR') and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == config.iter_num_U-1):
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i+1]].cpu().numpy())
                        # ε̂ estimé depuis x_t et x0 prédit
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        # η contrôle la variance du bruit DDIM (0 = déterministe)
                        eta_sigma = config.eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                        # ζ mélange la direction déterministe avec du bruit i.i.d. pur
                        x = sqrt_alphas_cumprod[t_im1] * x0 + \
                            np.sqrt(1-config.zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                + eta_sigma * torch.randn_like(x)) + \
                            np.sqrt(config.zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                    else:
                        pass

                    # Retour en arrière pour les itérations internes (iter_num_U > 1) :
                    # ré-bruite x_{t-1} vers x_t pour repartir de x_t à l'itération suivante.
                    if u < config.iter_num_U-1 and seq[i] != seq[-1]:
                        # Formule exacte du processus forward q(x_t | x_{t-1}) via la décomposition
                        # de la variance : plus stable numériquement que l'ajout direct de β_t.
                        sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                        x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                                sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

                # save the process
                x_0 = (x/2+0.5)

            total_num += batch_size

            # recover conditional part
            if config.task == "inpaint" and config.generate_mode in ['repaint','DiffPIR']:
                x[mask.to(torch.bool)] = (2*y-1)[mask.to(torch.bool)]

            # --------------------------------
            # (3) img_E
            # --------------------------------

            img_E = util.tensor2uint_batch(x_0)
            img_H_tensor = np.transpose(img_H, (0, 3, 1, 2))
            img_H_tensor = torch.from_numpy(img_H_tensor).to(device)
            img_H_tensor = img_H_tensor / 255 * 2 -1
            psnr = util.calculate_psnr_batch(x_0.detach()*2-1, img_H_tensor)
            test_results['psnr'].append(psnr * batch_size)
            
            if config.calc_LPIPS:
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score * batch_size)
                logger.info(f"batch{idx+1:->4d}--> PSNR: {psnr:.4f}dB; LPIPS: {lpips_score:.4f}; ave LPIPS: {sum(test_results['lpips']) / total_num:.4f}")
            else:
                logger.info(f'batch{idx+1:->4d}--> PSNR: {psnr:.4f}dB')

            if config.save_E:
                # util.imsave(img_E, os.path.join(config.E_path, f"{img_name}_x{sf}_{config.model_name+ext}"))
                util.imsave_batch(img_E, names, config.E_path, f"{config.model_name}_x{config.sf}_lambda{config.lambda_:.4f}_zeta{config.zeta:.4f}_")

            if config.n_channels == 1:
                img_H = img_H.squeeze()

            # --------------------------------
            # (4) img_L
            # --------------------------------

            img_L = util.single2uint(img_L).squeeze()

            if config.save_L:
                util.imsave_batch(img_L, names, config.E_path, f"LR_x{config.sf}_")

            if config.n_channels == 3:
                img_E_y = util.rgb2ycbcr_batch(x_0.detach()*2-1, only_y=True)
                img_H_y = util.rgb2ycbcr_batch(img_H_tensor, only_y=True)
                psnr_y = util.calculate_psnr_batch(img_E_y, img_H_y)
                test_results['psnr_y'].append(psnr_y * batch_size)
            
        # --------------------------------
        # Average PSNR and LPIPS for all images
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / total_num
        logger.info(f'-----------> Average PSNR(RGB) of ({config.testset_name}) scale factor: ({config.sf}), sigma: ({config.noise_level_model:.3f}): {ave_psnr:.4f} dB')
        test_results_ave['psnr_sf'].append(ave_psnr)

        if config.n_channels == 3:  # RGB image
            ave_psnr_y = sum(test_results['psnr_y']) / total_num
            logger.info(f'-----------> Average PSNR(Y) of ({config.testset_name}) scale factor: ({config.sf}), sigma: ({config.noise_level_model:.3f}): {ave_psnr_y:.4f} dB')
            test_results_ave['psnr_y_sf'].append(ave_psnr_y)

        if config.calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / total_num
            logger.info(f'-----------> Average LPIPS of ({config.testset_name}) scale factor: ({config.sf}), sigma: ({config.noise_level_model:.3f}): {ave_lpips:.4f}')
            test_results_ave['lpips'].append(ave_lpips)    
        return test_results_ave


    test_results_ave = OrderedDict()
    test_results_ave['psnr_sf'] = []
    test_results_ave['psnr_y_sf'] = []
    if config.calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        test_results_ave['lpips'] = []


    if config.task == "sr":
        ### SR
        for sf in [4]:
            config.sf = sf
            border = sf
            logger.info('--------- sf:{:>1d} ---------'.format(sf))

            # experiments
            lambdas = [config.lambda_*i for i in range(2,13)]
            for lambda_ in lambdas:
                for zeta_i in [config.zeta]:
                    config.lambda_ = lambda_
                    config.zeta = zeta_i
                    test_results_ave = test_rho(config)
    elif config.task == "deblur":
        ### Deblur
        border = 0
        lambdas = [config.lambda_*i for i in range(7,8)]
        for lambda_ in lambdas:
            for zeta_i in [config.zeta*i for i in range(3,4)]:
                config.lambda_ = lambda_
                config.zeta = zeta_i
                test_results_ave = test_rho(config)
    elif config.task == "inpaint":
        ### Inpaint
        border = 0
        lambdas = [config.lambda_*i for i in range(1,2)]
        for lambda_ in lambdas:
            #for zeta_i in [0,0.3,0.8,0.9,1.0]:
            for zeta_i in [config.zeta*i for i in range(1,2)]:
                config.lambda_ = lambda_
                config.zeta = zeta_i
                test_results_ave = test_rho(config)



    # ---------------------------------------
    # Average PSNR and LPIPS for all sf and parameters
    # ---------------------------------------

    ave_psnr_sf = sum(test_results_ave['psnr_sf']) / len(test_results_ave['psnr_sf'])
    logger.info(f'-----------> Average PSNR of ({config.testset_name}) {ave_psnr_sf:.4f} dB')
    if config.n_channels == 3:
        ave_psnr_y_sf = sum(test_results_ave['psnr_y_sf']) / len(test_results_ave['psnr_y_sf'])
        logger.info(f'-----------> Average PSNR-Y of ({config.testset_name}) {ave_psnr_y_sf:.4f} dB')
    if config.calc_LPIPS:
        ave_lpips_sf = sum(test_results_ave['lpips']) / len(test_results_ave['lpips'])
        logger.info(f'-----------> Average LPIPS of ({config.testset_name}) {ave_lpips_sf:.4f}')

if __name__ == '__main__':

    main()
