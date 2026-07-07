# utils/utils_model.py
#
# Fonctions utilitaires pour l'interface entre DiffPIR et le modèle de diffusion.
#
# Fonctions principales :
#   model_fn        : appelle le modèle de diffusion et retourne la sortie demandée
#   grad_and_value  : calcule le gradient de ||y - A(x_hat)||₂ par rapport à x
#   create_argparser: construit le parseur de paramètres pour le U-Net
#   test_mode       : wrapper pour tester avec padding/split/augmentation (non utilisé en inférence standard)
#   find_nearest    : trouve l'indice du pas de temps correspondant à un niveau de bruit donné

# -*- coding: utf-8 -*-
import numpy as np
import torch
from utils import utils_image as util
from functools import partial

from guided_diffusion.script_util import add_dict_to_argparser
import argparse

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


def test_mode(model_fn, model_diffusion, L, mode=0, refield=32, min_size=256, sf=1, modulo=1, noise_level=0, vec_t=None, \
        model_out_type='pred_xstart', diffusion=None, ddim_sample=False, alphas_cumprod=None):
    '''
    Wrapper permettant de tester le modèle avec différentes stratégies de gestion de la mémoire.
    Utile pour les grandes images qui ne tiennent pas en VRAM d'un seul coup.

    Modes disponibles :
      (0) normal    : test direct sans modification
      (1) pad       : padding pour que les dimensions soient multiples de `modulo`
      (2) split     : découpe l'image récursivement (recommandé pour grandes images)
      (3) x8        : moyenne sur 8 augmentations (rotations/symétries)
      (4) split + x8 : combinaison des modes 2 et 3
      (5) onesplit  : découpe en 4 quadrants une seule fois (plus rapide que mode 2)

    Paramètres :
      refield   : champ récepteur effectif du réseau (32 suffit pour le U-Net utilisé)
      min_size  : taille minimale d'un patch traité directement (256x256)
      modulo    : les dimensions doivent être multiples de cette valeur (16 pour le U-Net)
    '''

    model = partial(model_fn, model_diffusion=model_diffusion, diffusion=diffusion, ddim_sample=False, alphas_cumprod=alphas_cumprod)

    if mode == 0:
        E = test(model, L, noise_level, vec_t, model_out_type)
    elif mode == 1:
        E = test_pad(model, L, modulo, noise_level, vec_t, model_out_type)
    elif mode == 2:
        E = test_split(model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type)
    elif mode == 3:
        E = test_x8(model, L, modulo, noise_level, vec_t, model_out_type)
    elif mode == 4:
        E = test_split_x8(model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type)
    elif mode == 5:
        E = test_onesplit(model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type)
    return E


'''
# ---------------------------------------
# normal (0)
# ---------------------------------------
'''


def test(model, L, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
    return E


'''
# ---------------------------------------
# pad (1)
# ---------------------------------------
'''


def test_pad(model, L, modulo=16, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h/modulo)*modulo-h)
    paddingRight = int(np.ceil(w/modulo)*modulo-w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
    E = E[..., :h, :w]
    return E


'''
# ---------------------------------------
# split (function)
# ---------------------------------------
'''


def test_split_fn(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    '''
    Découpe récursive de l'image en 4 quadrants avec recouvrement (overlap = refield).
    Le recouvrement évite les artefacts aux jointures entre quadrants.

    Paramètres :
      refield  : taille du recouvrement en pixels (= champ récepteur effectif du réseau)
      min_size : taille minimale pour traiter un patch directement sans redécouper
      sf       : facteur d'échelle (pour SR ; 1 sinon)
      modulo   : garantit que les dimensions des patches sont multiples de cette valeur
    '''
    h, w = L.size()[-2:]
    if h*w <= min_size**2:
        L = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(L)
        E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
        E = E[..., :h*sf, :w*sf]
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)
        Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

        if h * w <= 4*(min_size**2):
            Es = [model(Ls[i], noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(4)]
        else:
            Es = [test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    '''
    Découpe en 4 quadrants une seule fois (sans récursion).
    Plus rapide que test_split pour des images de taille moyenne.
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h//2//refield+1)*refield)
    bottom = slice(h - (h//2//refield+1)*refield, h)
    left = slice(0, (w//2//refield+1)*refield)
    right = slice(w - (w//2//refield+1)*refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = [model(Ls[i], noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
    E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
    E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
    E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



'''
# ---------------------------------------
# split (2)
# ---------------------------------------
'''


def test_split(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    E = test_split_fn(model, L, refield=refield, min_size=min_size, sf=sf, modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type)
    return E


'''
# ---------------------------------------
# x8 (3)
# ---------------------------------------
'''


def test_x8(model, L, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    # Moyenne sur 8 versions (4 rotations × 2 symétries) : réduit le biais directionnel du U-Net.
    E_list = [test_pad(model, util.augment_img_tensor(L, mode=i), modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


'''
# ---------------------------------------
# split and x8 (4)
# ---------------------------------------
'''


def test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1, noise_level=15, vec_t=None, model_out_type='pred_xstart'):
    E_list = [test_split_fn(model, util.augment_img_tensor(L, mode=i), refield=refield, min_size=min_size, sf=sf, modulo=modulo, noise_level=noise_level, vec_t=vec_t, model_out_type=model_out_type) for i in range(8)]
    for k, i in enumerate(range(len(E_list))):
        if i==3 or i==5:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=8-i)
        else:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


# ----------------------------------------
# Interface avec le modèle de diffusion
# ----------------------------------------

def find_nearest(array, value):
    '''Retourne l'indice de la valeur la plus proche dans `array`.
    Utilisé pour mapper un niveau de bruit σ vers le pas de temps t correspondant.'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def model_fn(x, noise_level, model_diffusion, vec_t=None, model_out_type='pred_xstart', \
        diffusion=None, ddim_sample=False, alphas_cumprod=None, **model_kwargs):
    '''
    Appelle le modèle de diffusion pour un pas de débruitage et retourne la sortie demandée.

    Paramètres :
      x              : image bruitée courante x_t, forme (B,C,H,W) ∈ [-1,1]
      noise_level    : niveau de bruit σ en [0,255] → converti en pas de temps t
      model_diffusion: le U-Net DDPM chargé
      vec_t          : vecteur de pas de temps (si None, calculé depuis noise_level)
      model_out_type : type de sortie retournée :
        'pred_xstart'         → x̂₀ estimé directement (idéal pour DiffPIR)
        'pred_x_prev'         → x_{t-1} (un pas de diffusion inverse)
        'pred_x_prev_and_start' → (x_{t-1}, x̂₀) (pour DPS qui a besoin des deux)
        'epsilon'             → bruit résiduel estimé ε̂
        'score'               → score ∇_x log p(x_t)
      ddim_sample    : utiliser DDIM (True) ou DDPM (False) pour le pas de débruitage
      alphas_cumprod : produit cumulatif des α_t (nécessaire pour les calculs de σ)
    '''

    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)

    # Convertit le niveau de bruit en pas de temps discret t
    if not torch.is_tensor(vec_t):
        t_step = find_nearest(reduced_alpha_cumprod, (noise_level/255.))
        vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)

    if not ddim_sample:
        # Pas de diffusion inverse DDPM : échantillonne depuis p_θ(x_{t-1} | x_t)
        out = diffusion.p_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,   # clippe x̂₀ dans [-1,1] pour la stabilité
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
        )
    else:
        # Pas de diffusion inverse DDIM : déterministe (eta=0 dans model_fn, contrôlé ailleurs)
        out = diffusion.ddim_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
            eta=0,
        )

    # Extraction de la sortie selon le type demandé
    if model_out_type == 'pred_x_prev_and_start':
        return out["sample"], out["pred_xstart"]
    elif model_out_type == 'pred_x_prev':
        out = out["sample"]
    elif model_out_type == 'pred_xstart':
        out = out["pred_xstart"]
    elif model_out_type == 'epsilon':
        # Calcule ε̂ depuis x̂₀ : ε̂ = (x_t - √ᾱ_t * x̂₀) / √(1-ᾱ_t)
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
    elif model_out_type == 'score':
        # Score de Stein : s(x_t) = -ε̂ / √(1-ᾱ_t)
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
        out = - out / beta_prod_t ** (0.5)

    return out



'''
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
# _^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
'''


'''
# ---------------------------------------
# print
# ---------------------------------------
'''


# -------------------
# print model
# -------------------
def print_model(model):
    msg = describe_model(model)
    print(msg)


# -------------------
# print params
# -------------------
def print_params(model):
    msg = describe_params(model)
    print(msg)


'''
# ---------------------------------------
# information
# ---------------------------------------
'''


# -------------------
# model inforation
# -------------------
def info_model(model):
    msg = describe_model(model)
    return msg


# -------------------
# params inforation
# -------------------
def info_params(model):
    msg = describe_params(model)
    return msg


'''
# ---------------------------------------
# description
# ---------------------------------------
'''


# ----------------------------------------------
# model name and total number of parameters
# ----------------------------------------------
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# ----------------------------------------------
# parameters description
# ----------------------------------------------
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), name) + '\n'
    return msg

# ----------------------------------------
# Création du parseur de paramètres U-Net
# ----------------------------------------

def create_argparser(model_config):
    '''
    Construit le parseur de paramètres pour le U-Net DDPM.
    Les valeurs par défaut correspondent à la configuration de la plupart des checkpoints
    disponibles. model_config écrase les valeurs spécifiques au checkpoint chargé.

    Paramètres U-Net pertinents :
      num_channels        : largeur du réseau (128 pour FFHQ, 256 pour ImageNet)
      num_res_blocks      : nombre de blocs résiduels par niveau (1 ou 2)
      attention_resolutions : résolutions où l'attention est appliquée ("16" ou "8,16,32")
      learn_sigma         : si True, le modèle prédit aussi la variance (nécessaire pour IDDPM)
      dropout             : taux de dropout (0.1 par défaut)
      use_fp16            : utiliser la précision mixte float16 pour économiser la VRAM

    Paramètres de diffusion pertinents :
      diffusion_steps     : nombre de pas de temps (1000, fixe)
      noise_schedule      : type de schedule ('linear' par défaut)
      rescale_timesteps   : normalise t ∈ [0,1000] pour compatibilité entre checkpoints
    '''
    defaults = dict(
        clip_denoised=True,           # clippe x̂₀ dans [-1,1] à chaque pas
        num_samples=1,                # nombre d'échantillons à générer
        batch_size=1,
        use_ddim=False,               # utiliser DDIM (True) ou DDPM (False) par défaut
        model_path='',
        diffusion_steps=1000,         # nombre de pas T du processus de diffusion
        noise_schedule='linear',      # 'linear' ou 'cosine'
        num_head_channels=64,         # taille des têtes d'attention (canaux par tête)
        resblock_updown=True,         # utiliser les blocs résiduels pour up/down sampling
        use_fp16=False,               # float16 pour économiser la VRAM (risque de stabilité)
        use_scale_shift_norm=True,    # normalisation adaptative (AdaGN)
        num_heads=4,                  # nombre de têtes d'attention
        num_heads_upsample=-1,        # -1 = même que num_heads
        use_new_attention_order=False,
        timestep_respacing="",        # "" = pas de respace (1000 pas)
        use_kl=False,                 # loss KL (IDDPM) vs MSE (DDPM)
        predict_xstart=False,         # prédit x₀ (True) ou ε (False) pendant l'entraînement
        rescale_timesteps=False,      # normalise t pour compatibilité
        rescale_learned_sigmas=False,
        channel_mult="",              # multiplicateurs de canaux par niveau ("" = auto)
        learn_sigma=True,             # prédit aussi la variance σ_t (nécessaire pour IDDPM)
        class_cond=False,             # conditionnement par classe (False = non conditionné)
        use_checkpoint=False,         # gradient checkpointing pour économiser la VRAM
        image_size=256,               # résolution des images (fixe à 256)
        num_channels=128,             # largeur du U-Net (remplacé par model_config)
        num_res_blocks=1,             # blocs résiduels par niveau (remplacé par model_config)
        attention_resolutions="16",   # résolutions d'attention (remplacé par model_config)
        dropout=0.1,                  # dropout pendant l'entraînement (inactif en eval)
    )
    defaults.update(model_config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def grad_and_value(operator, x, x_hat, measurement):
    '''
    Calcule le gradient de la norme ||measurement - operator(x_hat)||₂ par rapport à x.
    Utilisé pour la guidance DPS et le solveur de premier ordre dans DiffPIR.

    Paramètres :
      operator    : opérateur de dégradation A (déconvolution, downsampling, masque...)
      x           : variable par rapport à laquelle on différencie (requiert requires_grad=True)
      x_hat       : entrée de l'opérateur (peut être différent de x pour DPS_y0)
      measurement : observation y

    Retourne :
      norm_grad : ∂||y - A(x_hat)||₂ / ∂x
      norm      : ||y - A(x_hat)||₂  (valeur scalaire de la norme)
    '''
    difference = measurement - operator(x_hat)
    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
    return norm_grad,  norm



if __name__ == '__main__':

    class Net(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model = Net()
    model = model.eval()
    print_model(model)
    print_params(model)
    x = torch.randn((2,3,400,400))
    torch.cuda.empty_cache()
    with torch.no_grad():
        for mode in range(5):
            y = test_mode(model, x, mode)
            print(y.shape)
