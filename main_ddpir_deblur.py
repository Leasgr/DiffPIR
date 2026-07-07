# main_ddpir_deblur.py
#
# Script autonome pour la déconvolution d'images (défloutage) via DiffPIR.
# À la différence de main_ddpir.py, les paramètres sont codés en dur dans main().
#
# Algorithme DiffPIR (Diffusion-based Plug-and-Play Image Restoration) :
#   Chaque itération t alterne entre :
#     Étape 1 (prior) : un pas de diffusion inverse via le U-Net  →  estime x0
#     Étape 2 (data)  : résolution d'un sous-problème au sens des moindres carrés
#                       (ici déconvolution FFT) pour corriger x0 vers x0_p
#   Le résultat x0_p est ensuite ré-bruitée au niveau t-1 pour l'itération suivante.

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
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from scipy import ndimage

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    # --- Bruit de mesure ---
    # Niveau de bruit AWGN ajouté à l'image dégradée (échelle [0,1]).
    # 12.75/255 ≈ σ=0.05, valeur typique pour tester la robustesse au bruit.
    noise_level_img         = 12.75/255.0
    # Niveau de bruit utilisé pour le modèle de diffusion (généralement == noise_level_img).
    noise_level_model       = noise_level_img

    # --- Modèle de diffusion ---
    # 'diffusion_ffhq_10m' : U-Net léger entraîné sur FFHQ (128 canaux, 1 bloc résiduel)
    # '256x256_diffusion_uncond' : U-Net plus large pour ImageNet (256 canaux, 2 blocs)
    model_name              = 'diffusion_ffhq_10m'

    # --- Jeu de test ---
    testset_name            = 'demo_test'  # 'imagenet_val' | 'ffhq_val' | 'demo_test'

    # --- Calendrier de diffusion ---
    # Nombre de pas de temps total du processus DDPM entraîné (fixe, ne pas modifier).
    num_train_timesteps     = 1000

    # --- Paramètres d'itération ---
    # iter_num : nombre d'évaluations de fonction (NFE), i.e. nombre de pas de
    # diffusion inverse effectués. Plus grand = meilleure qualité, plus lent.
    iter_num                = 100
    # iter_num_U : nombre d'itérations internes à chaque pas t (inspiré de RePaint).
    # Valeur > 1 améliore la cohérence sémantique mais multiplie le coût par iter_num_U.
    iter_num_U              = 1
    skip                    = num_train_timesteps//iter_num  # intervalle entre pas échantillonnés

    # --- Affichage et sauvegarde ---
    show_img                = False  # afficher les images pendant l'inférence
    save_L                  = True   # sauvegarder l'image dégradée (Low-quality)
    save_E                  = True   # sauvegarder l'image restaurée (Estimated)
    save_LEH                = False  # sauvegarder LR / Estimée / Haute-qualité côte à côte
    save_progressive        = False  # sauvegarder les images intermédiaires du processus
    border                  = 0

    # --- Paramètre sigma de conditionnement ---
    # sigma est la valeur de bruit associée à l'observation y dans le sous-problème HQS.
    # Doit être > 0 même si noise_level_img=0 pour éviter la division par zéro dans rho.
    sigma                   = max(0.001, noise_level_img)

    # --- Paramètre clé lambda_ ---
    # λ règle l'équilibre entre fidélité aux données et prior du modèle diffusion.
    # λ grand → forte fidélité aux données (image plus proche de y)
    # λ petit → forte influence du prior (image plus « générée »)
    lambda_                 = 1.0

    # sub_1_analytic : si True, résout l'étape 2 via FFT (solution exacte rapide).
    # Si False, utilise un gradient de premier ordre (plus lent, expérimental).
    sub_1_analytic          = True

    log_process             = False  # journaliser les valeurs min/max à chaque pas

    # ddim_sample : si True, utilise le sampler DDIM (déterministe) au lieu de DDPM.
    ddim_sample             = False

    # model_output_type : type de sortie demandée au U-Net.
    # 'pred_xstart' : prédit directement x0 (recommandé pour DiffPIR)
    # 'pred_x_prev' : prédit x_{t-1} (utilisé en repli si bruit résiduel trop faible)
    # 'epsilon'     : prédit le bruit ε
    # 'score'       : prédit le score ∇_x log p(x)
    model_output_type       = 'pred_xstart'

    # generate_mode : algorithme de restauration utilisé.
    # 'DiffPIR'  : méthode proposée dans l'article (step 2 via FFT ou gradient)
    # 'DPS_y0'   : Diffusion Posterior Sampling avec guidance sur x0
    # 'DPS_yt'   : Diffusion Posterior Sampling avec guidance sur y_t bruité
    # 'vanilla'  : diffusion sans guidance (génération libre, aucune contrainte de données)
    generate_mode           = 'DiffPIR'

    # skip_type : espacement des pas de temps échantillonnés.
    # 'uniform' : espacement régulier  (t = 0, skip, 2*skip, ...)
    # 'quad'    : espacement quadratique, concentre plus de pas aux grandes valeurs de t
    #             (bruit élevé), ce qui améliore la qualité perceptuelle.
    skip_type               = 'quad'

    # eta : paramètre DDIM contrôlant la stochasticité du pas de diffusion.
    # 0.0 = totalement déterministe (DDIM pur), 1.0 = stochastique (DDPM)
    eta                     = 0.0

    # zeta : mélange entre bruit déterministe (depuis ε) et bruit i.i.d. pur.
    # 0.0 = tout déterministe, 1.0 = tout stochastique (nouveau bruit tiré à chaque pas)
    # Influence la diversité vs la fidélité de la reconstruction.
    zeta                    = 0.1

    # guidance_scale : coefficient d'interpolation entre x0_prior et x0_data.
    # 1.0 = utilise entièrement la solution des données ; < 1.0 = mélange partiel.
    guidance_scale          = 1.0

    # calc_LPIPS : calculer la métrique LPIPS (perceptuelle) en plus du PSNR.
    calc_LPIPS              = True

    # --- Paramètres du noyau de flou ---
    # use_DIY_kernel : si True, génère un noyau aléatoire différent pour chaque image.
    # Si False, charge un noyau fixe depuis kernels/Levin09.mat.
    use_DIY_kernel          = True

    # blur_mode : type de flou appliqué à l'image.
    # 'Gaussian' : flou gaussien isotrope (tache)
    # 'motion'   : flou de mouvement (trajectoire aléatoire)
    blur_mode               = 'Gaussian'

    # kernel_size : taille du noyau de flou en pixels (doit être impair).
    kernel_size             = 61

    # kernel_std : écart-type du flou gaussien (pixels) ou intensité du mouvement.
    # Valeur effective : kernel_std * rand() pour varier par image (si use_DIY_kernel).
    kernel_std              = 3.0 if blur_mode == 'Gaussian' else 0.5

    sf                      = 1  # facteur d'échelle (1 = pas de SR, tâche de défloutage seul)
    task_current            = 'deblur'
    n_channels              = 3   # nombre de canaux (3 = RGB, fixe)
    cwd                     = ''  # répertoire racine (laisser vide si lancé depuis la racine)
    model_zoo               = os.path.join(cwd, 'model_zoo')
    testsets                = os.path.join(cwd, 'testsets')
    results                 = os.path.join(cwd, 'results')
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}_blurmode{blur_mode}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # Calendrier de bruit (noise schedule)
    # Suit le schedule linéaire de DDPM : β_t croit linéairement de beta_start à beta_end.
    # α_t = 1 - β_t,  ᾱ_t = ∏ α_s  (produit cumulatif)
    # √ᾱ_t et √(1-ᾱ_t) sont les coefficients de mélange signal/bruit à chaque pas t.
    # reduced_alpha_cumprod[t] = √(1-ᾱ_t)/√ᾱ_t  ≈ σ_équivalente sur l'image.
    # ----------------------------------------
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    # σ équivalente sur l'image pour chaque pas t (utilisée pour trouver t_start et noise_model_t)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)

    # noise_model_t : pas à partir duquel on arrête la correction de données (Eq. 6b).
    # Quand le niveau de bruit restant est inférieur au bruit de l'image observée,
    # la correction n'apporte plus rien et on revient à la diffusion libre.
    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0  # mis à 0 : la correction s'applique à tous les pas

    # t_start : pas de temps de départ de la diffusion inverse.
    # En partant de t=999 (bruit pur), on explore tout l'espace latent → plus de diversité.
    # On peut aussi partir d'un t plus bas pour démarrer depuis l'image dégradée bruitée.
    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img)
    t_start                 = num_train_timesteps - 1  # départ depuis le bruit pur (t=999)

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name)  # dossier des images dégradées
    E_path = os.path.join(results, result_name)    # dossier de sortie
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # Chargement du modèle de diffusion
    # Le U-Net est configuré différemment selon le checkpoint utilisé :
    #   diffusion_ffhq_10m    → 128 canaux, 1 bloc résiduel, attention à résolution 16
    #   256x256_diffusion_uncond → 256 canaux, 2 blocs résiduels, attention multi-résolution
    # ----------------------------------------

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if generate_mode != 'DPS_y0':
        # DPS_y0 requiert les gradients pour la guidance ; les autres modes n'en ont pas besoin.
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('use_DIY_kernel:{}, blur mode:{}'.format(use_DIY_kernel, blur_mode))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type):
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []

        for idx, img in enumerate(L_paths):
            # --------------------------------
            # Génération du noyau de flou
            # --------------------------------
            if use_DIY_kernel:
                # Graine différente par image pour reproductibilité mais variété des noyaux.
                np.random.seed(seed=idx*10)
                if blur_mode == 'Gaussian':
                    # Intensité variable autour de kernel_std (facteur entre 1 et 3).
                    kernel_std_i = kernel_std * np.abs(np.random.rand()*2+1)
                    kernel = GaussialBlurOperator(kernel_size=kernel_size, intensity=kernel_std_i, device=device)
                elif blur_mode == 'motion':
                    kernel = MotionBlurOperator(kernel_size=kernel_size, intensity=kernel_std, device=device)
                k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
                k = k_tensor.clone().detach().cpu().numpy()  # [0,1]
                k = np.squeeze(k)
                k = np.squeeze(k)
            else:
                k_index = 0
                kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'Levin09.mat'))['kernels']
                k = kernels[0, k_index].astype(np.float32)
            img_name, ext = os.path.splitext(os.path.basename(img))
            util.imsave(k*255.*200, os.path.join(E_path, f'motion_kernel_{img_name}{ext}'))

            # Noyau 4D pour la convolution PyTorch : forme (3, 3, H, W) via produit de Kronecker
            # afin d'appliquer le même noyau 2D indépendamment sur chaque canal RGB.
            k_4d = torch.from_numpy(k).to(device)
            k_4d = torch.einsum('ab,cd->abcd', torch.eye(3).to(device), k_4d)

            model_out_type = model_output_type

            # --------------------------------
            # (1) Préparation de l'image dégradée img_L
            # --------------------------------

            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)
            img_H = util.modcrop(img_H, 8)  # s'assure que la taille est divisible par 8

            # Convolution circulaire (mode='wrap') car la solution analytique FFT suppose
            # des conditions aux bords périodiques.
            img_L = ndimage.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
            util.imshow(img_L) if show_img else None
            img_L = util.uint2single(img_L)  # [0,255] → [0,1]

            np.random.seed(seed=0)
            # Normalise en [-1,1], ajoute le bruit AWGN (σ dans la plage [-1,1] = 2*σ_normalisé),
            # puis repasse en [0,1] pour les opérations suivantes.
            img_L = img_L * 2 - 1
            img_L += np.random.normal(0, noise_level_img * 2, img_L.shape)
            img_L = img_L / 2 + 0.5

            # --------------------------------
            # (2) Calcul de rho et sigma pour chaque pas
            # Rho[t] = λ * σ² / σ_k[t]²  est le paramètre de régularisation du sous-problème.
            # σ_k[t] est la variance du bruit sur x0 à l'instant t (conditionnelle à x_t).
            # --------------------------------

            sigmas = []    # niveaux de bruit équivalents (décroissants de t=999 à t=0)
            sigma_ks = []  # σ_k[t] = variance conditionnelle de x0 | x_t
            rhos = []      # ρ[t] = λσ²/σ_k[t]² (poids du terme de données dans HQS)
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart' and generate_mode == 'DiffPIR':
                    # Formule DiffPIR (Eq. A3 de l'article) : σ_k = √(1-ᾱ_t)/√ᾱ_t
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                else:
                    # Formule alternative : σ_k = √(β_t/α_t)
                    sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))
            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)

            # --------------------------------
            # (3) Initialisation de x et pré-calcul FFT
            # --------------------------------

            y = util.single2tensor4(img_L).to(device)  # observation y : (1,3,H,W) ∈ [0,1]

            # Initialisation de x à t_start en mélangeant y avec du bruit.
            # Plutôt que partir de bruit pur, on utilise y bruit au niveau de t_start,
            # ce qui accélère la convergence et évite les artefacts de début de trajectoire.
            t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
            sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
            x = sqrt_alpha_effective * (2*y-1) + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
                    sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)

            # Pré-calcul des matrices FFT réutilisées à chaque pas (économie de calcul).
            k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
            FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, sf)

            # --------------------------------
            # (4) Boucle principale de diffusion inverse
            # --------------------------------

            progress_img = []
            # Construction de la séquence de pas de temps (ordre croissant de bruit)
            if skip_type == 'uniform':
                seq = [i*skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps-1)
            elif skip_type == "quad":
                # Espacement quadratique : plus de pas aux niveaux de bruit élevés
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::max(len(seq)//10,1)]
            if progress_seq[-1] != seq[-1]:
                progress_seq.append(seq[-1])

            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                # Trouve le pas t_i correspondant au niveau de bruit courant
                t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
                if t_i > t_start:
                    continue  # saute les pas au-delà du point de départ
                for u in range(iter_num_U):
                    # --------------------------------
                    # Étape 1 : pas de diffusion inverse → estime x0
                    # Résout Eq. 6a : x_{t-1} ← p_θ(x_{t-1} | x_t)
                    # --------------------------------

                    if 'DPS' in generate_mode:
                        # DPS : requiert les gradients de x pour la guidance
                        x = x.requires_grad_()
                        xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,
                                    model_out_type='pred_x_prev_and_start',
                                    model_diffusion=model, diffusion=diffusion,
                                    ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                    else:
                        # DiffPIR / vanilla : pas de gradient requis ici
                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,
                                model_out_type=model_out_type,
                                model_diffusion=model, diffusion=diffusion,
                                ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                    # --------------------------------
                    # Étape 2 : correction de x0 par les données (Eq. 6b)
                    # --------------------------------

                    if seq[i] != seq[-1]:
                        if generate_mode == 'DiffPIR':
                            if sub_1_analytic:
                                if model_out_type == 'pred_xstart':
                                    # ρ[t_i] sous forme de tenseur 4D pour les opérations batch
                                    tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                    if i < num_train_timesteps-noise_model_t:
                                        # Déconvolution FFT : résout (B^T B + τI)x0_p = B^T y + τ x0
                                        # via la formule analytique dans le domaine fréquentiel.
                                        x0_p = x0 / 2 + 0.5  # normalise en [0,1] pour la FFT
                                        x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, sf)
                                        x0_p = x0_p * 2 - 1  # repasse en [-1,1]
                                        # guidance_scale permet d'atténuer la correction (1.0 = totale)
                                        x0 = x0 + guidance_scale * (x0_p-x0)
                                    else:
                                        # Niveau de bruit < bruit de l'image : la correction est inutile,
                                        # on revient à pred_x_prev (diffusion libre).
                                        model_out_type = 'pred_x_prev'
                                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,
                                                model_out_type=model_out_type,
                                                model_diffusion=model, diffusion=diffusion,
                                                ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                            else:
                                # Solveur de premier ordre (gradient) : alternative à la FFT.
                                # x0 ← x0 - (1/ρ) * ∇_{x0} ||y - Tx0||
                                x0 = x0.requires_grad_()
                                def Tx(x):
                                    x = x / 2 + 0.5
                                    pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                                    x_deblur = F.conv2d(pad_2d(x), k_4d)
                                    return x_deblur
                                norm_grad, norm = utils_model.grad_and_value(operator=Tx, x=x0, x_hat=x0, measurement=y)
                                x0 = x0 - norm_grad * norm / (rhos[t_i])
                                x0 = x0.detach_()
                        elif 'DPS' in generate_mode:
                            def Tx(x):
                                x = x / 2 + 0.5
                                pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                                x_deblur = F.conv2d(pad_2d(x), k_4d)
                                return x_deblur
                            if generate_mode == 'DPS_y0':
                                # Guidance DPS sur x0 : x_{t-1} ← x_{t-1} - ∇ ||y - Tx0||
                                norm_grad, norm = utils_model.grad_and_value(operator=Tx, x=x, x_hat=x0, measurement=y)
                                x = xt - norm_grad * 1.
                                x = x.detach_()
                            elif generate_mode == 'DPS_yt':
                                # Guidance DPS sur y_t (version bruitée de y au niveau t)
                                y_t = sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(y)
                                y_t = y_t/2 + 0.5
                                norm_grad, norm = utils_model.grad_and_value(operator=Tx, x=xt, x_hat=xt, measurement=y_t)
                                x = xt - norm_grad * lambda_ * norm / (rhos[t_i]) * 0.35
                                x = x.detach_()

                    # --------------------------------
                    # Re-bruitage de x0 au niveau t_{i-1}
                    # Implémente la formule DDIM généralisée (Eq. 12 de l'article) :
                    # x_{t-1} = √ᾱ_{t-1} * x0 + √(1-ζ) * (direction DDIM) + √ζ * bruit pur
                    # --------------------------------
                    if (generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and \
                            not (seq[i] == seq[-1] and u == iter_num_U-1):
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i+1]].cpu().numpy())
                        # Estime ε̂ (bruit résiduel) depuis x_t et x0
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        # Amplitude du bruit stochastique DDIM (η contrôle la variance)
                        eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                        # Formule finale : mélange direction déterministe + bruit DDIM + bruit pur (ζ)
                        x = sqrt_alphas_cumprod[t_im1] * x0 + \
                            np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                + eta_sigma * torch.randn_like(x)) + \
                            np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)

                    # Retour à x_t depuis x_{t-1} pour les itérations internes (iter_num_U > 1)
                    if u < iter_num_U-1 and seq[i] != seq[-1]:
                        sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                        x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                                sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

                # Repasse en [0,1] pour la sauvegarde intermédiaire
                x_0 = (x/2+0.5)
                if save_progressive and (seq[i] in progress_seq):
                    x_show = x_0.clone().detach().cpu().numpy()
                    x_show = np.squeeze(x_show)
                    if x_show.ndim == 3:
                        x_show = np.transpose(x_show, (1, 2, 0))
                    progress_img.append(x_show)
                    if log_process:
                        logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                    if show_img:
                        util.imshow(x_show)

            # --------------------------------
            # (3) Image restaurée finale
            # --------------------------------

            img_E = util.tensor2uint(x_0)
            psnr = util.calculate_psnr(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)

            if calc_LPIPS:
                img_H_tensor = np.transpose(img_H, (2, 0, 1))
                img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                img_H_tensor = img_H_tensor / 255 * 2 -1
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score)
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(idx+1, img_name+ext, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
            else:
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB'.format(idx+1, img_name+ext, psnr))

            if n_channels == 1:
                img_H = img_H.squeeze()

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+ext))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total,figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}{}'.format(noise_level_img,lambda_,current_time,psnr,ext)))

            if save_LEH:
                img_L = util.single2uint(img_L)
                k_v = k/np.max(k)*1.0
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if show_img else None
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_LEH'+ext))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_LR'+ext))

        # --------------------------------
        # Moyennes PSNR et LPIPS finales
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}): {:.4f} dB'.format(testset_name, noise_level_model, ave_psnr))

        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}) sigma: ({:.3f}): {:.4f}'.format(testset_name, noise_level_model, ave_lpips))


    # Boucle sur les valeurs de lambda_ à tester (range(7,8) → un seul λ = 7*lambda_)
    lambdas = [lambda_*i for i in range(7,8)]
    for lambda_ in lambdas:
        for zeta_i in [zeta*i for i in range(3,4)]:
            test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type)


if __name__ == '__main__':

    main()
