# main_ddpir_inpainting.py
#
# Script autonome pour l'inpainting (complétion d'image masquée) via DiffPIR.
#
# Particularité de l'inpainting par rapport au défloutage / SR :
#   L'opérateur de dégradation A est un masque binaire M :
#     y = M ⊙ x + bruit   (⊙ = produit élément à élément)
#   La solution analytique de l'étape 2 est explicite et ne requiert pas de FFT :
#     x0_p = (M ⊙ y + ρ * x0) / (M + ρ)   (element-wise)
#   À la fin, les pixels connus sont restaurés directement depuis y (pas de reconstruction).

import os.path
import cv2
import logging

import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict

from utils import utils_model
from utils import utils_logger
from utils import utils_image as util
from utils.utils_inpaint import mask_generator

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
    # 0/255 = pas de bruit AWGN sur les pixels connus (cas idéal d'inpainting).
    # Augmenter pour simuler un bruit de capteur sur les pixels observés.
    noise_level_img         = 0/255.0
    noise_level_model       = noise_level_img

    # --- Modèle de diffusion ---
    # 'diffusion_ffhq_10m' : recommandé pour les images de visages
    # '256x256_diffusion_uncond' : recommandé pour ImageNet (scènes générales)
    model_name              = 'diffusion_ffhq_10m'

    testset_name            = 'demo_test'  # 'imagenet_val' | 'ffhq_val' | 'demo_test'

    # --- Calendrier de diffusion (fixe, défini à l'entraînement) ---
    num_train_timesteps     = 1000

    # --- Paramètres d'itération ---
    # 20 pas suffisent souvent pour l'inpainting car le problème est moins mal posé
    # que la SR ou le défloutage (les pixels connus contraignent fortement la solution).
    iter_num                = 20
    iter_num_U              = 1  # itérations internes (>1 améliore la cohérence, voir RePaint)
    skip                    = num_train_timesteps//iter_num

    # --- Paramètres du masque ---
    # mask_name  : chemin vers un masque binaire (blanc=connu, noir=inconnu)
    # load_mask  : si True, charge mask_name ; si False, génère un masque aléatoire
    mask_name               = 'gt_keep_masks/face/000000.png'
    load_mask               = False

    # mask_type : type de masque généré aléatoirement (si load_mask=False).
    # 'box'     : un rectangle de pixels inconnus (bonne simulation d'occlusion)
    # 'random'  : pixels individuels masqués aléatoirement (bruit de sel et poivre)
    # 'both'    : combinaison box + random
    # 'extreme' : inverse du 'box' (seul le rectangle est conservé)
    mask_type               = 'random'

    # mask_len_range : [min, max] taille (en pixels) du côté du rectangle pour 'box'.
    mask_len_range          = [128, 129]

    # mask_prob_range : [min, max] probabilité qu'un pixel soit masqué pour 'random'.
    # [0.5, 0.5] = exactement 50% des pixels masqués.
    mask_prob_range         = [0.5, 0.5]

    # --- Affichage et sauvegarde ---
    show_img                = False
    save_L                  = False
    save_E                  = True
    save_LEH                = False
    save_progressive        = False
    save_progressive_mask   = False  # sauvegarder le processus avec le masque superposé

    # --- Paramètres d'inférence DiffPIR ---
    sigma                   = max(0.001, noise_level_img)

    # λ : poids du terme de données.
    # Pour l'inpainting, λ=1 suffit car la solution analytique est exacte (pas d'approximation).
    lambda_                 = 1.

    sub_1_analytic          = True  # toujours True pour l'inpainting (solution close-form disponible)

    # eta : stochasticité DDIM (0 = déterministe)
    eta                     = 0.0

    # zeta : mélange bruit déterministe / stochastique.
    # zeta=1.0 recommandé pour l'inpainting : tire un bruit frais à chaque pas,
    # ce qui évite que les régions masquées « collent » à une trajectoire fixe.
    zeta                    = 1.0

    guidance_scale          = 1.0

    # model_out_type : 'pred_xstart' recommandé pour DiffPIR
    model_out_type          = 'pred_xstart'

    # generate_mode : algorithme utilisé.
    # 'DiffPIR'  : solution analytique pour l'inpainting (Eq. 6b de l'article)
    # 'repaint'  : méthode RePaint (remplace pixels connus à chaque pas par y bruité)
    # 'vanilla'  : pas de contrainte de données (génération libre)
    generate_mode           = 'DiffPIR'

    # skip_type : espacement des pas de temps.
    skip_type               = 'quad'

    # ddim_sample : utiliser DDIM (True) ou DDPM (False) pour le pas de diffusion.
    ddim_sample             = False

    log_process             = False
    task_current            = 'ip'  # identifiant de la tâche (inpainting)
    n_channels              = 3
    cwd                     = ''
    model_zoo               = os.path.join(cwd, 'model_zoo')
    testsets                = os.path.join(cwd, 'testsets')
    results                 = os.path.join(cwd, 'results')
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{mask_type}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    calc_LPIPS              = True

    # ----------------------------------------
    # Calendrier de bruit (noise schedule)
    # Identique pour toutes les tâches DiffPIR.
    # ----------------------------------------
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    # σ équivalente : utilisée pour mapper noise_level → timestep t
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)

    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0

    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img)
    t_start                 = num_train_timesteps - 1  # départ depuis bruit pur (t=999)

    # ----------------------------------------
    # Chemins
    # ----------------------------------------

    L_path                  = os.path.join(testsets, testset_name)
    E_path                  = os.path.join(results, result_name)
    mask_path               = os.path.join(testsets, mask_name)
    util.mkdir(E_path)

    logger_name             = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger                  = logging.getLogger(logger_name)

    # ----------------------------------------
    # Chargement du modèle de diffusion
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
    # L'inpainting DiffPIR n'a jamais besoin des gradients du modèle.
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, mask_type:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, mask_type, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    def test_rho(lambda_=lambda_, model_out_type_=model_out_type, zeta=zeta):
        model_out_type = model_out_type_
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []

        for idx, img in enumerate(L_paths):

            # --------------------------------
            # (1) Chargement de l'image et du masque
            # --------------------------------

            idx += 1
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)  # image originale [0,255]

            # --------------------------------
            # (2) Génération / chargement du masque et de l'image masquée
            # --------------------------------
            if load_mask:
                # Masque binaire : True = pixel connu, False = pixel à reconstruire
                mask = util.imread_uint(mask_path, n_channels=n_channels).astype(bool)
            else:
                mask_gen = mask_generator(mask_type=mask_type,
                                          mask_len_range=mask_len_range,
                                          mask_prob_range=mask_prob_range)
                np.random.seed(seed=0)
                mask = mask_gen(util.uint2tensor4(img_H)).numpy()
                mask = np.squeeze(mask)
                mask = np.transpose(mask, (1, 2, 0))  # (H,W,C)

            # Image masquée : pixels inconnus mis à 0
            img_L = img_H * mask / 255.  # (H,W,C) ∈ [0,1]

            np.random.seed(seed=0)
            # Normalisation et ajout du bruit AWGN sur les pixels connus
            img_L = img_L * 2 - 1
            img_L += np.random.normal(0, noise_level_img * 2, img_L.shape)
            img_L = img_L / 2 + 0.5
            img_L = img_L * mask  # remet les pixels inconnus à 0 après le bruit

            # y en [-1,1] pour les opérations de diffusion
            y = util.single2tensor4(img_L).to(device)
            y = y * 2 - 1

            # Masque sous forme de tenseur float pour les opérations pondérées
            mask = util.single2tensor4(mask.astype(np.float32)).to(device)

            # Initialisation : mélange y bruité avec du bruit gaussien jusqu'au niveau t_start
            t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
            sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
            x = sqrt_alpha_effective * y + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
                    sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)

            # --------------------------------
            # (3) Calcul de rho et sigma
            # --------------------------------

            sigmas = []
            sigma_ks = []
            rhos = []
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart':
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                elif model_out_type == 'pred_x_prev':
                    sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))

            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)

            # --------------------------------
            # (4) Boucle principale de diffusion inverse
            # --------------------------------

            progress_img = []
            if skip_type == 'uniform':
                seq = [i*skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps-1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::(len(seq)//10)]
            progress_seq.append(seq[-1])

            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
                if t_i > t_start:
                    continue
                for u in range(iter_num_U):
                    # --------------------------------
                    # Étape 1 : diffusion inverse → estime x0
                    # --------------------------------

                    # Méthode RePaint : recolle les pixels connus bruités à leur niveau t_i
                    # avant chaque pas pour s'assurer de la cohérence des pixels observés.
                    if generate_mode == 'repaint':
                        x = (sqrt_alphas_cumprod[t_i] * y + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                                + (1-mask) * x

                    if model_out_type == 'pred_xstart':
                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,
                                model_out_type=model_out_type,
                                model_diffusion=model, diffusion=diffusion,
                                ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                    else:
                        x = utils_model.model_fn(x, noise_level=curr_sigma*255,
                                model_out_type=model_out_type,
                                model_diffusion=model, diffusion=diffusion,
                                ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                    # --------------------------------
                    # Étape 2 : solution analytique pour l'inpainting
                    # Résout : min_{x0_p} ||M(y - x0_p)||² + ρ||x0_p - x0||²
                    # Solution : x0_p = (M ⊙ y + ρ * x0) / (M + ρ)
                    # --------------------------------

                    if (generate_mode == 'DiffPIR') and not (seq[i] == seq[-1]):
                        if sub_1_analytic:
                            if model_out_type == 'pred_xstart':
                                if i < num_train_timesteps-noise_model_t:
                                    # Solution exacte closed-form pour le masque binaire.
                                    # Sur les pixels connus (M=1) : combine y et x0 avec poids ρ.
                                    # Sur les pixels inconnus (M=0) : garde x0 tel quel.
                                    x0_p = (mask*y + rhos[t_i].float()*x0).div(mask+rhos[t_i])
                                    x0 = x0 + guidance_scale * (x0_p-x0)
                                else:
                                    # Niveau de bruit trop bas pour la correction → diffusion libre
                                    model_out_type = 'pred_x_prev'
                                    x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,
                                        model_out_type=model_out_type,
                                        model_diffusion=model, diffusion=diffusion,
                                        ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                            elif model_out_type == 'pred_x_prev':
                                if i < num_train_timesteps-noise_model_t:
                                    # Correction directe sur x (pas x0) en mode pred_x_prev
                                    x = (mask*y + rhos[t_i].float()*x).div(mask+rhos[t_i])

                    # --------------------------------
                    # Re-bruitage vers t_{i-1} (formule DDIM généralisée avec zeta)
                    # --------------------------------

                    if (model_out_type == 'pred_xstart') and not (seq[i] == seq[-1]):
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i+1]].cpu().numpy())
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                        x = sqrt_alphas_cumprod[t_im1] * x0 + \
                            np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                + eta_sigma * torch.randn_like(x)) + \
                            np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)

                    # Retour en arrière pour les itérations internes (iter_num_U > 1)
                    if u < iter_num_U-1 and seq[i] != seq[-1]:
                        sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                        x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                                sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

                x_0 = (x/2+0.5)  # repasse en [0,1]
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

            # Post-traitement : restaure les pixels connus directement depuis y
            # (évite toute dégradation sur les régions observées).
            if generate_mode in ['repaint','DiffPIR']:
                x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

            # --------------------------------
            # Sauvegarde et évaluation
            # --------------------------------

            img_E = util.tensor2uint(x_0)
            psnr = util.calculate_psnr(img_E, img_H, border=0)
            test_results['psnr'].append(psnr)

            if calc_LPIPS:
                img_H_tensor = np.transpose(img_H, (2, 0, 1))
                img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                img_H_tensor = img_H_tensor / 255 * 2 - 1
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score)
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(idx, img_name+ext, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
            else:
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB'.format(idx, img_name+ext, psnr))
                pass

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+ext))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_L'+ext))

            if save_LEH:
                util.imsave(np.concatenate([util.single2uint(img_L), img_E, img_H], axis=1),
                            os.path.join(E_path, img_name+model_name+'_LEH'+ext))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                if generate_mode in ['repaint','DiffPIR']:
                    mask = np.squeeze(mask.cpu().numpy())
                    if mask.ndim == 3:
                        mask = np.transpose(mask, (1, 2, 0))
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total, figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+'_process_lambda_{:.3f}_{}{}'.format(lambda_,current_time,ext)))
                images = []
                y_t = np.squeeze((y/2+0.5).cpu().numpy())
                if y_t.ndim == 3:
                    y_t = np.transpose(y_t, (1, 2, 0))
                if generate_mode in ['repaint','DiffPIR']:
                    for x in progress_img:
                        images.append((y_t) * mask + (1-mask) * x)
                    img_total = cv2.hconcat(images)
                    if show_img:
                        util.imshow(img_total, figsize=(80,4))
                    if save_progressive_mask:
                        util.imsave(img_total*255., os.path.join(E_path, img_name+'_process_mask_lambda_{:.3f}_{}{}'.format(lambda_,current_time,ext)))

        # Résultats moyens
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}): {:.4f} dB'.format(testset_name, noise_level_model, ave_psnr))

        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}), sigma: ({:.3f}): {:.4f}'.format(testset_name, noise_level_model, ave_lpips))

    # Boucle sur les valeurs de lambda_ (range(1,2) → un seul λ = 1*lambda_)
    lambdas = [lambda_*i for i in range(1,2)]
    for lambda_ in lambdas:
        for zeta_i in [zeta*i for i in range(1,2)]:
            test_rho(lambda_, zeta=zeta_i)

if __name__ == '__main__':

    main()
