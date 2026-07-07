# utils/utils_sisr.py
#
# Solveur FFT pour la super-résolution et le défloutage dans DiffPIR.
#
# L'étape 2 de DiffPIR résout le sous-problème HQS (Half-Quadratic Splitting) :
#   min_{x} (1/2σ²) ||Ax - y||² + (ρ/2) ||x - z||²
# où A est l'opérateur de dégradation (flou + sous-échantillonnage), y l'observation,
# z = x̂₀ l'estimation de l'image propre depuis le modèle de diffusion, et ρ = λσ²/σ_k².
#
# Pour les opérateurs A modélisables par convolution + sous-échantillonnage,
# la solution analytique s'obtient dans le domaine de Fourier via les fonctions ci-dessous.
#
# Référence mathématique : Eq. (12) de l'article DiffPIR et travaux DPIR (Zhang et al., 2021).

# -*- coding: utf-8 -*-
import torch.fft
import torch

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp2d

def splits(a, sf):
    '''Découpe le tenseur a en sf×sf blocs distincts et les empile sur la dernière dimension.

    Utilisé dans data_solution pour calculer la moyenne des sous-bandes fréquentielles
    (équivalent de S^T en notation matricielle pour l'opérateur de sous-échantillonnage).

    Args:
        a: NxCxWxH
        sf: facteur de découpe
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf²)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''Convertit une PSF (Point Spread Function) en OTF (Optical Transfer Function).
    La PSF est zéro-paddée à la taille `shape` puis décalée circulairement pour
    que son centre soit en position (0,0) avant la FFT — condition nécessaire pour
    que la convolution circulaire soit exacte sans artefacts de phase.

    Args:
        psf: NxCxhxw  (noyau de flou)
        shape: [H, W]  (taille de l'image HR)
    Returns:
        otf: NxCxHxW  (complexe, dans le domaine de Fourier)
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    return otf


def upsample(x, sf=3):
    '''Sur-échantillonnage sf-fold par insertion de zéros (opérateur S^T en SR).
    Positionne chaque pixel LR au coin supérieur gauche de son patch HR.

    Args:
        x: NxCxWxH (image LR)
    Returns:
        z: NxCx(W*sf)x(H*sf) (image HR zéro-paddée)
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''Sous-échantillonnage sf-fold par sélection d'un pixel sur sf (opérateur S).
    Garde le pixel en position (0,0) de chaque patch sf×sf.

    Args:
        x: NxCxWxH (image HR)
    Returns:
        image LR : NxCx(W//sf)x(H//sf)
    '''
    st = 0
    return x[..., st::sf, st::sf]



def data_solution(x, FB, FBC, F2B, FBFy, alpha, sf):
    '''
    Solution analytique dans le domaine de Fourier pour le sous-problème HQS en SR.
    Résout : (B^T B + α I) x_out = B^T y + α x
    en exploitant la diagonalisation de l'opérateur B dans le domaine de Fourier.

    Formule (domaine de Fourier, après la propriété de sous-échantillonnage) :
      X_out = (F(α*x) + FBFy) / (F2B/sf² + α)
    où F2B/sf² est la somme des sous-bandes de |F(B)|².

    Args :
      x     : estimation courante x̂₀ ∈ [0,1], NxCxHxW
      FB    : TF du noyau B = F(k) zéro-paddé à taille HR
      FBC   : conjugué de FB
      F2B   : |FB|² = FB * FBC
      FBFy  : FBC * F(S^T y)  (pré-calculé une fois pour toutes)
      alpha : ρ[t], paramètre de régularisation (scalaire ou (1,1,1,1))
      sf    : facteur d'échelle SR
    Returns :
      Xest  : image restaurée ∈ [0,1], NxCxHxW
    '''
    FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))
    x1 = FB.mul(FR)
    # Moyenne des sf² sous-bandes : réalise l'opération S S^T dans le domaine fréquentiel
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    # Division fréquentielle : invW + alpha est le dénominateur diagonalisé
    invWBR = FBR.div(invW + alpha)
    FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/alpha
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

    return Xest


def pre_calculate(x, k, sf):
    '''
    Pré-calcule les matrices FFT réutilisées à chaque pas de diffusion.
    À appeler une seule fois avant la boucle principale pour économiser du calcul.

    Args:
        x:  NxCxHxW, image LR (observation y)
        k:  NxCxhxw, noyau de flou (doit avoir les bonnes dimensions pour le batch)
        sf: facteur d'échelle SR (1 pour le défloutage seul)

    Returns:
        FB   : TF du noyau B sur la grille HR (NxCxH*sf x W*sf, complexe)
        FBC  : conjugué de FB
        F2B  : |FB|² (puissance spectrale du noyau)
        FBFy : FBC * F(S^T y)  (terme constant de la solution, dépend seulement de y)
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)                        # S^T y : y upsamplé par insertion de zéros
    FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))   # F(B)^* * F(S^T y)
    return FB, FBC, F2B, FBFy




def classical_degradation(x, k, sf=3):
    '''Dégradation classique SR : convolution avec k puis sous-échantillonnage.

    Args:
        x: HxWxC, image HR [0,1] ou [0,255]
        k: hxw, noyau de flou (double)
        sf: facteur de sous-échantillonnage
    Returns:
        image LR sous-échantillonnée
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    st = 0
    return x[st::sf, st::sf, ...]



def shift_pixel(x, sf, upper_left=True):
    '''Décale les pixels pour la SR classique afin d'aligner le pixel LR
    avec le centre de son patch HR (correction du demi-pixel introduit par
    certaines conventions de sous-échantillonnage).

    Args:
        x: WxHxC ou WxH (image ou noyau)
        sf: facteur d'échelle
        upper_left: si True, décale vers le haut-gauche ; sinon vers le bas-droite
    '''
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x
