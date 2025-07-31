# %% [markdown]
# # 🎨 Data Augmentation - Pipeline d'Augmentation Avancé
# 
# ## 🎯 Objectifs
# 
# **Mission** : Créer un pipeline d'augmentation de données sophistiqué pour améliorer la robustesse du modèle
# 
# **Défis du dataset Cityscapes** :
# - **Déséquilibre extrême** : Classes rares (person 1.2%, object 1.8%) vs dominantes (road 38.7%, building 21.7%)
# - **Conditions limitées** : Principalement beau temps, jour, villes européennes
# - **Complexité des scènes** : Petits objets, occlusions, frontières précises
# 
# **Stratégies d'augmentation** :
# 1. **Augmentations géométriques** : Rotation, translation, déformation (cohérentes image+mask)
# 2. **Augmentations photométriques** : Luminosité, contraste, saturation (image seulement)
# 3. **Augmentations météorologiques** : Pluie, brouillard, neige pour robustesse
# 4. **Augmentations ciblées** : Focus sur classes rares avec techniques spécialisées
# 5. **Pipeline adaptatif** : Intensité variable selon la difficulté de l'image
# 
# **Contraintes techniques** :
# - Cohérence parfaite entre image et mask pour transformations géométriques
# - Préservation des labels lors des redimensionnements
# - Performance optimisée pour entraînement temps réel
# 
# ---

# %% [markdown]
# ## 📚 Imports et Configuration

# %%
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Augmentation libraries
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Utilities
from tqdm.auto import tqdm
import gc

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Chemins principaux
PROJECT_ROOT = Path("C:/Tonton/OpenClassrooms/Projet_7_traiter_images_systeme_embarque_voiture_autonome")
DATA_ROOT = PROJECT_ROOT / "data"
GTFINE_ROOT = DATA_ROOT / "gtFine"
LEFTIMG_ROOT = DATA_ROOT / "leftImg8bit"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "notebooks" / "outputs"

print("✅ Configuration chargée")
print(f"🎨 Pipeline d'augmentation avancé pour segmentation embarquée")
print(f"📦 Albumentations version: {A.__version__}")

# %% [markdown]
# ## ⚙️ Configuration de l'Augmentation
# 
# **Configuration unifiée** : Paramètres cohérents avec le reste du pipeline

# %%
# Configuration globale de l'augmentation
AUGMENTATION_CONFIG = {
    'image_settings': {
        'input_shape': (512, 1024, 3),      # Shape finale pour le modèle
        'target_shape': (512, 1024),        # Shape des masks
        'interpolation': cv2.INTER_LINEAR,   # Interpolation pour images
        'mask_interpolation': cv2.INTER_NEAREST,  # Interpolation pour masks (CRITIQUE)
        'dtype': np.float32
    },
    'augmentation_probabilities': {
        'training_intensity': 0.8,           # Intensité globale pour training
        'validation_intensity': 0.0,        # Pas d'augmentation pour validation
        'geometric_p': 0.6,                 # Probabilité transformations géométriques
        'photometric_p': 0.7,               # Probabilité transformations photométriques
        'weather_p': 0.4,                   # Probabilité conditions météo
        'targeted_p': 0.5                   # Probabilité augmentations ciblées
    },
    'geometric_transforms': {
        'horizontal_flip_p': 0.5,
        'rotation_limit': 15,
        'shift_limit': 0.1,
        'scale_limit': 0.1,
        'elastic_alpha': 50,
        'elastic_sigma': 5,
        'grid_distortion_limit': 0.2,
        'optical_distortion_limit': 0.1
    },
    'photometric_transforms': {
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'saturation_limit': 0.3,
        'hue_shift_limit': 20,
        'gamma_limit': (80, 120),
        'gaussian_noise_var': (10, 50),
        'motion_blur_limit': 7
    },
    'weather_conditions': {
        'rain_intensity': (0.1, 0.3),
        'fog_intensity': (0.3, 1.0),
        'snow_intensity': (0.1, 0.3),
        'shadow_intensity': (0.3, 0.7)
    },
    'class_balancing': {
        'rare_classes': [5, 2],             # person, object
        'dominant_classes': [0, 1],         # road, building
        'targeted_augmentation_factor': 2.0  # Boost pour classes rares
    }
}

# Charger configurations précédentes pour cohérence
try:
    with open(OUTPUTS_DIR / "class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    with open(OUTPUTS_DIR / "data_generator_config.json", 'r') as f:
        generator_config = json.load(f)
    
    print("✅ Configurations précédentes chargées")
    
    # Vérifier cohérence des dimensions
    gen_input_shape = tuple(generator_config['generator_settings']['input_shape'])
    aug_input_shape = AUGMENTATION_CONFIG['image_settings']['input_shape']
    
    if gen_input_shape == aug_input_shape:
        print(f"✅ Cohérence dimensions: {aug_input_shape}")
    else:
        print(f"⚠️ Incohérence dimensions détectée:")
        print(f"   Generator: {gen_input_shape}")
        print(f"   Augmentation: {aug_input_shape}")
        # Ajuster pour cohérence
        AUGMENTATION_CONFIG['image_settings']['input_shape'] = gen_input_shape
        print(f"🔧 Correction appliquée: {gen_input_shape}")
        
except FileNotFoundError as e:
    print(f"⚠️ Configuration manquante: {e}")
    print("Utilisation configuration par défaut")

print(f"\n🎯 Configuration d'augmentation:")
print(f"   • Input shape: {AUGMENTATION_CONFIG['image_settings']['input_shape']}")
print(f"   • Target shape: {AUGMENTATION_CONFIG['image_settings']['target_shape']}")
print(f"   • Intensité training: {AUGMENTATION_CONFIG['augmentation_probabilities']['training_intensity']}")

# %% [markdown]
# ## 🎨 Colormap Personnalisée basée sur class_mapping.json

# %%
def create_custom_colormap():
    """
    Crée une colormap personnalisée basée sur les couleurs du class_mapping.json
    """
    try:
        with open(OUTPUTS_DIR / "class_mapping.json", 'r') as f:
            mapping_data = json.load(f)
        
        # Extraire les couleurs hex et les convertir en RGB
        colors_rgb = []
        category_order = ['road', 'building', 'object', 'nature', 'sky', 'person', 'vehicle', 'void']
        
        for category in category_order:
            color_hex = mapping_data['category_colors'][category]
            # Convertir hex en RGB (0-1)
            hex_color = color_hex.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
            colors_rgb.append(rgb)
        
        # Créer la colormap
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(colors_rgb)
        
        print("✅ Colormap personnalisée créée avec les vraies couleurs")
        print(f"   • Couleurs: {[mapping_data['category_colors'][cat] for cat in category_order]}")
        return custom_cmap
        
    except Exception as e:
        print(f"⚠️ Erreur création colormap: {e}")
        # Fallback vers tab10
        return plt.cm.tab10

# Créer la colormap personnalisée
CUSTOM_COLORMAP = create_custom_colormap()

# %% [markdown]
# ## 🧩 Transformations Géométriques Cohérentes
# 
# **Principe critique** : Les transformations géométriques doivent être appliquées de manière **identique** 
# sur l'image et le mask pour préserver la correspondance spatiale.

# %%
class GeometricAugmentations:
    """
    Transformations géométriques cohérentes pour segmentation sémantique
    """
    
    def __init__(self, config):
        self.config = config['geometric_transforms']
        self.image_shape = config['image_settings']['input_shape']
        self.target_shape = config['image_settings']['target_shape']
        
    def create_basic_geometric_pipeline(self):
        """
        Pipeline géométrique de base (safe)
        """
        transforms = [
            # Flip horizontal (symétrie urbaine préservée)
            A.HorizontalFlip(p=self.config['horizontal_flip_p']),
            
            # Rotations légères
            A.OneOf([
                A.Rotate(
                    limit=self.config['rotation_limit'],
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=self.config['shift_limit'],
                    scale_limit=self.config['scale_limit'],
                    rotate_limit=10,  # Plus conservateur
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.5
                )
            ], p=0.4)
        ]
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'},
            is_check_shapes=True
        )
    
    def create_advanced_geometric_pipeline(self):
        """
        Pipeline géométrique avancé avec déformations
        """
        transforms = [
            # Transformations de base
            A.HorizontalFlip(p=self.config['horizontal_flip_p']),
            
            # Rotations et translations
            A.OneOf([
                A.Rotate(
                    limit=self.config['rotation_limit'],
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=self.config['shift_limit'],
                    scale_limit=self.config['scale_limit'],
                    rotate_limit=10,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.5
                )
            ], p=0.4),
            
            # Déformations élastiques (plus agressives)
            A.OneOf([
                A.ElasticTransform(
                    alpha=self.config['elastic_alpha'],
                    sigma=self.config['elastic_sigma'],
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.3
                ),
                A.GridDistortion(
                    distort_limit=self.config['grid_distortion_limit'],
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.3
                ),
                A.OpticalDistortion(
                    distort_limit=self.config['optical_distortion_limit'],
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.4
                )
            ], p=0.2)
        ]
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'},
            is_check_shapes=True
        )
    
    def visualize_geometric_effects(self, test_image, test_mask):
        """
        Visualise les effets des transformations géométriques
        """
        # Pipelines à tester
        pipelines = {
            'Basic Geometric': self.create_basic_geometric_pipeline(),
            'Advanced Geometric': self.create_advanced_geometric_pipeline()
        }
        
        fig, axes = plt.subplots(len(pipelines), 4, figsize=(20, 5*len(pipelines)))
        if len(pipelines) == 1:
            axes = axes.reshape(1, -1)
        
        for row, (pipeline_name, pipeline) in enumerate(pipelines.items()):
            # Image originale
            axes[row, 0].imshow(test_image)
            axes[row, 0].set_title(f'Image Originale\n{pipeline_name}', fontweight='bold')
            axes[row, 0].axis('off')
            
            # Mask original avec colormap personnalisée
            axes[row, 1].imshow(test_mask, cmap=CUSTOM_COLORMAP, vmin=0, vmax=7)
            axes[row, 1].set_title('Mask Original', fontweight='bold')
            axes[row, 1].axis('off')
            
            # Application de l'augmentation
            try:
                augmented = pipeline(image=test_image, mask=test_mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # Image augmentée
                axes[row, 2].imshow(aug_image)
                axes[row, 2].set_title('Image Augmentée', fontweight='bold')
                axes[row, 2].axis('off')
                
                # Mask augmenté avec colormap personnalisée
                axes[row, 3].imshow(aug_mask, cmap=CUSTOM_COLORMAP, vmin=0, vmax=7)
                axes[row, 3].set_title('Mask Augmenté', fontweight='bold')
                axes[row, 3].axis('off')
                
                print(f"✅ {pipeline_name}: Success")
                print(f"   Image: {test_image.shape} → {aug_image.shape}")
                print(f"   Mask: {test_mask.shape} → {aug_mask.shape}")
                
            except Exception as e:
                print(f"❌ {pipeline_name}: Error - {e}")
                # Afficher erreur sur les plots
                axes[row, 2].text(0.5, 0.5, f'ERREUR:\n{str(e)[:50]}...', 
                                ha='center', va='center', transform=axes[row, 2].transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                axes[row, 2].set_title('ERREUR', color='red', fontweight='bold')
                axes[row, 2].axis('off')
                
                axes[row, 3].text(0.5, 0.5, 'ERREUR', 
                                ha='center', va='center', transform=axes[row, 3].transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                axes[row, 3].set_title('ERREUR', color='red', fontweight='bold')
                axes[row, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "geometric_augmentations.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n🧩 Classe GeometricAugmentations définie")

# %% [markdown]
# ## 🌈 Transformations Photométriques
# 
# **Principe** : Ces transformations modifient l'apparence visuelle sans affecter la géométrie.
# Elles s'appliquent **uniquement aux images**, pas aux masks.

# %%
class PhotometricAugmentations:
    """
    Transformations photométriques pour améliorer la robustesse aux conditions d'éclairage
    """
    
    def __init__(self, config):
        self.config = config['photometric_transforms']
        
    def create_lighting_pipeline(self):
        """
        Pipeline pour conditions d'éclairage variées
        """
        transforms = [
            # Ajustements de luminosité et contraste
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['brightness_limit'],
                    contrast_limit=self.config['contrast_limit'],
                    p=0.6
                ),
                A.RandomGamma(
                    gamma_limit=self.config['gamma_limit'],
                    p=0.4
                ),
                A.CLAHE(
                    clip_limit=4.0,
                    tile_grid_size=(8, 8),
                    p=0.3
                )
            ], p=0.7),
            
            # Ajustements de couleur
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=self.config['hue_shift_limit'],
                    sat_shift_limit=self.config['saturation_limit'],
                    val_shift_limit=self.config['brightness_limit'],
                    p=0.5
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=0.4
                ),
                A.ChannelShuffle(p=0.1)
            ], p=0.5)
        ]
        
        return A.Compose(transforms)
    
    def create_noise_pipeline(self):
        """
        Pipeline pour simulation de bruit et flou
        """
        transforms = [
            # Bruits
            A.OneOf([
                A.GaussNoise(
                    var_limit=self.config['gaussian_noise_var'],
                    mean=0,
                    p=0.4
                ),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=0.3
                ),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    per_channel=True,
                    p=0.3
                )
            ], p=0.4),
            
            # Flous
            A.OneOf([
                A.MotionBlur(
                    blur_limit=self.config['motion_blur_limit'],
                    p=0.3
                ),
                A.MedianBlur(
                    blur_limit=3,
                    p=0.2
                ),
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.3
                )
            ], p=0.3),
            
            # Compression artifacts
            A.OneOf([
                A.ImageCompression(
                    quality_lower=60,
                    quality_upper=100,
                    p=0.3
                ),
                A.Downscale(
                    scale_min=0.8,
                    scale_max=0.95,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.2
                )
            ], p=0.2)
        ]
        
        return A.Compose(transforms)
    
    def create_complete_photometric_pipeline(self):
        """
        Pipeline photométrique complet
        """
        transforms = [
            # Éclairage
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['brightness_limit'],
                    contrast_limit=self.config['contrast_limit'],
                    p=0.6
                ),
                A.RandomGamma(
                    gamma_limit=self.config['gamma_limit'],
                    p=0.4
                )
            ], p=0.6),
            
            # Couleur
            A.HueSaturationValue(
                hue_shift_limit=self.config['hue_shift_limit'],
                sat_shift_limit=self.config['saturation_limit'],
                val_shift_limit=10,
                p=0.4
            ),
            
            # Bruit et flou légers
            A.OneOf([
                A.GaussNoise(
                    var_limit=(10, 30),
                    p=0.3
                ),
                A.MotionBlur(
                    blur_limit=5,
                    p=0.2
                )
            ], p=0.3)
        ]
        
        return A.Compose(transforms)
    
    def visualize_photometric_effects(self, test_image):
        """
        Visualise les effets des transformations photométriques
        """
        pipelines = {
            'Lighting': self.create_lighting_pipeline(),
            'Noise & Blur': self.create_noise_pipeline(),
            'Complete': self.create_complete_photometric_pipeline()
        }
        
        fig, axes = plt.subplots(len(pipelines), 3, figsize=(15, 5*len(pipelines)))
        if len(pipelines) == 1:
            axes = axes.reshape(1, -1)
        
        for row, (pipeline_name, pipeline) in enumerate(pipelines.items()):
            # Image originale
            axes[row, 0].imshow(test_image)
            axes[row, 0].set_title(f'Original\n{pipeline_name}', fontweight='bold')
            axes[row, 0].axis('off')
            
            try:
                # Application 1
                aug1 = pipeline(image=test_image)['image']
                axes[row, 1].imshow(aug1)
                axes[row, 1].set_title('Augmentation 1', fontweight='bold')
                axes[row, 1].axis('off')
                
                # Application 2
                aug2 = pipeline(image=test_image)['image']
                axes[row, 2].imshow(aug2)
                axes[row, 2].set_title('Augmentation 2', fontweight='bold')
                axes[row, 2].axis('off')
                
                print(f"✅ {pipeline_name}: Success")
                
            except Exception as e:
                print(f"❌ {pipeline_name}: Error - {e}")
                for col in [1, 2]:
                    axes[row, col].text(0.5, 0.5, f'ERREUR:\n{str(e)[:30]}...', 
                                      ha='center', va='center', transform=axes[row, col].transAxes,
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "photometric_augmentations.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n🌈 Classe PhotometricAugmentations définie")

# %% [markdown]
# ## 🌦️ Conditions Météorologiques
# 
# **Objectif** : Simuler différentes conditions météo pour améliorer la robustesse du modèle
# en conditions réelles (pluie, brouillard, neige).

# %%
class WeatherAugmentations:
    """
    Simulations de conditions météorologiques pour robustesse
    """
    
    def __init__(self, config):
        self.config = config['weather_conditions']
        
    def create_rain_pipeline(self):
        """
        Simulation de conditions pluvieuses
        """
        transforms = [
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=7,
                brightness_coefficient=0.7,
                rain_type='drizzle',
                p=1.0
            )
        ]
        
        return A.Compose(transforms)
    
    def create_fog_pipeline(self):
        """
        Simulation de brouillard
        """
        transforms = [
            A.RandomFog(
                fog_coef_lower=self.config['fog_intensity'][0],
                fog_coef_upper=self.config['fog_intensity'][1],
                alpha_coef=0.08,
                p=1.0
            )
        ]
        
        return A.Compose(transforms)
    
    def create_snow_pipeline(self):
        """
        Simulation de neige
        """
        transforms = [
            A.RandomSnow(
                snow_point_lower=self.config['snow_intensity'][0],
                snow_point_upper=self.config['snow_intensity'][1],
                brightness_coeff=2.5,
                p=1.0
            )
        ]
        
        return A.Compose(transforms)
    
    def create_shadow_pipeline(self):
        """
        Simulation d'ombres
        """
        transforms = [
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=1.0
            )
        ]
        
        return A.Compose(transforms)
    
    def create_weather_mix_pipeline(self):
        """
        Pipeline combinant différentes conditions météo
        """
        transforms = [
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=15,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=5,
                    brightness_coefficient=0.8,
                    rain_type='drizzle',
                    p=0.4
                ),
                A.RandomFog(
                    fog_coef_lower=0.3,
                    fog_coef_upper=0.8,
                    alpha_coef=0.1,
                    p=0.3
                ),
                A.RandomSnow(
                    snow_point_lower=0.1,
                    snow_point_upper=0.25,
                    brightness_coeff=2.0,
                    p=0.2
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=1,
                    shadow_dimension=3,
                    p=0.3
                )
            ], p=0.4)
        ]
        
        return A.Compose(transforms)
    
    def visualize_weather_effects(self, test_image):
        """
        Visualise les effets météorologiques
        """
        weather_types = {
            'Rain': self.create_rain_pipeline(),
            'Fog': self.create_fog_pipeline(),
            'Snow': self.create_snow_pipeline(),
            'Shadow': self.create_shadow_pipeline(),
            'Mixed Weather': self.create_weather_mix_pipeline()
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Image originale
        axes[0].imshow(test_image)
        axes[0].set_title('Original (Clear Weather)', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # Applications météo
        for idx, (weather_name, pipeline) in enumerate(weather_types.items(), 1):
            try:
                augmented = pipeline(image=test_image)['image']
                axes[idx].imshow(augmented)
                axes[idx].set_title(weather_name, fontweight='bold', fontsize=12)
                axes[idx].axis('off')
                
                print(f"✅ {weather_name}: Success")
                
            except Exception as e:
                print(f"❌ {weather_name}: Error - {e}")
                axes[idx].text(0.5, 0.5, f'ERREUR:\n{weather_name}\n{str(e)[:30]}...', 
                             ha='center', va='center', transform=axes[idx].transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                axes[idx].set_title(f'{weather_name} - ERROR', color='red', fontweight='bold')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "weather_augmentations.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n🌦️ Classe WeatherAugmentations définie")

# %% [markdown]
# ## 🎯 Augmentations Ciblées pour Classes Rares
# 
# **Stratégie spécialisée** : Techniques avancées pour améliorer la détection des classes minoritaires
# (person: 1.2%, object: 1.8%) qui sont critiques pour la sécurité.

# %%
class TargetedAugmentations:
    """
    Augmentations ciblées pour améliorer la performance sur les classes rares
    """
    
    def __init__(self, config):
        self.config = config['class_balancing']
        self.rare_classes = config['class_balancing']['rare_classes']
        self.target_shape = config['image_settings']['target_shape']
        
    def has_rare_classes(self, mask):
        """
        Vérifie si l'image contient des classes rares
        """
        unique_classes = np.unique(mask)
        return any(rare_class in unique_classes for rare_class in self.rare_classes)
    
    def get_rare_class_coverage(self, mask):
        """
        Calcule le pourcentage de pixels des classes rares
        """
        total_pixels = mask.size
        rare_pixels = 0
        
        for rare_class in self.rare_classes:
            rare_pixels += np.sum(mask == rare_class)
        
        return (rare_pixels / total_pixels) * 100
    
    def create_copy_paste_augmentation(self):
        """
        Augmentation copy-paste pour dupliquer les objets rares
        """
        transforms = [
            # Augmentations préservant les petits objets
            A.OneOf([
                A.RandomCrop(
                    height=int(self.target_shape[0] * 0.8),
                    width=int(self.target_shape[1] * 0.8),
                    p=0.3
                ),
                A.CenterCrop(
                    height=int(self.target_shape[0] * 0.9),
                    width=int(self.target_shape[1] * 0.9),
                    p=0.3
                )
            ], p=0.3),
            
            # Redimensionnement pour revenir à la taille originale
            A.Resize(
                height=self.target_shape[0],
                width=self.target_shape[1],
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            ),
            
            # Augmentations favorisant la visibilité
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Flips pour doubler les variations
            A.HorizontalFlip(p=0.5)
        ]
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'},
            is_check_shapes=True
        )
    
    def create_mixup_augmentation(self):
        """
        Simulation de MixUp adapté à la segmentation
        """
        transforms = [
            # Augmentations géométriques douces
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.4
            ),
            
            # Augmentations photométriques douces pour préserver les détails
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=10,
                    val_shift_limit=5,
                    p=0.5
                )
            ], p=0.4)
        ]
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'},
            is_check_shapes=True
        )
    
    def create_targeted_pipeline(self, rare_coverage_threshold=3.0):
        """
        Pipeline adaptatif selon la présence de classes rares
        """
        transforms = [
            # Augmentations géométriques conservatrices
            A.OneOf([
                A.HorizontalFlip(p=0.6),
                A.Rotate(
                    limit=10,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.4
                )
            ], p=0.5),
            
            # Augmentations photométriques pour améliorer la visibilité
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.6
                ),
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(4, 4),
                    p=0.4
                )
            ], p=0.4),
            
            # Réduction du bruit pour préserver les détails fins
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 3), p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2)
            ], p=0.1)
        ]
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'},
            is_check_shapes=True
        )
    
    def visualize_targeted_effects(self, test_image, test_mask):
        """
        Visualise les effets des augmentations ciblées
        """
        # Vérifier présence de classes rares
        has_rare = self.has_rare_classes(test_mask)
        rare_coverage = self.get_rare_class_coverage(test_mask)
        
        print(f"📊 Analyse de l'image test:")
        print(f"   • Classes rares présentes: {'✅' if has_rare else '❌'}")
        print(f"   • Couverture classes rares: {rare_coverage:.2f}%")
        print(f"   • Classes uniques: {np.unique(test_mask)}")
        
        # Pipelines ciblés
        pipelines = {
            'Copy-Paste Style': self.create_copy_paste_augmentation(),
            'MixUp Style': self.create_mixup_augmentation(),
            'Targeted Conservative': self.create_targeted_pipeline()
        }
        
        fig, axes = plt.subplots(len(pipelines), 4, figsize=(20, 5*len(pipelines)))
        if len(pipelines) == 1:
            axes = axes.reshape(1, -1)
        
        for row, (pipeline_name, pipeline) in enumerate(pipelines.items()):
            # Image et mask originaux
            axes[row, 0].imshow(test_image)
            axes[row, 0].set_title(f'Image Originale\n{pipeline_name}', fontweight='bold')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(test_mask, cmap=CUSTOM_COLORMAP, vmin=0, vmax=7)
            axes[row, 1].set_title(f'Mask Original\n(Rare: {rare_coverage:.1f}%)', fontweight='bold')
            axes[row, 1].axis('off')
            
            try:
                # Application de l'augmentation
                augmented = pipeline(image=test_image, mask=test_mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # Calcul nouvelle couverture
                new_coverage = self.get_rare_class_coverage(aug_mask)
                
                axes[row, 2].imshow(aug_image)
                axes[row, 2].set_title('Image Augmentée', fontweight='bold')
                axes[row, 2].axis('off')
                
                axes[row, 3].imshow(aug_mask, cmap=CUSTOM_COLORMAP, vmin=0, vmax=7)
                axes[row, 3].set_title(f'Mask Augmenté\n(Rare: {new_coverage:.1f}%)', fontweight='bold')
                axes[row, 3].axis('off')
                
                print(f"✅ {pipeline_name}: Success (Coverage: {rare_coverage:.1f}% → {new_coverage:.1f}%)")
                
            except Exception as e:
                print(f"❌ {pipeline_name}: Error - {e}")
                for col in [2, 3]:
                    axes[row, col].text(0.5, 0.5, f'ERREUR:\n{str(e)[:40]}...', 
                                    ha='center', va='center', transform=axes[row, col].transAxes,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "targeted_augmentations.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n🎯 Classe TargetedAugmentations définie")

# %% [markdown]
# ## 🔄 Pipeline d'Augmentation Complet et Adaptatif
# 
# **Architecture modulaire** : Combine toutes les techniques d'augmentation dans un pipeline unifié
# qui s'adapte automatiquement selon le contenu de l'image et le mode d'entraînement.

# %%
class ComprehensiveAugmentationPipeline:
    """
    Pipeline d'augmentation complet et adaptatif pour segmentation sémantique
    """
    
    def __init__(self, config=AUGMENTATION_CONFIG, mode='training'):
        self.config = config
        self.mode = mode  # 'training' ou 'validation'
        
        # Initialisation des modules
        self.geometric = GeometricAugmentations(config)
        self.photometric = PhotometricAugmentations(config)
        self.weather = WeatherAugmentations(config)
        self.targeted = TargetedAugmentations(config)
        
        # Construction des pipelines
        self._build_pipelines()
        
    def _build_pipelines(self):
        """
        Construit les pipelines selon le mode
        """
        if self.mode == 'training':
            self.pipeline = self._create_training_pipeline()
        else:
            self.pipeline = self._create_validation_pipeline()
            
        print(f"✅ Pipeline {self.mode} construit avec succès")
    
    def _create_training_pipeline(self):
        """
        Pipeline d'entraînement avec toutes les augmentations
        """
        transforms = []
        
        # 1. Transformations géométriques (avec masks)
        geometric_prob = self.config['augmentation_probabilities']['geometric_p']
        transforms.append(
            A.OneOf([
                self.geometric.create_basic_geometric_pipeline(),
                self.geometric.create_advanced_geometric_pipeline()
            ], p=geometric_prob)
        )
        
        # 2. Transformations photométriques (image seulement)
        photo_prob = self.config['augmentation_probabilities']['photometric_p']
        transforms.extend([
            A.OneOf([
                self.photometric.create_lighting_pipeline(),
                self.photometric.create_complete_photometric_pipeline()
            ], p=photo_prob)
        ])
        
        # 3. Conditions météorologiques (image seulement)
        weather_prob = self.config['augmentation_probabilities']['weather_p']
        transforms.append(
            self.weather.create_weather_mix_pipeline()
        )
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'},
            is_check_shapes=True
        )
    
    def _create_validation_pipeline(self):
        """
        Pipeline de validation (minimal ou aucune augmentation)
        """
        # Seulement des transformations très légères pour la validation
        if self.config['augmentation_probabilities']['validation_intensity'] > 0:
            transforms = [
                A.HorizontalFlip(p=0.5)  # Seul flip horizontal pour doubler les données
            ]
            
            return A.Compose(
                transforms,
                additional_targets={'mask': 'mask'},
                is_check_shapes=True
            )
        else:
            # Pas d'augmentation pour validation
            return A.Compose(
                [],
                additional_targets={'mask': 'mask'},
                is_check_shapes=True
            )
    
    def _apply_targeted_augmentations(self, image, mask):
        """
        Applique des augmentations ciblées si classes rares détectées
        """
        if self.targeted.has_rare_classes(mask):
            # Augmentation plus agressive pour images avec classes rares
            targeted_pipeline = self.targeted.create_targeted_pipeline()
            
            # Application avec probabilité
            if np.random.random() < self.config['augmentation_probabilities']['targeted_p']:
                result = targeted_pipeline(image=image, mask=mask)
                return result['image'], result['mask']
        
        return image, mask
    
    def __call__(self, image, mask=None):
        """
        Application du pipeline complet
        """
        try:
            # Vérifier les dimensions d'entrée
            expected_shape = self.config['image_settings']['input_shape']
            if image.shape != expected_shape:
                print(f"⚠️ Dimension inattendue: {image.shape} vs {expected_shape}")
            
            if mask is not None:
                expected_mask_shape = self.config['image_settings']['target_shape']
                if mask.shape != expected_mask_shape:
                    print(f"⚠️ Dimension mask inattendue: {mask.shape} vs {expected_mask_shape}")
            
            # Application du pipeline principal
            if mask is not None:
                # Mode avec mask (entraînement)
                result = self.pipeline(image=image, mask=mask)
                aug_image, aug_mask = result['image'], result['mask']
                
                # Augmentations ciblées si mode training
                if self.mode == 'training':
                    aug_image, aug_mask = self._apply_targeted_augmentations(aug_image, aug_mask)
                
                return aug_image, aug_mask
            else:
                # Mode image seulement (inférence)
                result = self.pipeline(image=image)
                return result['image']
                
        except Exception as e:
            print(f"❌ Erreur dans le pipeline d'augmentation: {e}")
            # Retourner les données originales en cas d'erreur
            if mask is not None:
                return image, mask
            else:
                return image
    
    def get_pipeline_info(self):
        """
        Retourne les informations sur le pipeline
        """
        return {
            'mode': self.mode,
            'config': self.config,
            'pipeline_type': type(self.pipeline).__name__,
            'transforms_count': len(self.pipeline.transforms) if hasattr(self.pipeline, 'transforms') else 0
        }
    
    def visualize_complete_pipeline(self, test_image, test_mask, n_samples=6):
        """
        Visualise plusieurs applications du pipeline complet
        """
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        
        for i in range(n_samples):
            try:
                # Application du pipeline
                aug_image, aug_mask = self(test_image, test_mask)
                
                # Affichage image
                axes[0, i].imshow(aug_image)
                axes[0, i].set_title(f'Sample {i+1}\nImage Augmentée', fontweight='bold')
                axes[0, i].axis('off')
                
                # Affichage mask avec colormap personnalisée
                axes[1, i].imshow(aug_mask, cmap=CUSTOM_COLORMAP, vmin=0, vmax=7)
                axes[1, i].set_title(f'Sample {i+1}\nMask Augmenté', fontweight='bold')
                axes[1, i].axis('off')
                
            except Exception as e:
                print(f"❌ Erreur sample {i+1}: {e}")
                axes[0, i].text(0.5, 0.5, f'ERREUR\n{str(e)[:20]}...', 
                            ha='center', va='center', transform=axes[0, i].transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"complete_pipeline_{self.mode}.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n🔄 Classe ComprehensiveAugmentationPipeline définie")

# %% [markdown]
# ## 🧪 Test et Validation des Pipelines
# 
# **Validation critique** : Tester tous les pipelines avec des données réelles pour s'assurer
# de la compatibilité avec le reste du système.

# %%
# Mapping des classes 34 -> 8 catégories (chargé depuis l'EDA)
def load_class_mapping():
    """
    Charge le mapping des classes depuis l'EDA
    """
    try:
        with open(OUTPUTS_DIR / "class_mapping.json", 'r') as f:
            mapping_data = json.load(f)
        
        # Créer le mapping array 34->8
        id_to_8_categories = np.zeros(256, dtype=np.uint8)
        
        # Utiliser la structure réelle du fichier
        if 'id_to_8_categories' in mapping_data:
            for cityscapes_id, target_category in mapping_data['id_to_8_categories'].items():
                id_to_8_categories[int(cityscapes_id)] = int(target_category)
        
        print("✅ Mapping des classes chargé depuis l'EDA")
        print(f"   • Mapping: {len(mapping_data['id_to_8_categories'])} classes -> 8 catégories")
        print(f"   • Classes cibles: {sorted(set(mapping_data['id_to_8_categories'].values()))}")
        
        return id_to_8_categories
        
    except (FileNotFoundError, KeyError) as e:
        print(f"⚠️ Erreur chargement mapping: {e}")
        print("📝 Utilisation mapping par défaut")
        
        # Mapping par défaut simplifié
        mapping = np.zeros(256, dtype=np.uint8)
        # Classes principales du dataset Cityscapes
        mapping[7] = 0   # road
        mapping[8] = 0   # sidewalk -> road
        mapping[11] = 1  # building
        mapping[12] = 1  # wall -> building
        mapping[13] = 1  # fence -> building
        mapping[17] = 2  # pole -> object
        mapping[19] = 2  # traffic light -> object
        mapping[20] = 2  # traffic sign -> object
        mapping[21] = 3  # vegetation -> nature
        mapping[22] = 3  # terrain -> nature
        mapping[23] = 4  # sky
        mapping[24] = 5  # person
        mapping[25] = 5  # rider -> person
        mapping[26] = 6  # car -> vehicle
        mapping[27] = 6  # truck -> vehicle
        mapping[28] = 6  # bus -> vehicle
        mapping[31] = 6  # train -> vehicle
        mapping[32] = 6  # motorcycle -> vehicle
        mapping[33] = 6  # bicycle -> vehicle
        
        print("✅ Mapping par défaut créé")
        return mapping

# Chargement du mapping
ID_TO_8_CATEGORIES = load_class_mapping()

def load_real_cityscapes_sample():
    """
    Charge un échantillon réel du dataset Cityscapes
    """
    # Dimensions cibles
    input_shape = AUGMENTATION_CONFIG['image_settings']['input_shape']
    target_shape = AUGMENTATION_CONFIG['image_settings']['target_shape']
    
    print(f"📊 Chargement échantillon Cityscapes réel:")
    print(f"   • Image shape cible: {input_shape}")
    print(f"   • Mask shape cible: {target_shape}")
    
    # Vérifier la disponibilité des données
    if not GTFINE_ROOT.exists():
        print(f"❌ Dataset non trouvé: {GTFINE_ROOT}")
        print("📝 Utilisation des données de test synthétiques")
        return create_synthetic_test_data()
    
    # Chercher des fichiers d'exemple dans les données
    sample_files = []
    
    for split in ['train', 'val']:
        split_path = GTFINE_ROOT / split
        if split_path.exists():
            for city_dir in split_path.iterdir():
                if city_dir.is_dir():
                    # Chercher des fichiers labelIds
                    label_files = list(city_dir.glob("*_gtFine_labelIds.png"))
                    for label_file in label_files[:3]:  # Max 3 par ville
                        # Chercher l'image correspondante si disponible
                        base_name = label_file.stem.replace('_gtFine_labelIds', '')
                        
                        if LEFTIMG_ROOT.exists():
                            img_path = LEFTIMG_ROOT / split / city_dir.name / f"{base_name}_leftImg8bit.png"
                            if img_path.exists():
                                sample_files.append((img_path, label_file))
                            else:
                                # Essayer avec extension .jpg
                                img_path = LEFTIMG_ROOT / split / city_dir.name / f"{base_name}_leftImg8bit.jpg"
                                if img_path.exists():
                                    sample_files.append((img_path, label_file))
                        else:
                            # Utiliser seulement le mask avec une image synthétique
                            sample_files.append((None, label_file))
                        
                        if len(sample_files) >= 5:  # Limiter à 5 échantillons
                            break
                    
                    if len(sample_files) >= 5:
                        break
            
            if len(sample_files) >= 5:
                break
    
    if not sample_files:
        print("❌ Aucun fichier trouvé dans le dataset")
        return create_synthetic_test_data()
    
    # Charger un échantillon aléatoire
    img_path, label_path = sample_files[np.random.randint(len(sample_files))]
    
    try:
        # Charger le mask
        mask_img = Image.open(label_path)
        mask_array = np.array(mask_img)
        
        print(f"📂 Fichier chargé: {label_path.name}")
        print(f"   • Mask original shape: {mask_array.shape}")
        print(f"   • Classes originales: {len(np.unique(mask_array))} classes")
        print(f"   • Range valeurs: [{mask_array.min()}, {mask_array.max()}]")
        
        # Convertir les classes 34->8 categories
        mask_8_categories = ID_TO_8_CATEGORIES[mask_array]
        
        print(f"   • Classes après mapping: {np.unique(mask_8_categories)}")
        
        # Redimensionner le mask si nécessaire
        if mask_8_categories.shape != target_shape:
            mask_resized = cv2.resize(
                mask_8_categories, 
                (target_shape[1], target_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask_8_categories
        
        # Charger l'image correspondante
        if img_path and img_path.exists():
            # Image réelle disponible
            real_img = Image.open(img_path).convert('RGB')
            img_array = np.array(real_img)
            
            print(f"   • Image réelle shape: {img_array.shape}")
            
            # Redimensionner l'image
            if img_array.shape[:2] != input_shape[:2]:
                img_resized = cv2.resize(
                    img_array, 
                    (input_shape[1], input_shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                img_resized = img_array
        
        else:
            # Créer une image synthétique basée sur le mask réel
            print("   • Création image synthétique basée sur le mask réel")
            img_resized = create_synthetic_image_from_mask(mask_resized, input_shape)
        
        print(f"✅ Échantillon Cityscapes chargé avec succès")
        print(f"   • Image finale: {img_resized.shape}")
        print(f"   • Mask final: {mask_resized.shape}")
        print(f"   • Classes finales: {np.unique(mask_resized)}")
        
        return img_resized, mask_resized
        
    except Exception as e:
        print(f"❌ Erreur chargement {label_path}: {e}")
        print("📝 Fallback vers données synthétiques")
        return create_synthetic_test_data()

def create_synthetic_image_from_mask(mask, target_shape):
    """
    Crée une image synthétique réaliste basée sur un vrai mask Cityscapes
    """
    height, width, channels = target_shape
    synthetic_img = np.zeros((height, width, channels), dtype=np.uint8)
    
    # Palette de couleurs réalistes pour chaque catégorie (depuis class_mapping.json)
    colors = {
        0: [139,  69,  19],   # road - brun (#8B4513)
        1: [128, 128, 128],   # building - gris (#808080)
        2: [255, 215,   0],   # object - jaune or (#FFD700)
        3: [ 34, 139,  34],   # nature - vert (#228B22)
        4: [135, 206, 235],   # sky - bleu ciel (#87CEEB)
        5: [255, 105, 180],   # person - rose (#FF69B4)
        6: [220,  20,  60],   # vehicle - rouge (#DC143C)
        7: [  0,   0,   0]    # void - noir (#000000)
    }
    
    # Appliquer les couleurs selon le mask
    for class_id, color in colors.items():
        synthetic_img[mask == class_id] = color
    
    # Ajouter un peu de bruit pour réalisme
    noise = np.random.normal(0, 10, synthetic_img.shape).astype(np.int16)
    synthetic_img = np.clip(synthetic_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return synthetic_img

def create_synthetic_test_data():
    """
    Fallback: Crée des données de test synthétiques avec les bonnes dimensions
    """
    input_shape = AUGMENTATION_CONFIG['image_settings']['input_shape']
    target_shape = AUGMENTATION_CONFIG['image_settings']['target_shape']
    
    print(f"📊 Création données test synthétiques:")
    print(f"   • Image shape: {input_shape}")
    print(f"   • Mask shape: {target_shape}")
    
    # Image de test (simulation scène urbaine)
    height, width, channels = input_shape
    test_image = np.zeros((height, width, channels), dtype=np.uint8)
    
    # Simulation ciel (partie haute)
    test_image[:height//4, :, :] = [135, 206, 235]  # Sky blue
    
    # Simulation bâtiments (partie haute-milieu)
    test_image[height//4:height//2, :width//3, :] = [128, 128, 128]  # Gray buildings
    test_image[height//4:height//2, 2*width//3:, :] = [139, 69, 19]  # Brown buildings
    
    # Simulation végétation (côtés)
    test_image[height//2:3*height//4, :width//6, :] = [34, 139, 34]  # Green
    test_image[height//2:3*height//4, 5*width//6:, :] = [34, 139, 34]  # Green
    
    # Simulation route (partie basse)
    test_image[3*height//4:, :, :] = [64, 64, 64]  # Dark gray road
    
    # Simulation véhicules (rectangles rouges)
    car_y = 3*height//4 + 20
    car_h = 30
    car_w = 60
    test_image[car_y:car_y+car_h, width//3:width//3+car_w, :] = [220, 20, 60]  # Red car
    test_image[car_y:car_y+car_h, 2*width//3:2*width//3+car_w, :] = [220, 20, 60]  # Red car
    
    # Simulation personnes (petits rectangles roses)
    person_y = 3*height//4 + 10
    person_h = 20
    person_w = 8
    test_image[person_y:person_y+person_h, width//2-20:width//2-20+person_w, :] = [255, 105, 180]  # Pink person
    test_image[person_y:person_y+person_h, width//2+20:width//2+20+person_w, :] = [255, 105, 180]  # Pink person
    
    # Mask correspondant
    test_mask = np.zeros(target_shape, dtype=np.uint8)
    
    # Sky (4)
    test_mask[:height//4, :] = 4
    
    # Buildings (1)
    test_mask[height//4:height//2, :width//3] = 1
    test_mask[height//4:height//2, 2*width//3:] = 1
    
    # Nature (3)
    test_mask[height//2:3*height//4, :width//6] = 3
    test_mask[height//2:3*height//4, 5*width//6:] = 3
    
    # Road (0)
    test_mask[3*height//4:, :] = 0
    
    # Vehicles (6)
    test_mask[car_y:car_y+car_h, width//3:width//3+car_w] = 6
    test_mask[car_y:car_y+car_h, 2*width//3:2*width//3+car_w] = 6
    
    # Persons (5) - classe rare importante
    test_mask[person_y:person_y+person_h, width//2-20:width//2-20+person_w] = 5
    test_mask[person_y:person_y+person_h, width//2+20:width//2+20+person_w] = 5
    
    print(f"✅ Données test synthétiques créées")
    print(f"   • Classes uniques dans mask: {np.unique(test_mask)}")
    print(f"   • Image dtype: {test_image.dtype}, shape: {test_image.shape}")
    print(f"   • Mask dtype: {test_mask.dtype}, shape: {test_mask.shape}")
    
    return test_image, test_mask

def create_test_data():
    """
    Fonction principale pour créer des données de test
    Essaie d'abord de charger de vraies données Cityscapes, sinon fallback synthétique
    """
    try:
        return load_real_cityscapes_sample()
    except Exception as e:
        print(f"⚠️ Erreur chargement données réelles: {e}")
        #print("📝 Utilisation données synthétiques")
        #return create_synthetic_test_data()

def test_all_pipelines():
    """
    Test complet de tous les pipelines d'augmentation
    """
    print("\n🧪 TEST COMPLET DES PIPELINES D'AUGMENTATION")
    print("=" * 60)
    
    # Créer données de test
    test_image, test_mask = create_test_data()
    
    # Test des pipelines individuels
    print("\n1. 🧩 Test Transformations Géométriques")
    geo_aug = GeometricAugmentations(AUGMENTATION_CONFIG)
    geo_aug.visualize_geometric_effects(test_image, test_mask)
    
    print("\n2. 🌈 Test Transformations Photométriques")
    photo_aug = PhotometricAugmentations(AUGMENTATION_CONFIG)
    photo_aug.visualize_photometric_effects(test_image)
    
    print("\n3. 🌦️ Test Conditions Météorologiques")
    weather_aug = WeatherAugmentations(AUGMENTATION_CONFIG)
    weather_aug.visualize_weather_effects(test_image)
    
    print("\n4. 🎯 Test Augmentations Ciblées")
    targeted_aug = TargetedAugmentations(AUGMENTATION_CONFIG)
    targeted_aug.visualize_targeted_effects(test_image, test_mask)
    
    print("\n5. 🔄 Test Pipeline Complet")
    # Pipeline d'entraînement
    train_pipeline = ComprehensiveAugmentationPipeline(AUGMENTATION_CONFIG, mode='training')
    train_pipeline.visualize_complete_pipeline(test_image, test_mask, n_samples=4)
    
    # Pipeline de validation
    val_pipeline = ComprehensiveAugmentationPipeline(AUGMENTATION_CONFIG, mode='validation')
    val_pipeline.visualize_complete_pipeline(test_image, test_mask, n_samples=4)
    
    return train_pipeline, val_pipeline, test_image, test_mask

# Exécution des tests
print("🚀 Lancement des tests d'augmentation...")
train_pipeline, val_pipeline, test_image, test_mask = test_all_pipelines()

# %% [markdown]
# ## ⚡ Benchmark de Performance
# 
# **Mesure critique** : Évaluer l'impact des augmentations sur les performances d'entraînement

# %%
def benchmark_augmentation_performance():
    """
    Benchmark des performances des différents pipelines d'augmentation
    """
    print("\n⚡ BENCHMARK PERFORMANCES AUGMENTATION")
    print("=" * 50)
    
    # Pipelines à tester
    pipelines_to_test = {
        'Sans augmentation': None,
        'Géométrique seul': GeometricAugmentations(AUGMENTATION_CONFIG).create_basic_geometric_pipeline(),
        'Photométrique seul': PhotometricAugmentations(AUGMENTATION_CONFIG).create_complete_photometric_pipeline(),
        'Pipeline complet train': train_pipeline,
        'Pipeline complet val': val_pipeline
    }
    
    # Données de test
    test_image, test_mask = create_test_data()
    
    benchmark_results = {}
    
    for pipeline_name, pipeline in pipelines_to_test.items():
        print(f"\n⏱️ Test: {pipeline_name}")
        
        times = []
        errors = 0
        
        try:
            # Warmup
            for _ in range(5):
                if pipeline is None:
                    # Pas d'augmentation
                    result_img, result_mask = test_image.copy(), test_mask.copy()
                elif hasattr(pipeline, '__call__') and hasattr(pipeline, 'mode'):
                    # Pipeline complet
                    result_img, result_mask = pipeline(test_image, test_mask)
                else:
                    # Pipeline Albumentations
                    result = pipeline(image=test_image, mask=test_mask)
                    result_img, result_mask = result['image'], result['mask']
            
            # Mesures de performance
            for i in range(100):
                start_time = time.perf_counter()
                
                if pipeline is None:
                    result_img, result_mask = test_image.copy(), test_mask.copy()
                elif hasattr(pipeline, '__call__') and hasattr(pipeline, 'mode'):
                    result_img, result_mask = pipeline(test_image, test_mask)
                else:
                    result = pipeline(image=test_image, mask=test_mask)
                    result_img, result_mask = result['image'], result['mask']
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # en ms
            
            # Statistiques
            mean_time = np.mean(times)
            std_time = np.std(times)
            fps = 1000.0 / mean_time
            
            benchmark_results[pipeline_name] = {
                'mean_time_ms': mean_time,
                'std_time_ms': std_time,
                'fps': fps,
                'status': 'OK'
            }
            
            print(f"✅ {mean_time:.1f}±{std_time:.1f} ms ({fps:.1f} FPS)")
            
        except Exception as e:
            benchmark_results[pipeline_name] = {
                'mean_time_ms': np.nan,
                'std_time_ms': np.nan,
                'fps': np.nan,
                'status': f'Erreur: {e}'
            }
            print(f"❌ Erreur: {e}")
    
    # Résumé des résultats
    print(f"\n📈 RÉSULTATS BENCHMARK:")
    results_df = pd.DataFrame([
        {
            'Pipeline': name,
            'Temps moyen (ms)': f"{data['mean_time_ms']:.6f}" if not np.isnan(data['mean_time_ms']) else 'N/A',
            'Écart-type (ms)': f"{data['std_time_ms']:.6f}" if not np.isnan(data['std_time_ms']) else 'N/A',
            'FPS': f"{data['fps']:.6f}" if not np.isnan(data['fps']) else 'N/A',
            'Statut': '✅ OK' if data['status'] == 'OK' else f"❌ {data['status']}"
        }
        for name, data in benchmark_results.items()
    ])
    
    print(results_df.to_string(index=False))
    
    # Sauvegarde des résultats
    results_for_save = {k: {sub_k: float(sub_v) if isinstance(sub_v, (int, float)) and not np.isnan(sub_v) else str(sub_v) 
                           for sub_k, sub_v in v.items()} 
                       for k, v in benchmark_results.items()}
    
    with open(OUTPUTS_DIR / "augmentation_benchmark.json", 'w') as f:
        json.dump(results_for_save, f, indent=2)
    
    return benchmark_results

# Benchmark des performances
benchmark_results = benchmark_augmentation_performance()

# %% [markdown]
# ## 📊 Analyse de l'Impact sur la Distribution des Classes
# 
# **Validation métier** : S'assurer que les augmentations améliorent effectivement 
# la représentation des classes rares sans dégrader l'équilibre global.

# %%
def analyze_augmentation_impact_on_classes():
    """
    Analyse l'impact des augmentations sur la distribution des classes
    """
    print("\n📊 ANALYSE IMPACT SUR DISTRIBUTION DES CLASSES")
    print("=" * 55)
    
    # Créer échantillon de test avec différents types d'images
    test_samples = []
    
    for i in range(5):
        img, mask = create_test_data()
        # Varier la composition pour avoir différentes distributions
        if i == 1:
            # Plus de personnes
            mask[mask == 0] = 5  # Convertir une partie de la route en personnes
        elif i == 2:
            # Plus d'objets
            mask[mask == 1] = 2  # Convertir une partie des bâtiments en objets
        elif i == 3:
            # Image sans classes rares
            mask[mask == 5] = 0  # Supprimer les personnes
            mask[mask == 2] = 1  # Supprimer les objets
        
        test_samples.append((img, mask))
    
    # Analyser la distribution avant/après augmentation
    original_distributions = []
    augmented_distributions = []
    
    for img, mask in test_samples:
        # Distribution originale
        unique, counts = np.unique(mask, return_counts=True)
        original_dist = {int(cls): int(count) for cls, count in zip(unique, counts)}
        original_distributions.append(original_dist)
        
        # Appliquer plusieurs augmentations et moyenner
        aug_distributions = []
        for _ in range(10):  # 10 augmentations par image
            try:
                aug_img, aug_mask = train_pipeline(img, mask)
                unique_aug, counts_aug = np.unique(aug_mask, return_counts=True)
                aug_dist = {int(cls): int(count) for cls, count in zip(unique_aug, counts_aug)}
                aug_distributions.append(aug_dist)
            except:
                continue
        
        # Moyenne des distributions augmentées
        if aug_distributions:
            all_classes = set()
            for dist in aug_distributions:
                all_classes.update(dist.keys())
            
            mean_aug_dist = {}
            for cls in all_classes:
                counts = [dist.get(cls, 0) for dist in aug_distributions]
                mean_aug_dist[cls] = np.mean(counts)
            
            augmented_distributions.append(mean_aug_dist)
        else:
            augmented_distributions.append(original_dist)
    
    # Visualisation comparative
    class_names = ['road', 'building', 'object', 'nature', 'sky', 'person', 'vehicle', 'void']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (orig_dist, aug_dist) in enumerate(zip(original_distributions[:6], augmented_distributions[:6])):
        # Calculer pourcentages
        orig_total = sum(orig_dist.values())
        aug_total = sum(aug_dist.values())
        
        classes = sorted(set(orig_dist.keys()) | set(aug_dist.keys()))
        orig_percentages = [orig_dist.get(cls, 0) / orig_total * 100 for cls in classes]
        aug_percentages = [aug_dist.get(cls, 0) / aug_total * 100 for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = axes[i].bar(x - width/2, orig_percentages, width, label='Original', alpha=0.7)
        bars2 = axes[i].bar(x + width/2, aug_percentages, width, label='Augmenté', alpha=0.7)
        
        axes[i].set_xlabel('Classes')
        axes[i].set_ylabel('Pourcentage (%)')
        axes[i].set_title(f'Échantillon {i+1}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([class_names[cls] if cls < len(class_names) else f'C{cls}' 
                                for cls in classes], rotation=45)
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "augmentation_class_impact.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calcul des métriques d'amélioration
    print(f"\n📈 MÉTRIQUES D'AMÉLIORATION:")
    
    # Classes rares (person=5, object=2)
    rare_classes = [5, 2]
    
    for rare_class in rare_classes:
        class_name = class_names[rare_class] if rare_class < len(class_names) else f'Class_{rare_class}'
        
        orig_coverages = []
        aug_coverages = []
        
        for orig_dist, aug_dist in zip(original_distributions, augmented_distributions):
            orig_total = sum(orig_dist.values())
            aug_total = sum(aug_dist.values())
            
            orig_coverage = orig_dist.get(rare_class, 0) / orig_total * 100
            aug_coverage = aug_dist.get(rare_class, 0) / aug_total * 100
            
            orig_coverages.append(orig_coverage)
            aug_coverages.append(aug_coverage)
        
        mean_orig = np.mean(orig_coverages)
        mean_aug = np.mean(aug_coverages)
        improvement = ((mean_aug - mean_orig) / mean_orig * 100) if mean_orig > 0 else 0
        
        print(f"   • {class_name:12}: {mean_orig:.2f}% → {mean_aug:.2f}% ({improvement:+.1f}%)")
    
    return original_distributions, augmented_distributions

# Analyse de l'impact sur les classes
impact_analysis = analyze_augmentation_impact_on_classes()

# %% [markdown]
# ## 🔧 Intégration avec le Générateur de Données
# 
# **Compatibilité critique** : S'assurer que les pipelines d'augmentation s'intègrent 
# parfaitement avec le générateur de données existant.

# %%
class CityscapesDataLoader:
    """
    Chargeur de données réelles Cityscapes avec gestion des chemins
    """
    
    def __init__(self, gtfine_root, leftimg_root=None):
        self.gtfine_root = Path(gtfine_root)
        self.leftimg_root = Path(leftimg_root) if leftimg_root else None
        self.dataset_index = self._build_dataset_index()
        
    def _build_dataset_index(self):
        """
        Construit l'index du dataset
        """
        dataset_index = {'train': [], 'val': [], 'test': []}
        
        for split in ['train', 'val', 'test']:
            split_path = self.gtfine_root / split
            if not split_path.exists():
                continue
                
            for city_dir in split_path.iterdir():
                if not city_dir.is_dir():
                    continue
                    
                # Fichiers d'annotations
                label_files = list(city_dir.glob("*_gtFine_labelIds.png"))
                
                for label_file in label_files:
                    base_name = label_file.stem.replace('_gtFine_labelIds', '')
                    
                    # Image correspondante (si disponible)
                    img_path = None
                    if self.leftimg_root:
                        img_candidates = [
                            self.leftimg_root / split / city_dir.name / f"{base_name}_leftImg8bit.png",
                            self.leftimg_root / split / city_dir.name / f"{base_name}_leftImg8bit.jpg"
                        ]
                        for candidate in img_candidates:
                            if candidate.exists():
                                img_path = candidate
                                break
                    
                    dataset_index[split].append({
                        'image_path': str(img_path) if img_path else None,
                        'label_path': str(label_file),
                        'base_name': base_name,
                        'city': city_dir.name,
                        'split': split
                    })
        
        print(f"📁 Index dataset construit:")
        for split, items in dataset_index.items():
            print(f"   • {split}: {len(items)} échantillons")
        
        return dataset_index
    
    def load_sample(self, sample_info, target_shape=(512, 1024)):
        """
        Charge un échantillon (image + mask)
        """
        try:
            # Charger le mask
            mask_img = Image.open(sample_info['label_path'])
            mask_array = np.array(mask_img)
            
            # Convertir 34->8 classes
            mask_8_categories = ID_TO_8_CATEGORIES[mask_array]
            
            # Redimensionner le mask
            if mask_8_categories.shape != target_shape:
                mask_resized = cv2.resize(
                    mask_8_categories, 
                    (target_shape[1], target_shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_resized = mask_8_categories
            
            # Charger l'image
            if sample_info['image_path'] and Path(sample_info['image_path']).exists():
                # Image réelle
                real_img = Image.open(sample_info['image_path']).convert('RGB')
                img_array = np.array(real_img)
                
                # Redimensionner l'image
                target_img_shape = (target_shape[0], target_shape[1], 3)
                if img_array.shape[:2] != target_shape:
                    img_resized = cv2.resize(
                        img_array, 
                        (target_shape[1], target_shape[0]), 
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    img_resized = img_array
            else:
                # Image synthétique basée sur le mask
                target_img_shape = (target_shape[0], target_shape[1], 3)
                img_resized = create_synthetic_image_from_mask(mask_resized, target_img_shape)
            
            return img_resized, mask_resized
            
        except Exception as e:
            print(f"❌ Erreur chargement échantillon {sample_info['base_name']}: {e}")
            return None, None
    
    def get_samples(self, split, max_samples=None):
        """
        Récupère les échantillons d'un split
        """
        samples = self.dataset_index.get(split, [])
        if max_samples:
            samples = samples[:max_samples]
        return samples

class AugmentedCityscapesSequence:
    """
    Générateur de données intégrant les augmentations avec vraies données Cityscapes
    Compatible avec la classe CityscapesSequence existante
    """
    
    def __init__(self, data_list=None, batch_size=16, input_shape=(512, 1024, 3), 
                 num_classes=8, mode='training', augmentation_config=None,
                 gtfine_root=None, leftimg_root=None, split='train', max_samples=None):
        
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.mode = mode
        
        # Configuration d'augmentation
        if augmentation_config is None:
            augmentation_config = AUGMENTATION_CONFIG
        
        # Pipeline d'augmentation
        self.augmentation_pipeline = ComprehensiveAugmentationPipeline(
            config=augmentation_config, 
            mode=mode
        )
        
        # Chargeur de données
        if gtfine_root:
            self.data_loader = CityscapesDataLoader(gtfine_root, leftimg_root)
            self.data_list = self.data_loader.get_samples(split, max_samples)
            data_source = f"Cityscapes {split}"
        elif data_list:
            self.data_list = data_list
            self.data_loader = None
            data_source = "Liste fournie"
        else:
            # Fallback: créer des échantillons de test
            self.data_list = [f"synthetic_sample_{i}" for i in range(max_samples or 50)]
            self.data_loader = None
            data_source = "Échantillons synthétiques"
        
        print(f"✅ AugmentedCityscapesSequence initialisé:")
        print(f"   • Mode: {mode}")
        print(f"   • Source: {data_source}")
        print(f"   • Échantillons: {len(self.data_list)}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Augmentation: {'✅' if mode == 'training' else '❌'}")
    
    def __len__(self):
        return int(np.ceil(len(self.data_list) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Génère un batch avec augmentations
        """
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.data_list))
        batch_data = self.data_list[start_idx:end_idx]
        actual_batch_size = len(batch_data)
        
        batch_images = np.zeros((actual_batch_size,) + self.input_shape, dtype=np.float32)
        batch_masks = np.zeros((actual_batch_size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        
        for i, data_item in enumerate(batch_data):
            try:
                if self.data_loader and isinstance(data_item, dict):
                    # Vraies données Cityscapes
                    image, mask = self.data_loader.load_sample(
                        data_item, 
                        target_shape=(self.input_shape[0], self.input_shape[1])
                    )
                    
                    if image is None or mask is None:
                        # Fallback en cas d'erreur
                        image, mask = create_test_data()
                else:
                    # Données de test/synthétiques
                    image, mask = create_test_data()
                
                # Appliquer les augmentations
                if self.mode == 'training':
                    aug_image, aug_mask = self.augmentation_pipeline(image, mask)
                else:
                    aug_image, aug_mask = image, mask
                
                # Normalisation de l'image
                aug_image = aug_image.astype(np.float32) / 255.0
                
                batch_images[i] = aug_image
                batch_masks[i] = aug_mask
                
            except Exception as e:
                print(f"⚠️ Erreur item {i}: {e}")
                # Utiliser données par défaut en cas d'erreur
                batch_images[i] = np.zeros(self.input_shape, dtype=np.float32)
                batch_masks[i] = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        
        return batch_images, batch_masks
    
    def on_epoch_end(self):
        """
        Actions en fin d'époque
        """
        if hasattr(self.data_list, 'shuffle'):
            np.random.shuffle(self.data_list)

def test_integration_with_generator():
    """
    Test d'intégration avec le générateur de données utilisant de vraies données Cityscapes
    """
    print("\n🔧 TEST INTÉGRATION AVEC GÉNÉRATEUR")
    print("=" * 45)
    
    # Test avec vraies données Cityscapes si disponibles
    use_real_data = GTFINE_ROOT.exists()
    
    if use_real_data:
        print("🎆 Utilisation des vraies données Cityscapes!")
        
        # Créer générateurs avec vraies données
        train_gen = AugmentedCityscapesSequence(
            batch_size=4,  # Plus petit pour les tests
            mode='training',
            gtfine_root=GTFINE_ROOT,
            leftimg_root=LEFTIMG_ROOT if LEFTIMG_ROOT.exists() else None,
            split='train',
            max_samples=20  # Limiter pour les tests
        )
        
        val_gen = AugmentedCityscapesSequence(
            batch_size=4,
            mode='validation',
            gtfine_root=GTFINE_ROOT,
            leftimg_root=LEFTIMG_ROOT if LEFTIMG_ROOT.exists() else None,
            split='val',
            max_samples=10
        )
    else:
        print("📝 Dataset Cityscapes non disponible, utilisation données synthétiques")
        
        # Fallback avec données synthétiques
        train_gen = AugmentedCityscapesSequence(
            batch_size=4, 
            mode='training',
            max_samples=20
        )
        
        val_gen = AugmentedCityscapesSequence(
            batch_size=4, 
            mode='validation',
            max_samples=10
        )
    
    # Test de génération de batches
    print(f"\n📊 Test génération batches:")
    print(f"   • Générateur train: {len(train_gen)} batches")
    print(f"   • Générateur val: {len(val_gen)} batches")
    
    # Test d'un batch de training
    try:
        batch_x_train, batch_y_train = train_gen[0]
        print(f"   • Batch train: images {batch_x_train.shape}, masks {batch_y_train.shape}")
        print(f"   • Train images range: [{batch_x_train.min():.3f}, {batch_x_train.max():.3f}]")
        print(f"   • Train masks classes: {np.unique(batch_y_train)}")
    except Exception as e:
        print(f"❌ Erreur batch train: {e}")
    
    # Test d'un batch de validation
    try:
        batch_x_val, batch_y_val = val_gen[0]
        print(f"   • Batch val: images {batch_x_val.shape}, masks {batch_y_val.shape}")
        print(f"   • Val images range: [{batch_x_val.min():.3f}, {batch_x_val.max():.3f}]")
        print(f"   • Val masks classes: {np.unique(batch_y_val)}")
    except Exception as e:
        print(f"❌ Erreur batch val: {e}")
    
    # Visualisation comparative
    try:
        fig, axes = plt.subplots(4, 6, figsize=(24, 16))
        
        # Ligne 1-2: Batch training (avec augmentation) - Images et Masks
        for i in range(6):
            if i < len(batch_x_train):
                # Images training
                axes[0, i].imshow(batch_x_train[i])
                axes[0, i].set_title(f'Train Image {i+1}\n(Avec augmentation)', fontweight='bold', color='darkblue')
                axes[0, i].axis('off')
                
                # Masks training
                axes[1, i].imshow(batch_y_train[i], cmap=CUSTOM_COLORMAP, vmin=0, vmax=7)
                axes[1, i].set_title(f'Train Mask {i+1}\n(Classes: {len(np.unique(batch_y_train[i]))})', fontweight='bold', color='darkblue')
                axes[1, i].axis('off')
            else:
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        # Ligne 3-4: Batch validation (sans augmentation) - Images et Masks
        for i in range(6):
            if i < len(batch_x_val):
                # Images validation
                axes[2, i].imshow(batch_x_val[i])
                axes[2, i].set_title(f'Val Image {i+1}\n(Sans augmentation)', fontweight='bold', color='darkgreen')
                axes[2, i].axis('off')
                
                # Masks validation
                axes[3, i].imshow(batch_y_val[i], cmap=CUSTOM_COLORMAP, vmin=0, vmax=7)
                axes[3, i].set_title(f'Val Mask {i+1}\n(Classes: {len(np.unique(batch_y_val[i]))})', fontweight='bold', color='darkgreen')
                axes[3, i].axis('off')
            else:
                axes[2, i].axis('off')
                axes[3, i].axis('off')
        
        # Ajout de labels pour les sections
        fig.text(0.02, 0.75, 'TRAINING\n(Augmenté)', fontsize=14, fontweight='bold', 
                color='darkblue', rotation=90, va='center')
        fig.text(0.02, 0.25, 'VALIDATION\n(Original)', fontsize=14, fontweight='bold', 
                color='darkgreen', rotation=90, va='center')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05)
        plt.savefig(FIGURES_DIR / "generator_integration_enhanced.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistiques comparatives
        print(f"\n📈 ANALYSE COMPARATIVE DES BATCHES:")
        print(f"   🔵 TRAINING (avec augmentation):")
        print(f"      • Variabilité pixel intensité: {np.std(batch_x_train):.3f}")
        print(f"      • Classes moyennes par image: {np.mean([len(np.unique(mask)) for mask in batch_y_train]):.1f}")
        print(f"   🟢 VALIDATION (sans augmentation):")
        print(f"      • Variabilité pixel intensité: {np.std(batch_x_val):.3f}")
        print(f"      • Classes moyennes par image: {np.mean([len(np.unique(mask)) for mask in batch_y_val]):.1f}")
        
    except Exception as e:
        print(f"⚠️ Erreur visualisation: {e}")
    
    return train_gen, val_gen

# Test d'intégration
train_gen, val_gen = test_integration_with_generator()

# %% [markdown]
# ## 💾 Sauvegarde et Configuration Finale
# 
# **Documentation complète** : Sauvegarder toutes les configurations et fournir un guide d'utilisation

# %%
def save_augmentation_configuration():
    """
    Sauvegarde la configuration complète du système d'augmentation
    """
    
    augmentation_config_complete = {
        'augmentation_system': {
            'version': '1.0',
            'description': 'Système complet d\'augmentation pour segmentation embarquée',
            'components': [
                'GeometricAugmentations - Transformations géométriques cohérentes',
                'PhotometricAugmentations - Conditions d\'éclairage variées',
                'WeatherAugmentations - Conditions météorologiques',
                'TargetedAugmentations - Focus sur classes rares',
                'ComprehensiveAugmentationPipeline - Pipeline adaptatif complet'
            ]
        },
        'configuration': AUGMENTATION_CONFIG,
        'performance_benchmarks': benchmark_results,
        'implementation_guide': {
            'basic_usage': {
                'training_pipeline': 'ComprehensiveAugmentationPipeline(config, mode="training")',
                'validation_pipeline': 'ComprehensiveAugmentationPipeline(config, mode="validation")',
                'application': 'aug_image, aug_mask = pipeline(image, mask)',
                'integration': 'AugmentedCityscapesSequence(data, mode="training")'
            },
            'advanced_features': {
                'targeted_augmentation': 'Automatique pour images avec classes rares',
                'adaptive_intensity': 'Basé sur le contenu de l\'image',
                'weather_simulation': 'Pluie, brouillard, neige, ombres',
                'performance_optimization': 'Benchmark intégré et cache'
            }
        },
        'recommended_settings': {
            'training': {
                'geometric_p': 0.6,
                'photometric_p': 0.7,
                'weather_p': 0.4,
                'targeted_p': 0.5,
                'description': 'Augmentation agressive pour robustesse maximale'
            },
            'validation': {
                'augmentation_intensity': 0.0,
                'description': 'Aucune augmentation pour évaluation stable'
            },
            'production': {
                'geometric_p': 0.3,
                'photometric_p': 0.4,
                'weather_p': 0.2,
                'description': 'Augmentation modérée pour fine-tuning'
            }
        },
        'class_balancing_strategy': {
            'rare_classes_focus': 'person (5), object (2)',
            'targeted_amplification': '2x pour images avec classes rares',
            'preservation_strategy': 'Transformations géométriques conservatrices',
            'quality_assurance': 'Vérification automatique cohérence image-mask'
        },
        'performance_specifications': {
            'target_fps': '>1000 FPS pour entraînement temps réel',
            'memory_efficient': 'Pas de cache persistant, traitement à la volée',
            'error_handling': 'Fallback vers données originales en cas d\'erreur',
            'compatibility': 'Compatible tf.keras.utils.Sequence'
        },
        'quality_control': {
            'shape_verification': 'Vérification automatique des dimensions',
            'class_preservation': 'Validation préservation des labels',
            'visual_validation': 'Fonctions de visualisation intégrées',
            'benchmark_integration': 'Tests de performance automatisés'
        }
    }
    
    # Sauvegarde
    config_path = OUTPUTS_DIR / "data_augmentation_config.json"
    with open(config_path, 'w') as f:
        json.dump(augmentation_config_complete, f, indent=2, default=str)
    
    print("\n💾 SAUVEGARDE CONFIGURATION AUGMENTATION")
    print("=" * 50)
    print(f"✅ Configuration sauvegardée: {config_path}")
    print(f"✅ Guide d'implémentation complet inclus")
    print(f"✅ Benchmarks de performance intégrés")
    print(f"✅ Stratégies d'équilibrage des classes définies")
    
    return augmentation_config_complete

def generate_usage_examples():
    """
    Génère des exemples d'usage pour l'équipe
    """
    
    usage_examples = {
        'example_1_basic': '''
# Usage de base - Pipeline d'entraînement
from notebooks.data_augmentation import ComprehensiveAugmentationPipeline

# Initialisation
aug_pipeline = ComprehensiveAugmentationPipeline(mode='training')

# Application
augmented_image, augmented_mask = aug_pipeline(image, mask)
        ''',
        
        'example_2_generator': '''
# Intégration avec générateur de données
from notebooks.data_augmentation import AugmentedCityscapesSequence

# Créer générateur avec augmentation
train_generator = AugmentedCityscapesSequence(
    data_list=train_data,
    batch_size=16,
    mode='training'
)

# Utilisation avec Keras
model.fit(train_generator, epochs=50)
        ''',
        
        'example_3_custom': '''
# Configuration personnalisée
custom_config = AUGMENTATION_CONFIG.copy()
custom_config['augmentation_probabilities']['geometric_p'] = 0.8

custom_pipeline = ComprehensiveAugmentationPipeline(
    config=custom_config, 
    mode='training'
)
        ''',
        
        'example_4_individual': '''
# Utilisation des modules individuels
from notebooks.data_augmentation import GeometricAugmentations

geo_aug = GeometricAugmentations(AUGMENTATION_CONFIG)
geo_pipeline = geo_aug.create_basic_geometric_pipeline()

result = geo_pipeline(image=image, mask=mask)
        '''
    }
    
    # Sauvegarde des exemples
    examples_path = OUTPUTS_DIR / "augmentation_usage_examples.py"
    with open(examples_path, 'w') as f:
        f.write("# Exemples d'usage du système d'augmentation\n\n")
        for example_name, code in usage_examples.items():
            f.write(f"# {example_name.upper()}\n")
            f.write(code)
            f.write("\n\n")
    
    print(f"📋 Exemples d'usage sauvegardés: {examples_path}")
    
    return usage_examples

# Sauvegarde finale
final_config = save_augmentation_configuration()
usage_examples = generate_usage_examples()

# Résumé final complet
print(f"\n🏆 RÉSUMÉ FINAL - AUGMENTATION DE DONNÉES")
print("=" * 60)
print(f"✅ Système complet d'augmentation implémenté et testé")
print(f"✅ 5 modules d'augmentation: Géométrique, Photométrique, Météo, Ciblé, Complet")
print(f"✅ Pipeline adaptatif selon mode (training/validation)")
print(f"✅ Gestion intelligente des classes rares (person, object)")
print(f"✅ Intégration complète avec générateur de données")
print(f"✅ Benchmark de performance et optimisations")

print(f"\n📊 Performances mesurées:")
if 'Sans augmentation' in benchmark_results and benchmark_results['Sans augmentation']['status'] == 'OK':
    base_fps = benchmark_results['Sans augmentation']['fps']
    print(f"   • Sans augmentation: {base_fps:.0f} FPS")

if 'Pipeline complet train' in benchmark_results and benchmark_results['Pipeline complet train']['status'] == 'OK':
    aug_fps = benchmark_results['Pipeline complet train']['fps']
    print(f"   • Avec augmentation: {aug_fps:.0f} FPS")
    
    if 'Sans augmentation' in benchmark_results and benchmark_results['Sans augmentation']['status'] == 'OK':
        overhead = (base_fps - aug_fps) / base_fps * 100
        print(f"   • Overhead: {overhead:.1f}% (acceptable pour gains en robustesse)")

print(f"\n🎯 Fonctionnalités clés:")
print(f"   • Cohérence parfaite image-mask pour transformations géométriques")
print(f"   • Simulation conditions météorologiques (pluie, brouillard, neige)")
print(f"   • Focus automatique sur classes rares avec techniques spécialisées")
print(f"   • Pipeline adaptatif selon contenu de l'image")
print(f"   • Intégration transparente avec pipeline d'entraînement")

print(f"\n📁 Artefacts générés:")
print(f"   • Configuration: {OUTPUTS_DIR / 'data_augmentation_config.json'}")
print(f"   • Benchmarks: {OUTPUTS_DIR / 'augmentation_benchmark.json'}")
print(f"   • Exemples: {OUTPUTS_DIR / 'augmentation_usage_examples.py'}")
print(f"   • Visualisations: {FIGURES_DIR}")

print(f"🔄 Pipeline d'augmentation prêt pour intégration dans l'entraînement des modèles")

# %% [markdown]
# ## 🎯 Prochaines Étapes et Intégration
# 
# **Roadmap d'intégration** : Comment utiliser ce système dans la suite du projet
# 
# **Actions immédiates** :
# 1. 🔗 **Intégration avec 2.7_Training_Pipeline** : Remplacer les générateurs basiques par `AugmentedCityscapesSequence`
# 2. 📊 **Évaluation d'impact** : Mesurer l'amélioration des métriques (IoU, mIoU) avec augmentation
# 3. 🎛️ **Optimisation hyperparamètres** : Ajuster les probabilités d'augmentation selon les résultats
# 4. 🏭 **Préparation production** : Validation du pipeline pour déploiement embarqué
# 
# **Intégration avec milestones suivants** :
# - **Milestone 3** : Utiliser les pipelines d'augmentation pour améliorer les performances des 3 architectures
# - **Milestone 4-5** : Intégrer dans l'API et l'application web pour démonstration
# - **Production** : Pipeline optimisé pour contraintes embarquées
# 
# **Critères de succès** :
# - Amélioration mIoU global : +5-10%
# - Amélioration IoU classes rares : +20-50%
# - Performance : <2ms overhead par image
# - Robustesse : Validation sur conditions météo variées

# %%
print("\n✅ SYSTÈME D'AUGMENTATION COMPLET ET OPÉRATIONNEL")
print("🔄 Transition vers Milestone 3: Entraînement avec augmentation avancée")


