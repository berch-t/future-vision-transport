# %% [markdown]
# # 🏗️ Model Implementation - Architectures de Segmentation
# 
# ## 🎯 Objectifs
# 
# **Mission** : Implémenter 5 architectures de segmentation en Keras/TensorFlow 
# 
# **Stratégie d'implémentation** :
# 1. **Structure modulaire** : Classes réutilisables et configurables
# 2. **Transfer learning** : Backbones pré-entraînés ImageNet
# 3. **Optimisation embarquée** : Architectures adaptées aux contraintes
# 4. **Tests unitaires** : Validation des dimensions et fonctionnalités
# 
# **Architectures implémentées** :
# - U-Net + EfficientNet (Encoder-Decoder classique)
# - DeepLabV3+ + MobileNet (Atrous convolutions + efficacité)  
# - Segformer-B0 (Vision Transformer léger)
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
import matplotlib.pyplot as plt

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2,
    MobileNetV2, VGG16
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# Configuration
plt.style.use('seaborn-v0_8')

# Chemins du projet
PROJECT_ROOT = Path("C:/Tonton/OpenClassrooms/Projet_7_traiter_images_systeme_embarque_voiture_autonome")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = NOTEBOOKS_DIR / "outputs"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"

print("✅ Configuration chargée")
print(f"🔗 TensorFlow: {tf.__version__}")
print(f"🎯 Implémentation de 5 architectures modulaires")

# %% [markdown]
# ## ⚙️ Configuration Expérimentale

# %%
# Configuration globale des modèles
MODEL_CONFIG = {
    'input_shape': (512, 1024, 3),
    'num_classes': 8,
    'activation': 'softmax',
    'dropout_rate': 0.3,
    'batch_norm': True,
    'l2_reg': 1e-4
}

# Charger le mapping des classes
with open(OUTPUTS_DIR / "class_mapping.json", 'r') as f:
    class_mapping = json.load(f)

print("🎯 Configuration des modèles:")
print(f"   • Input shape: {MODEL_CONFIG['input_shape']}")
print(f"   • Nombre de classes: {MODEL_CONFIG['num_classes']}")
print(f"   • Régularisation L2: {MODEL_CONFIG['l2_reg']}")

# %% [markdown]
# ## 🏗️ Classe de Base pour Modèles de Segmentation

# %%
class SegmentationModel:
    """
    Classe de base pour tous les modèles de segmentation.
    Fournit une interface commune et des utilities partagées.
    """
    
    def __init__(self, input_shape=(512, 1024, 3), num_classes=8, name="BaseSegmentationModel"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = None
        
    def build_model(self):
        """À implémenter dans les classes filles"""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=None):
        """Compile le modèle avec les paramètres spécifiés"""
        if metrics is None:
            metrics = ['accuracy']
            
        if self.model is None:
            raise ValueError("Model must be built before compilation")
            
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"✅ Modèle {self.name} compilé")
        
    def get_model_info(self):
        """Retourne les informations du modèle"""
        if self.model is None:
            return {"error": "Model not built"}
            
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        return {
            'name': self.name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': self.input_shape,
            'output_shape': self.model.output_shape,
            'layers': len(self.model.layers)
        }
    
    def summary(self):
        """Affiche le résumé du modèle"""
        if self.model is not None:
            return self.model.summary()
        else:
            print("❌ Modèle non construit")

print("✅ Classe de base SegmentationModel définie")

# %% [markdown]
# ## 🔥 Architecture 1: U-Net + EfficientNet
# 
# **Conception** : Encoder-Decoder avec skip connections et backbone efficace

# %%
class UNetEfficientNet(SegmentationModel):
    """
    U-Net avec backbone EfficientNet pour encodage efficace
    """
    
    def __init__(self, backbone='B0', input_shape=(512, 1024, 3), num_classes=8, freeze_backbone=False):
        super().__init__(input_shape, num_classes, f"UNet_EfficientNet{backbone}")
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        
    def build_model(self):
        """Construit le modèle U-Net avec EfficientNet backbone"""
        
        # Sélection du backbone
        if self.backbone_name == 'B0':
            backbone = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.backbone_name == 'B1':
            backbone = EfficientNetB1(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.backbone_name == 'B2':
            backbone = EfficientNetB2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Backbone {self.backbone_name} non supporté")
        
        # Gel du backbone si spécifié
        if self.freeze_backbone:
            backbone.trainable = False
            
        # Points d'extraction pour skip connections
        skip_layer_names = {
            'B0': ['block2a_expand_activation', 'block3a_expand_activation', 
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B1': ['block2a_expand_activation', 'block3a_expand_activation', 
                   'block4a_expand_activation', 'block6a_expand_activation'],
            'B2': ['block2a_expand_activation', 'block3a_expand_activation', 
                   'block4a_expand_activation', 'block6a_expand_activation']
        }
        
        # Extraction des features pour skip connections
        skip_layers = [backbone.get_layer(name).output for name in skip_layer_names[self.backbone_name]]
        
        # Input
        inputs = backbone.input
        
        # Encoder (bottom)
        encoder_output = backbone.output
        
        # Bridge
        bridge = layers.Conv2D(512, 3, padding='same', activation='relu', 
                              kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(encoder_output)
        bridge = layers.BatchNormalization()(bridge)
        bridge = layers.Dropout(MODEL_CONFIG['dropout_rate'])(bridge)
        
        # Decoder avec skip connections
        x = bridge
        
        # Calcul automatique des tailles d'upsampling
        skip_filters = [256, 128, 64, 32]
        
        for i, (skip_layer, filters) in enumerate(zip(reversed(skip_layers), skip_filters)):
            # Upsampling
            x = layers.UpSampling2D(2, interpolation='bilinear')(x)
            
            # Ajustement de la taille si nécessaire
            skip_shape = skip_layer.shape[1:3]
            x_shape = x.shape[1:3]
            
            # Redimensionnement pour correspondre au skip layer
            if skip_shape != x_shape:
                x = layers.Resizing(skip_shape[0], skip_shape[1])(x)
            
            # Concatenation avec skip connection
            x = layers.Concatenate()([x, skip_layer])
            
            # Convolutions du decoder
            x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                             kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                             kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(MODEL_CONFIG['dropout_rate'] * 0.5)(x)
        
        # Upsampling final pour retrouver la taille d'origine
        x = layers.UpSampling2D(4, interpolation='bilinear')(x)
        
        # Ajustement final de taille
        if x.shape[1:3] != self.input_shape[:2]:
            x = layers.Resizing(self.input_shape[0], self.input_shape[1])(x)
        
        # Couche de classification finale
        outputs = layers.Conv2D(self.num_classes, 1, activation=MODEL_CONFIG['activation'], name='segmentation_output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        print(f"✅ {self.name} construit avec succès")
        return self.model

# Test de construction U-Net EfficientNet
print("\n🔥 CONSTRUCTION U-NET + EFFICIENTNET")
print("=" * 50)

unet_efficient = UNetEfficientNet(backbone='B0', freeze_backbone=True)
model_unet = unet_efficient.build_model()

info_unet = unet_efficient.get_model_info()
print(f"📊 U-Net EfficientNet-B0:")
print(f"   • Paramètres totaux: {info_unet['total_params']:,}")
print(f"   • Paramètres entraînables: {info_unet['trainable_params']:,}")
print(f"   • Couches: {info_unet['layers']}")

# %% [markdown]
# ## 🚀 Architecture 2: DeepLabV3+ + MobileNet
# 
# **Conception** : ASPP + Decoder léger avec backbone mobile optimisé

# %%
class DeepLabV3Plus(SegmentationModel):
    """
    DeepLabV3+ avec backbone MobileNetV2 pour efficacité embarquée,
    réécrit pour n'utiliser que des tailles statiques.
    """

    def __init__(self, input_shape=(512, 1024, 3), num_classes=8, output_stride=16):
        super().__init__(input_shape, num_classes, "DeepLabV3Plus_MobileNetV2")
        self.output_stride = output_stride
        # Déduire statiquement les résolutions intermédiaires
        h, w, _ = input_shape
        # MobileNetV2 reduce spatial by factor 32 by default
        self.low_res = (h // 4, w // 4)    # bloc_1_expand_relu → 1/4
        self.high_res = (h // 32, w // 32) # backbone.output → 1/32

    def atrous_spatial_pyramid_pooling(self, x):
        """
        ASPP statique : chaque branche est redimensionnée
        en fonction de self.high_res, connu à l'instanciation.
        """
        # Branch 1: 1x1 conv
        b1 = layers.Conv2D(256, 1, padding='same', activation='relu',
                           kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
        b1 = layers.BatchNormalization()(b1)

        # Branch 2-4: atrous conv
        branches = [b1]
        for rate in (6, 12, 18):
            b = layers.Conv2D(256, 3, padding='same', dilation_rate=rate,
                              activation='relu', kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
            b = layers.BatchNormalization()(b)
            branches.append(b)

        # Branch 5: global pooling
        gp = layers.GlobalAveragePooling2D()(x)              # (batch, C)
        gp = layers.Reshape((1, 1, x.shape[-1]))(gp)        # (batch,1,1,C)
        gp = layers.Conv2D(256, 1, activation='relu',
                           kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(gp)
        gp = layers.BatchNormalization()(gp)
        # Upsample statique vers high_res
        gp = layers.Resizing(self.high_res[0], self.high_res[1],
                             interpolation='bilinear')(gp)
        branches.append(gp)

        # Concat + conv final
        concat = layers.Concatenate()(branches)
        out = layers.Conv2D(256, 1, padding='same', activation='relu',
                            kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(concat)
        out = layers.BatchNormalization()(out)
        out = layers.Dropout(MODEL_CONFIG['dropout_rate'])(out)
        return out

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Backbone MobileNetV2
        backbone = MobileNetV2(weights='imagenet', include_top=False,
                              input_tensor=inputs, alpha=1.0)

        # Low-level (1/4) et high-level (1/32) features
        low_feat  = backbone.get_layer('block_3_expand_relu').output
        high_feat = backbone.output

        # ASPP + upsampling direct vers low_res
        aspp = self.atrous_spatial_pyramid_pooling(high_feat)
        x = layers.Resizing(self.low_res[0], self.low_res[1],
                            interpolation='bilinear')(aspp)

        # Réduction des low-level features puis concat
        low = layers.Conv2D(48, 1, padding='same', activation='relu',
                            kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(low_feat)
        low = layers.BatchNormalization()(low)
        concat = layers.Concatenate()([x, low])

        # Decoder final
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                          kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(concat)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(MODEL_CONFIG['dropout_rate'])(x)

        # Upsample final vers résolution d'entrée
        x = layers.Resizing(self.input_shape[0], self.input_shape[1],
                            interpolation='bilinear')(x)

        outputs = layers.Conv2D(self.num_classes, 1,
                                activation=MODEL_CONFIG['activation'],
                                name='segmentation_output')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        print(f"✅ {self.name} construit avec succès")
        return self.model

# Test de construction DeepLabV3+
print("\n🚀 CONSTRUCTION DEEPLABV3+ + MOBILENET")
print("=" * 50)

deeplab = DeepLabV3Plus()
model_deeplab = deeplab.build_model()

info_deeplab = deeplab.get_model_info()
print(f"📊 DeepLabV3+ MobileNetV2:")
print(f"   • Paramètres totaux: {info_deeplab['total_params']:,}")
print(f"   • Paramètres entraînables: {info_deeplab['trainable_params']:,}")
print(f"   • Couches: {info_deeplab['layers']}")

# %%
# Couche d'attention efficace pour Segformer
class EfficientSelfAttention(layers.Layer):
    def __init__(self, num_heads, sr_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.norm = layers.LayerNormalization()
        self.attn = None
        self.reduce = None

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        if self.sr_ratio > 1:
            self.reduce = layers.Conv2D(
                embed_dim,
                kernel_size=self.sr_ratio,
                strides=self.sr_ratio,
                padding='same'
            )
        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=embed_dim // self.num_heads,
            attention_axes=(1, 2),
            dropout=0.1
        )
        super().build(input_shape)

    def call(self, x):
        kv = x
        if self.sr_ratio > 1:
            kv = self.reduce(x)
            kv = self.norm(kv)
        return self.attn(query=x, key=kv, value=kv)

    def compute_output_shape(self, input_shape):
        return input_shape

# %% [markdown]
# ## 🌟 Architecture 3: Segformer-B0 (Vision Transformer)
# 
# **Conception** : Architecture Transformer adaptée à la segmentation, version légère

# %%
class SegformerB0(SegmentationModel):
    """
    Segformer-B0: Vision Transformer léger pour segmentation sémantique
    Implémentation adaptée sans reshapes manuels
    """
    def __init__(self, input_shape=(512, 1024, 3), num_classes=8, patch_size=4):
        super().__init__(input_shape, num_classes, "Segformer_B0")
        self.patch_size = patch_size
        self.embed_dims = [32, 64, 160, 256]
        self.num_heads = [1, 2, 5, 8]
        self.depths = [2, 2, 2, 2]

    def overlap_patch_embed(self, x, embed_dim, patch_size=7, stride=4):
        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding='same',
            kernel_regularizer=l2(MODEL_CONFIG['l2_reg'])
        )(x)
        x = layers.BatchNormalization()(x)
        return x

    def mix_ffn(self, x, embed_dim, expansion_factor=4):
        expanded_dim = embed_dim * expansion_factor
        x = layers.Dense(expanded_dim, activation='gelu')(x)
        x = layers.DepthwiseConv2D(3, padding='same')(x)
        x = layers.Dense(embed_dim)(x)
        return x

    def transformer_block(self, x, embed_dim, num_heads, sr_ratio=1):
        shortcut = x
        x = layers.LayerNormalization()(x)
        x = EfficientSelfAttention(num_heads=num_heads, sr_ratio=sr_ratio)(x)
        x = layers.Add()([shortcut, x])
        shortcut = x
        x = layers.LayerNormalization()(x)
        x = self.mix_ffn(x, embed_dim)
        x = layers.Add()([shortcut, x])
        return x

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = inputs
        encoder_features = []
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        sr_ratios = [8, 4, 2, 1]

        for i, (embed_dim, num_heads, depth) in enumerate(zip(self.embed_dims, self.num_heads, self.depths)):
            x = self.overlap_patch_embed(x, embed_dim, patch_sizes[i], strides[i])
            for _ in range(depth):
                x = self.transformer_block(x, embed_dim, num_heads, sr_ratios[i])
            encoder_features.append(x)

        # Decoder: upsample chaque feature vers 1/4 de la résolution d'entrée (128x256)
        decoder_features = []
        for i, features in enumerate(encoder_features):
            projected = layers.Conv2D(256, 1, padding='same')(features)
            upsampling_factor = 2 ** i  # i=0:1, i=1:2, i=2:4, i=3:8
            if upsampling_factor > 1:
                projected = layers.UpSampling2D(upsampling_factor, interpolation='bilinear')(projected)
            decoder_features.append(projected)

        fused = layers.Concatenate()(decoder_features)
        x = layers.Conv2D(256, 1, padding='same', activation='relu')(fused)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(MODEL_CONFIG['dropout_rate'])(x)
        x = layers.UpSampling2D(4, interpolation='bilinear')(x)
        outputs = layers.Conv2D(
            self.num_classes, 1,
            activation=MODEL_CONFIG['activation'],
            name='segmentation_output'
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        print(f"✅ {self.name} construit avec succès")
        return self.model

# Test de construction Segformer (version allégée pour éviter les erreurs de mémoire)
print("\n🌟 CONSTRUCTION SEGFORMER-B0 (SIMPLIFIÉ)")
print("=" * 50)

try:
    segformer = SegformerB0()
    model_segformer = segformer.build_model()
    
    info_segformer = segformer.get_model_info()
    print(f"📊 Segformer-B0:")
    print(f"   • Paramètres totaux: {info_segformer['total_params']:,}")
    print(f"   • Paramètres entraînables: {info_segformer['trainable_params']:,}")
    print(f"   • Couches: {info_segformer['layers']}")
    
except Exception as e:
    print(f"⚠️ Construction Segformer échouée: {e}")
    print("💡 Version simplifiée sera utilisée en cas d'erreur de mémoire")
    model_segformer = None

# %% [markdown]
# ## 🎯 Architecture 4: UNet Mini (Non Pré-entraîné) - MILESTONE 1
# 
# **Conception** : U-Net simple comme recommandé dans Milestone 1
# "un modèle simple non pré-entraîné comme « Unet mini »"

# %%
class UNetMini(SegmentationModel):
    """
    U-Net Mini: Architecture simple sans pré-entraînement
    Conforme aux recommandations Milestone 1
    """
    
    def __init__(self, input_shape=(512, 1024, 3), num_classes=8):
        super().__init__(input_shape, num_classes, "UNet_Mini")
        
    def conv_block(self, x, filters, kernel_size=3, activation='relu'):
        """Bloc de convolution standard avec BatchNorm et Dropout"""
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation,
                         kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
        if MODEL_CONFIG['batch_norm']:
            x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation,
                         kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
        if MODEL_CONFIG['batch_norm']:
            x = layers.BatchNormalization()(x)
        return x
    
    def encoder_block(self, x, filters):
        """Bloc encoder: Conv + MaxPool"""
        conv = self.conv_block(x, filters)
        pool = layers.MaxPooling2D(2)(conv)
        pool = layers.Dropout(MODEL_CONFIG['dropout_rate'])(pool)
        return conv, pool
    
    def decoder_block(self, x, skip_connection, filters):
        """Bloc decoder: UpSampling + Concat + Conv"""
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)
        
        # Ajustement de taille si nécessaire
        if x.shape[1:3] != skip_connection.shape[1:3]:
            target_height, target_width = skip_connection.shape[1:3]
            x = layers.Resizing(target_height, target_width)(x)
        
        x = layers.Concatenate()([x, skip_connection])
        x = self.conv_block(x, filters)
        x = layers.Dropout(MODEL_CONFIG['dropout_rate'] * 0.5)(x)
        return x
    
    def build_model(self):
        """Construit le modèle U-Net Mini sans pré-entraînement"""
        
        inputs = Input(shape=self.input_shape)
        
        # Encoder (Contracting Path)
        # Configuration optimisée pour Cityscapes 512x1024
        conv1, pool1 = self.encoder_block(inputs, 64)      # 512x1024 -> 256x512
        conv2, pool2 = self.encoder_block(pool1, 128)      # 256x512 -> 128x256
        conv3, pool3 = self.encoder_block(pool2, 256)      # 128x256 -> 64x128
        conv4, pool4 = self.encoder_block(pool3, 512)      # 64x128 -> 32x64
        
        # Bridge (Bottom)
        bridge = self.conv_block(pool4, 1024)              # 32x64
        bridge = layers.Dropout(MODEL_CONFIG['dropout_rate'])(bridge)
        
        # Decoder (Expanding Path)
        dec4 = self.decoder_block(bridge, conv4, 512)      # 32x64 -> 64x128
        dec3 = self.decoder_block(dec4, conv3, 256)        # 64x128 -> 128x256
        dec2 = self.decoder_block(dec3, conv2, 128)        # 128x256 -> 256x512
        dec1 = self.decoder_block(dec2, conv1, 64)         # 256x512 -> 512x1024
        
        # Couche de classification finale
        outputs = layers.Conv2D(self.num_classes, 1, 
                               activation=MODEL_CONFIG['activation'], 
                               name='segmentation_output')(dec1)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        print(f"✅ {self.name} construit avec succès")
        return self.model

# Test de construction UNet Mini
print("\n🎯 CONSTRUCTION UNET MINI (NON PRÉ-ENTRAÎNÉ) - MILESTONE 1")
print("=" * 60)

unet_mini = UNetMini()
model_unet_mini = unet_mini.build_model()

info_unet_mini = unet_mini.get_model_info()
print(f"📊 UNet Mini:")
print(f"   • Paramètres totaux: {info_unet_mini['total_params']:,}")
print(f"   • Paramètres entraînables: {info_unet_mini['trainable_params']:,}")
print(f"   • Couches: {info_unet_mini['layers']}")

# %% [markdown]
# ## 🎯 Architecture 5: VGG16 UNet (Pré-entraîné) - MILESTONE 1
# 
# **Conception** : U-Net avec encoder VGG16 comme recommandé dans Milestone 1
# "un modèle pré-entraîné comme « VGG16 Unet » (encoder = VGG16 pré-entraîné)"

# %%
class VGG16UNet(SegmentationModel):
    """
    VGG16 U-Net: U-Net avec encoder VGG16 pré-entraîné ImageNet
    Conforme aux recommandations Milestone 1
    """
    
    def __init__(self, input_shape=(512, 1024, 3), num_classes=8, freeze_backbone=False):
        super().__init__(input_shape, num_classes, "VGG16_UNet")
        self.freeze_backbone = freeze_backbone
        
    def conv_block(self, x, filters, kernel_size=3, activation='relu'):
        """Bloc de convolution pour le decoder"""
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation,
                         kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
        if MODEL_CONFIG['batch_norm']:
            x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation,
                         kernel_regularizer=l2(MODEL_CONFIG['l2_reg']))(x)
        if MODEL_CONFIG['batch_norm']:
            x = layers.BatchNormalization()(x)
        return x
    
    def decoder_block(self, x, skip_connection, filters):
        """Bloc decoder avec skip connections"""
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)
        
        # Ajustement de taille pour correspondre au skip connection
        if x.shape[1:3] != skip_connection.shape[1:3]:
            target_height, target_width = skip_connection.shape[1:3]
            x = layers.Resizing(target_height, target_width)(x)
        
        x = layers.Concatenate()([x, skip_connection])
        x = self.conv_block(x, filters)
        x = layers.Dropout(MODEL_CONFIG['dropout_rate'] * 0.5)(x)
        return x
    
    def build_model(self):
        """Construit le modèle VGG16 U-Net"""
        
        # Encoder VGG16 pré-entraîné
        vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Gel du backbone si spécifié
        if self.freeze_backbone:
            vgg16_base.trainable = False
            
        # Extraction des features pour skip connections
        # VGG16 architecture: block1, block2, block3, block4, block5
        skip_layers = [
            vgg16_base.get_layer('block1_conv2').output,  # 512x1024x64
            vgg16_base.get_layer('block2_conv2').output,  # 256x512x128  
            vgg16_base.get_layer('block3_conv3').output,  # 128x256x256
            vgg16_base.get_layer('block4_conv3').output,  # 64x128x512
        ]
        
        inputs = vgg16_base.input
        
        # Bottom (Bridge) - Sortie VGG16
        bridge = vgg16_base.output  # 16x32x512
        bridge = self.conv_block(bridge, 1024)
        bridge = layers.Dropout(MODEL_CONFIG['dropout_rate'])(bridge)
        
        # Decoder avec skip connections
        # Remontée progressive avec les features VGG16
        dec4 = self.decoder_block(bridge, skip_layers[3], 512)    # 16x32 -> 32x64
        dec3 = self.decoder_block(dec4, skip_layers[2], 256)      # 32x64 -> 64x128
        dec2 = self.decoder_block(dec3, skip_layers[1], 128)      # 64x128 -> 128x256
        dec1 = self.decoder_block(dec2, skip_layers[0], 64)       # 128x256 -> 256x512
        
        # Remontée finale vers résolution d'origine
        final = layers.UpSampling2D(2, interpolation='bilinear')(dec1)  # 256x512 -> 512x1024
        
        # Ajustement final de taille si nécessaire
        if final.shape[1:3] != self.input_shape[:2]:
            final = layers.Resizing(self.input_shape[0], self.input_shape[1])(final)
        
        # Couche de classification finale
        outputs = layers.Conv2D(self.num_classes, 1, 
                               activation=MODEL_CONFIG['activation'], 
                               name='segmentation_output')(final)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        print(f"✅ {self.name} construit avec succès")
        return self.model

# Test de construction VGG16 UNet
print("\n🎯 CONSTRUCTION VGG16 UNET (PRÉ-ENTRAÎNÉ) - MILESTONE 1")
print("=" * 60)

vgg16_unet = VGG16UNet(freeze_backbone=True)
model_vgg16_unet = vgg16_unet.build_model()

info_vgg16_unet = vgg16_unet.get_model_info()
print(f"📊 VGG16 UNet:")
print(f"   • Paramètres totaux: {info_vgg16_unet['total_params']:,}")
print(f"   • Paramètres entraînables: {info_vgg16_unet['trainable_params']:,}")
print(f"   • Couches: {info_vgg16_unet['layers']}")



print(f"\n✅ 5 MODÈLES IMPLÉMENTÉS")

# Nettoyage mémoire
tf.keras.backend.clear_session()



# %% [markdown]
# ## 🔧 Factory Pattern pour Création de Modèles

# %%
class ModelFactory:
    """
    Factory pour créer et gérer les différents modèles de segmentation
    """
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Crée un modèle selon le type spécifié
        
        Args:
            model_type: Type de modèle ('unet', 'deeplab', 'segformer')
            **kwargs: Arguments spécifiques au modèle
        """
        
        model_registry = {
            'unet': UNetEfficientNet,
            'deeplab': DeepLabV3Plus,
            'segformer': SegformerB0,
            'unet_mini': UNetMini,
            'vgg16_unet': VGG16UNet
        }
        
        if model_type.lower() not in model_registry:
            raise ValueError(f"Type de modèle '{model_type}' non reconnu. "
                           f"Types disponibles: {list(model_registry.keys())}")
        
        model_class = model_registry[model_type.lower()]
        return model_class(**kwargs)
    
    @staticmethod
    def get_model_comparison():
        """
        Compare les modèles implémentés
        """
        models_info = []
        
        # Test de chaque modèle
        test_configs = [
            {'type': 'unet', 'name': 'U-Net EfficientNet-B0', 'kwargs': {'backbone': 'B0'}},
            {'type': 'deeplab', 'name': 'DeepLabV3+ MobileNetV2', 'kwargs': {}},
            {'type': 'segformer', 'name': 'Segformer-B0', 'kwargs': {}},
            {'type': 'unet_mini', 'name': 'UNet Mini (Milestone 1)', 'kwargs': {}},
            {'type': 'vgg16_unet', 'name': 'VGG16 UNet (Milestone 1)', 'kwargs': {}}
        ]
        
        for config in test_configs:
            try:
                model = ModelFactory.create_model(config['type'], **config['kwargs'])
                built_model = model.build_model()
                info = model.get_model_info()
                
                models_info.append({
                    'Architecture': config['name'],
                    'Paramètres (M)': info['total_params'] / 1e6,
                    'Paramètres Entraînables (M)': info['trainable_params'] / 1e6,
                    'Couches': info['layers'],
                    'Statut': '✅ Opérationnel'
                })
                
                # Nettoyage mémoire
                del built_model, model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                models_info.append({
                    'Architecture': config['name'],
                    'Paramètres (M)': 'N/A',
                    'Paramètres Entraînables (M)': 'N/A',
                    'Couches': 'N/A',
                    'Statut': f'❌ Erreur: {str(e)[:50]}...'
                })
        
        return pd.DataFrame(models_info)

# Fonction de comparaison disponible mais pas exécutée automatiquement
def run_model_comparison():
    """
    Exécute la comparaison des modèles (à appeler manuellement)
    """
    print("\n📊 COMPARAISON DES MODÈLES IMPLÉMENTÉS")
    print("=" * 60)
    
    try:
        df_models = ModelFactory.get_model_comparison()
        print(df_models.to_string(index=False))
        return df_models
    except Exception as e:
        print(f"⚠️ Erreur lors de la comparaison: {e}")
        print("💡 Assurer-vous que toutes les classes sont définies avant d'appeler cette fonction")
        return None

print("\n✅ Modèles importés depuis 2.2")
print("💡 Utilisez run_model_comparison() pour comparer les modèles")

# %% [markdown]
# ## 🧪 Tests Unitaires des Architectures

# %%
def test_model_architecture(model_instance, test_name):
    """
    Test unitaire pour valider l'architecture d'un modèle
    """
    print(f"\n🧪 TEST: {test_name}")
    print("-" * 40)
    
    try:
        # Construction du modèle
        model = model_instance.build_model()
        
        # Tests des dimensions
        assert model.input_shape == (None,) + model_instance.input_shape, "Erreur input shape"
        assert model.output_shape == (None, model_instance.input_shape[0], 
                                     model_instance.input_shape[1], 
                                     model_instance.num_classes), "Erreur output shape"
        
        # Test de prédiction avec données aléatoires
        test_input = np.random.random((1,) + model_instance.input_shape)
        prediction = model.predict(test_input, verbose=0)
        
        # Vérifications de la prédiction
        assert prediction.shape == (1, model_instance.input_shape[0], 
                                   model_instance.input_shape[1], 
                                   model_instance.num_classes), "Erreur shape prédiction"
        
        # Vérification softmax (somme = 1 par pixel)
        pixel_sums = np.sum(prediction[0], axis=-1)
        assert np.allclose(pixel_sums, 1.0, atol=1e-5), "Erreur normalisation softmax"
        
        # Informations du modèle
        info = model_instance.get_model_info()
        
        print(f"✅ {test_name} - Tests réussis")
        print(f"   • Paramètres: {info['total_params']:,}")
        print(f"   • Shape output: {prediction.shape}")
        print(f"   • Min/Max prédiction: {prediction.min():.6f}/{prediction.max():.6f}")
        
        return True, info
        
    except Exception as e:
        print(f"❌ {test_name} - Erreur: {e}")
        return False, None

# Tests des architectures
print("\n🧪 TESTS UNITAIRES DES ARCHITECTURES")
print("=" * 50)

test_results = []

# Test U-Net EfficientNet
unet_test = UNetEfficientNet(backbone='B0')
success, info = test_model_architecture(unet_test, "U-Net + EfficientNet-B0")
test_results.append(('U-Net EfficientNet-B0', success, info))

# Test DeepLabV3+
deeplab_test = DeepLabV3Plus()
success, info = test_model_architecture(deeplab_test, "DeepLabV3+ + MobileNetV2")
test_results.append(('DeepLabV3+ MobileNetV2', success, info))

# Test Segformer (avec gestion d'erreur)
try:
    segformer_test = SegformerB0()
    success, info = test_model_architecture(segformer_test, "Segformer-B0")
    test_results.append(('Segformer-B0', success, info))
except Exception as e:
    print(f"⚠️ Segformer-B0 non testé: {e}")
    test_results.append(('Segformer-B0', False, None))

# Test UNet Mini
unet_mini_test = UNetMini()
success_mini, info_mini = test_model_architecture(unet_mini_test, "UNet Mini (Non pré-entraîné)")
test_results.append(('UNet Mini', success_mini, info_mini))

# Test VGG16 UNet
vgg16_test = VGG16UNet()
success_vgg16, info_vgg16 = test_model_architecture(vgg16_test, "VGG16 UNet (Pré-entraîné)")
test_results.append(('VGG16 UNet', success_vgg16, info_vgg16))

# Nettoyage mémoire
tf.keras.backend.clear_session()

# %% [markdown]
# ## 📋 Configuration Transfer Learning

# %%
def create_transfer_learning_strategy():
    """
    Définit les stratégies de transfer learning pour chaque architecture
    """
    
    strategies = {
        'unet_efficientnet': {
            'phase_1': {
                'description': 'Freeze backbone, train decoder only',
                'epochs': 5,
                'freeze_backbone': True,
                'learning_rate': 1e-3,
                'rationale': 'Adaptation rapide du decoder aux données Cityscapes'
            },
            'phase_2': {
                'description': 'Unfreeze backbone, fine-tune all',
                'epochs': 20,
                'freeze_backbone': False,
                'learning_rate': 1e-4,
                'rationale': 'Fine-tuning complet avec LR réduit'
            },
            'phase_3': {
                'description': 'Full training with decay',
                'epochs': 10,
                'freeze_backbone': False,
                'learning_rate': 1e-5,
                'rationale': 'Optimisation finale avec très petit LR'
            }
        },
        'deeplab_mobilenet': {
            'phase_1': {
                'description': 'Train ASPP and decoder only',
                'epochs': 8,
                'freeze_backbone': True,
                'learning_rate': 1e-3,
                'rationale': 'Focus sur l\'adaptation de l\'ASPP'
            },
            'phase_2': {
                'description': 'Fine-tune all layers',
                'epochs': 22,
                'freeze_backbone': False,
                'learning_rate': 5e-4,
                'rationale': 'Fine-tuning global MobileNet'
            },
            'phase_3': {
                'description': 'Final optimization',
                'epochs': 5,
                'freeze_backbone': False,
                'learning_rate': 1e-5,
                'rationale': 'Convergence finale'
            }
        },
        'segformer': {
            'phase_1': {
                'description': 'Train decoder head only',
                'epochs': 10,
                'freeze_backbone': True,
                'learning_rate': 1e-3,
                'rationale': 'Adaptation MLP head aux 8 classes'
            },
            'phase_2': {
                'description': 'Full model fine-tuning',
                'epochs': 25,
                'freeze_backbone': False,
                'learning_rate': 3e-4,
                'rationale': 'Fine-tuning Transformer complet'
            }
        },
        'unet_mini': {
            'phase_1': {
                'description': 'Train all from scratch',
                'epochs': 50,
                'freeze_backbone': False,
                'learning_rate': 1e-3,
                'rationale': 'Entraînement complet sans pré-entraînement'
            },
            'phase_2': {
                'description': 'Fine-tune with decay',
                'epochs': 20,
                'freeze_backbone': False,
                'learning_rate': 1e-4,
                'rationale': 'Convergence finale avec LR réduit'
            }
        },
        'vgg16_unet': {
            'phase_1': {
                'description': 'Freeze VGG16, train decoder only',
                'epochs': 8,
                'freeze_backbone': True,
                'learning_rate': 1e-3,
                'rationale': 'Adaptation decoder aux features VGG16'
            },
            'phase_2': {
                'description': 'Unfreeze VGG16, fine-tune all',
                'epochs': 25,
                'freeze_backbone': False,
                'learning_rate': 5e-4,
                'rationale': 'Fine-tuning complet VGG16 + decoder'
            },
            'phase_3': {
                'description': 'Final optimization',
                'epochs': 7,
                'freeze_backbone': False,
                'learning_rate': 1e-5,
                'rationale': 'Convergence finale optimisée'
            }
        }
    }
    
    # Sauvegarde des stratégies
    with open(OUTPUTS_DIR / "transfer_learning_strategies.json", 'w') as f:
        json.dump(strategies, f, indent=2)
    
    print("📋 STRATÉGIES DE TRANSFER LEARNING")
    print("=" * 50)
    
    for model_name, phases in strategies.items():
        print(f"\n🏗️ {model_name.upper()}:")
        total_epochs = sum(phase['epochs'] for phase in phases.values())
        print(f"   📊 Total époques: {total_epochs}")
        
        for phase_name, config in phases.items():
            print(f"   • {phase_name}: {config['description']}")
            print(f"     - Époques: {config['epochs']}, LR: {config['learning_rate']}")
    
    return strategies

# Création des stratégies
transfer_strategies = create_transfer_learning_strategy()

# %% [markdown]
# ## 💾 Sauvegarde et Export des Modèles

# %%
def save_model_architectures():
    """
    Sauvegarde les architectures et métadonnées des modèles
    """
    
    # Résumé des implémentations
    models_impl = [r for r in test_results if r[1]]
    models_fail = [r for r in test_results if not r[1]]
    implementation_summary = {
        'models_implemented': int(len(models_impl)),
        'total_models_planned': 5,
        'successful_models': [r[0] for r in models_impl],
        'failed_models': [r[0] for r in models_fail],
        'model_specifications': {
            'input_shape': MODEL_CONFIG['input_shape'],
            'num_classes': MODEL_CONFIG['num_classes'],
            'activation': MODEL_CONFIG['activation']
        }
    }
    
    # Paramètres détaillés par modèle
    model_details = {}
    for model_name, success, info in test_results:
        if success and info:
            model_details[model_name] = {
                'total_params': int(info['total_params']),
                'trainable_params': int(info['trainable_params']),
                'layers': int(info['layers']),
                'memory_estimate_mb': float(info['total_params'] * 4 / (1024 * 1024))
            }
    
    implementation_summary['model_details'] = model_details
    
    # Sauvegarde
    with open(OUTPUTS_DIR / "model_implementations.json", 'w') as f:
        json.dump(implementation_summary, f, indent=2)
    
    print("\n💾 SAUVEGARDE DES IMPLÉMENTATIONS")
    print("=" * 50)
    print(f"✅ {implementation_summary['models_implemented']}/{implementation_summary['total_models_planned']} modèles implémentés avec succès")
    print(f"📁 Métadonnées sauvegardées: {OUTPUTS_DIR / 'model_implementations.json'}")
    print(f"📁 Stratégies TL sauvegardées: {OUTPUTS_DIR / 'transfer_learning_strategies.json'}")
    
    return implementation_summary

# Sauvegarde finale
summary = save_model_architectures()

# Affichage du résumé final
print(f"\n🏆 RÉSUMÉ FINAL - IMPLÉMENTATION MODÈLES")
print("=" * 60)
print(f"✅ Modèles opérationnels: {', '.join(summary['successful_models'])}")
if summary['failed_models']:
    print(f"⚠️ Modèles échoués: {', '.join(summary['failed_models'])}")
print(f"🎯 Configuration: {summary['model_specifications']['input_shape']} → {summary['model_specifications']['num_classes']} classes")
print(f"🔧 Factory Pattern implémenté pour création modulaire")
print(f"📋 Stratégies Transfer Learning définies pour chaque architecture")


