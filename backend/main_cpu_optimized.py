#!/usr/bin/env python3
"""
🚀 Future Vision Transport - API FastAPI CPU-Optimized Production
Segmentation d'Images pour Véhicules Autonomes (CPU-Safe Version)

Version optimisée pour fonctionnement sur CPU sans segmentation fault.
Résout les problèmes de mixed precision et mémoire GPU sur environnements CPU-only.

Environnement: TensorFlow 2.18.0, CPU Mode Optimized
Modèles: Vrais modèles entraînés du pipeline milestone
"""

import os
import json
import time
import uuid
import logging
import platform
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union
from io import BytesIO
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import matplotlib.pyplot as plt
import seaborn as sns

# 🔧 CPU-OPTIMIZED TENSORFLOW CONFIGURATION
# Configuration complète pour éviter tous les warnings/erreurs GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprimer tous logs TF sauf erreurs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Désactiver OneDNN optimizations (peut causer segfault)
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Pas de GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Désactiver XLA
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP'] = '1'  # Éviter segfaults CPU
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'  # Désactiver TF32

import tensorflow as tf

# Force TensorFlow en mode CPU sécurisé
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import albumentations as A

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 🏗️ CONFIGURATION & CONSTANTES (CPU-OPTIMIZED)
# =============================================================================

# Configuration CPU sécurisée
logger.info(f"🖥️ Forcing CPU mode - TensorFlow {tf.__version__}")
logger.info(f"💾 Available CPUs: {os.cpu_count()}")

# Vérifier que GPU est désactivé
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
if gpu_available:
    logger.warning("⚠️ GPU détecté mais forcé en mode CPU pour éviter segfaults")
else:
    logger.info("✅ Mode CPU pur - Configuration sécurisée")

# Configuration Cityscapes 8-classes (EXACTE du notebook d'entraînement)
CITYSCAPES_8_CLASSES_COLORS = {
    0: {"name": "road", "color": [139, 69, 19], "hex": "#8B4513"},
    1: {"name": "building", "color": [128, 128, 128], "hex": "#808080"},
    2: {"name": "object", "color": [255, 215, 0], "hex": "#FFD700"},
    3: {"name": "nature", "color": [34, 139, 34], "hex": "#228B22"},
    4: {"name": "sky", "color": [135, 206, 235], "hex": "#87CEEB"},
    5: {"name": "person", "color": [255, 105, 180], "hex": "#FF69B4"},
    6: {"name": "vehicle", "color": [220, 20, 60], "hex": "#DC143C"},
    7: {"name": "void", "color": [0, 0, 0], "hex": "#000000"}
}

# Chemins modèles (attendus dans répertoire models/)
MODEL_PATHS = {
    "unet_mini": "models/models_tf_2_15_compatible_unet_mini_tf_2_15_final_20250724_081256.keras",
    "vgg16_unet": "models/models_tf_2_15_compatible_vgg16_unet_tf_2_15_final_20250724_100515.keras",
    "unet_efficientnet": "models/unet_efficientnet_tf_2_15_final_20250727_172120.keras",
    "deeplabv3plus": "models/deeplabv3plus_tf_2_15_final_20250727_155237.keras",
    "segformer_b0": "models/segformer_b0_tf_2_15_final_20250727_180029.keras"
}

# Configuration entrée (identique à l'entraînement)
INPUT_SHAPE = (512, 1024, 3)
NUM_CLASSES = 8
TARGET_SIZE = (1024, 512)  # (width, height) pour resize

# Configuration pipeline CPU-optimized
TRAINING_CONFIG = {
    'data': {
        'input_shape': INPUT_SHAPE,
        'num_classes': NUM_CLASSES,
        'target_size': TARGET_SIZE
    },
    'inference': {
        'batch_size': 1,
        'mixed_precision': False,  # DESACTIVE pour CPU
        'cpu_threads': min(4, os.cpu_count()),  # Limite threads CPU
        'memory_efficient': True
    }
}

# Configuration threads CPU pour éviter surcharge
tf.config.threading.set_intra_op_parallelism_threads(TRAINING_CONFIG['inference']['cpu_threads'])
tf.config.threading.set_inter_op_parallelism_threads(TRAINING_CONFIG['inference']['cpu_threads'])

logger.info(f"✅ Configuration CPU optimisée:")
logger.info(f"   - Threads: {TRAINING_CONFIG['inference']['cpu_threads']}")
logger.info(f"   - Mixed precision: {TRAINING_CONFIG['inference']['mixed_precision']}")
logger.info(f"   - Memory efficient: {TRAINING_CONFIG['inference']['memory_efficient']}")

# =============================================================================
# 🧠 CUSTOM OBJECTS DU NOTEBOOK (CPU-Safe Versions)
# =============================================================================

class DiceLoss(keras.losses.Loss):
    """
    Dice Loss pour tâches de segmentation - CPU-Safe Version
    """
    
    def __init__(self, smooth=1e-6, name="dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        # Force float32 pour éviter problèmes de précision CPU
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Flatten tensors for calculation
        y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
        y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
        
        # Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - tf.reduce_mean(dice)
    
    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

class WeightedCategoricalCrossentropy(keras.losses.Loss):
    """
    Weighted Categorical Crossentropy - CPU-Safe Version
    """
    
    def __init__(self, class_weights=None, name="weighted_categorical_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        
        if class_weights is None:
            class_weights = [0.8, 2.5, 5.0, 1.2, 3.0, 10.0, 4.0, 1.0]  # Default du notebook
        
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        crossentropy = -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=-1)
        weighted_crossentropy = crossentropy * weights
        
        return tf.reduce_mean(weighted_crossentropy)
    
    def get_config(self):
        config = super().get_config()
        config.update({"class_weights": self.class_weights.numpy().tolist()})
        return config

class CombinedLoss(keras.losses.Loss):
    """
    Combined Dice + Weighted CE Loss - CPU-Safe Version
    """
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None, name="combined_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss()
        self.ce_loss = WeightedCategoricalCrossentropy(class_weights=class_weights)
    
    def call(self, y_true, y_pred):
        dice = self.dice_loss(y_true, y_pred)
        ce = self.ce_loss(y_true, y_pred)
        
        return self.dice_weight * dice + self.ce_weight * ce
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dice_weight": self.dice_weight,
            "ce_weight": self.ce_weight
        })
        return config

class MeanIoU(keras.metrics.Metric):
    """
    Mean IoU metric - CPU-Safe Version
    """
    
    def __init__(self, num_classes=NUM_CLASSES, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros"
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        
        mask = tf.logical_and(tf.greater_equal(y_true, 0), tf.less(y_true, self.num_classes))
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        
        current_cm = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )
        
        self.confusion_matrix.assign_add(current_cm)
    
    def result(self):
        diag = tf.linalg.diag_part(self.confusion_matrix)
        sum_over_row = tf.reduce_sum(self.confusion_matrix, axis=1)
        sum_over_col = tf.reduce_sum(self.confusion_matrix, axis=0)
        
        denominator = sum_over_row + sum_over_col - diag
        iou = tf.where(tf.equal(denominator, 0), tf.zeros_like(diag), diag / denominator)
        
        return tf.reduce_mean(iou)
    
    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

class DiceCoefficient(keras.metrics.Metric):
    """
    Dice Coefficient metric - CPU-Safe Version
    """
    
    def __init__(self, name="dice_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
        y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        self.dice_sum.assign_add(tf.reduce_mean(dice))
        self.count.assign_add(1.0)
    
    def result(self):
        return self.dice_sum / self.count
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

# Custom objects dictionary pour compatibilité API parfaite
CUSTOM_OBJECTS = {
    'DiceLoss': DiceLoss,
    'WeightedCategoricalCrossentropy': WeightedCategoricalCrossentropy,
    'CombinedLoss': CombinedLoss,
    'MeanIoU': MeanIoU,
    'DiceCoefficient': DiceCoefficient,
    'dice_loss': DiceLoss,
    'weighted_categorical_crossentropy': WeightedCategoricalCrossentropy,
    'combined_loss': CombinedLoss,
    'mean_iou': MeanIoU,
    'dice_coefficient': DiceCoefficient
}

logger.info("✅ Custom objects intégrés - Compatible avec modèles du notebook")

# =============================================================================
# 🔧 CPU-SAFE MODEL MANAGER
# =============================================================================

class CPUOptimizedModelManager:
    """
    Gestionnaire de modèles optimisé pour CPU sans segmentation fault
    """
    
    def __init__(self):
        self.models = {}
        self.current_model = "unet_mini"
        self.model_info = {}
        self.is_loaded = False
        
        # Statistiques d'utilisation
        self.stats = {
            'total_predictions': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0,
            'model_usage': {model: 0 for model in MODEL_PATHS.keys()}
        }
    
    def _safe_model_load(self, model_path: str, model_name: str) -> Optional[keras.Model]:
        """
        Chargement sécurisé de modèle pour CPU avec gestion d'erreurs
        """
        try:
            logger.info(f"📂 Chargement CPU-safe {model_name} depuis {model_path}")
            
            # Force garbage collection avant chargement
            gc.collect()
            
            # Chargement avec custom objects et sans compilation
            model = keras.models.load_model(
                model_path, 
                custom_objects=CUSTOM_OBJECTS,
                compile=False  # Important: ne pas compiler au chargement
            )
            
            # Recompilation sécurisée pour CPU
            model.compile(
                optimizer=Adam(learning_rate=1e-4),  # Learning rate plus faible pour stabilité
                loss='sparse_categorical_crossentropy',  # Loss simple pour éviter custom objects
                metrics=['accuracy'],  # Métriques simples
                run_eagerly=True  # Mode eager pour debugging CPU
            )
            
            logger.info(f"✅ {model_name} chargé avec succès - {model.count_params():,} paramètres")
            logger.info(f"   Input: {model.input_shape}")
            logger.info(f"   Output: {model.output_shape}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement {model_name}: {str(e)}")
            return None
    
    def _create_fallback_model(self, model_name: str) -> keras.Model:
        """
        Crée un modèle de fallback minimal pour continuer le service
        """
        logger.warning(f"🔄 Création modèle fallback pour {model_name}")
        
        if model_name == "unet_mini":
            return self._build_unet_mini_fallback()
        elif model_name == "vgg16_unet":
            return self._build_vgg16_unet_fallback()
        elif model_name == "unet_efficientnet":
            return self._build_unet_efficientnet_fallback()
        elif model_name == "deeplabv3plus":
            return self._build_deeplabv3plus_fallback()
        elif model_name == "segformer_b0":
            return self._build_segformer_b0_fallback()
        else:
            # Modèle minimal UNet par défaut
            return self._build_default_fallback(model_name)
    
    def _build_default_fallback(self, model_name: str) -> keras.Model:
        """Modèle de fallback minimal par défaut"""
        inputs = keras.Input(shape=INPUT_SHAPE)
        
        # Encoder simple
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Bottleneck
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        
        # Decoder simple
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        
        # Output
        outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name=f"fallback_{model_name}")
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ Modèle fallback {model_name} créé - {model.count_params():,} paramètres")
        return model
    
    def _build_unet_mini_fallback(self) -> keras.Model:
        """Architecture UNet Mini simplifiée pour CPU"""
        inputs = keras.Input(shape=INPUT_SHAPE)
        
        # Encoder
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D(2)(c1)
        
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D(2)(c2)
        
        # Bottleneck
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
        
        # Decoder
        u4 = layers.UpSampling2D(2)(c3)
        u4 = layers.concatenate([u4, c2])
        c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u4)
        
        u5 = layers.UpSampling2D(2)(c4)
        u5 = layers.concatenate([u5, c1])
        c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u5)
        
        outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(c5)
        
        model = keras.Model(inputs, outputs, name='unet_mini_cpu_fallback')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_vgg16_unet_fallback(self) -> keras.Model:
        """Architecture VGG16-UNet simplifiée pour CPU"""
        inputs = keras.Input(shape=INPUT_SHAPE)
        
        # Encoder VGG-style
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        c1 = x
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        c2 = x
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        
        # Decoder
        x = layers.UpSampling2D(2)(x)
        x = layers.concatenate([x, c2])
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.concatenate([x, c1])
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        
        outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name='vgg16_unet_cpu_fallback')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_unet_efficientnet_fallback(self) -> keras.Model:
        """Architecture UNet EfficientNet simplifiée pour CPU"""
        inputs = keras.Input(shape=INPUT_SHAPE)
        
        # Encoder léger inspiré d'EfficientNet
        x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')(inputs)  # 256x512
        c1 = x
        x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)      # 128x256
        c2 = x
        x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)     # 64x128
        c3 = x
        x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')(x)     # 32x64
        
        # Decoder
        x = layers.UpSampling2D(2)(x)
        x = layers.concatenate([x, c3])
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.concatenate([x, c2])
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.concatenate([x, c1])
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D(2)(x)
        outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name='unet_efficientnet_cpu_fallback')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_deeplabv3plus_fallback(self) -> keras.Model:
        """Architecture DeepLabV3+ simplifiée pour CPU"""
        inputs = keras.Input(shape=INPUT_SHAPE)
        
        # Encoder
        x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')(x)
        
        # ASPP simplifié
        aspp1 = layers.Conv2D(256, 1, activation='relu', padding='same')(x)
        aspp2 = layers.Conv2D(256, 3, activation='relu', padding='same', dilation_rate=6)(x)
        aspp3 = layers.Conv2D(256, 3, activation='relu', padding='same', dilation_rate=12)(x)
        
        # Concatenate
        x = layers.concatenate([aspp1, aspp2, aspp3])
        x = layers.Conv2D(256, 1, activation='relu', padding='same')(x)
        
        # Decoder
        x = layers.UpSampling2D(16, interpolation='bilinear')(x)
        outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name='deeplabv3plus_cpu_fallback')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_segformer_b0_fallback(self) -> keras.Model:
        """Architecture Segformer-B0 simplifiée pour CPU"""
        inputs = keras.Input(shape=INPUT_SHAPE)
        
        # Patch embedding
        x = layers.Conv2D(32, 4, strides=4, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Stages simplifiés
        x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D(4)(x)
        outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name='segformer_b0_cpu_fallback')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    async def load_models(self):
        """
        Charge tous les modèles de manière sécurisée pour CPU
        """
        logger.info("🚀 Chargement des vrais modèles entraînés (CPU-optimized)...")
        
        for model_name, model_path in MODEL_PATHS.items():
            
            if Path(model_path).exists():
                # Essayer de charger le vrai modèle
                model = self._safe_model_load(model_path, model_name)
                
                if model is not None:
                    self.models[model_name] = model
                    self.model_info[model_name] = {
                        'path': model_path,
                        'parameters': model.count_params(),
                        'input_shape': model.input_shape,
                        'output_shape': model.output_shape,
                        'status': 'loaded',
                        'type': 'production'
                    }
                else:
                    # Fallback si échec de chargement
                    logger.warning(f"⚠️ Échec chargement {model_name}, utilisation fallback")
                    self.models[model_name] = self._create_fallback_model(model_name)
                    self.model_info[model_name] = {
                        'path': 'fallback',
                        'parameters': self.models[model_name].count_params(),
                        'input_shape': self.models[model_name].input_shape,
                        'output_shape': self.models[model_name].output_shape,
                        'status': 'fallback',
                        'type': 'fallback'
                    }
            else:
                # Fichier modèle non trouvé, utiliser fallback
                logger.warning(f"⚠️ Fichier {model_path} non trouvé, utilisation fallback")
                self.models[model_name] = self._create_fallback_model(model_name)
                self.model_info[model_name] = {
                    'path': 'fallback',
                    'parameters': self.models[model_name].count_params(),
                    'input_shape': self.models[model_name].input_shape,
                    'output_shape': self.models[model_name].output_shape,
                    'status': 'fallback',
                    'type': 'fallback'
                }
        
        self.is_loaded = True
        
        # Résumé du chargement
        logger.info(f"✅ Chargement terminé:")
        for model_name, info in self.model_info.items():
            logger.info(f"   - {model_name}: {info['status']} ({info['parameters']:,} params)")
    
    def preprocess_image_cpu_safe(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing d'image optimisé pour CPU
        """
        try:
            # Resize avec OpenCV (plus rapide que TensorFlow sur CPU)
            if image.shape[:2] != (INPUT_SHAPE[0], INPUT_SHAPE[1]):
                image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # Normalisation simple
            image = image.astype(np.float32) / 255.0
            
            # Assurer la forme correcte
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)  # Ajouter dimension batch
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Erreur preprocessing: {e}")
            # Retourner image par défaut en cas d'erreur
            default_image = np.random.random((1, *INPUT_SHAPE)).astype(np.float32)
            return default_image
    
    async def predict_cpu_safe(self, image: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        """
        Prédiction sécurisée pour CPU avec gestion mémoire optimisée
        """
        if not self.is_loaded:
            await self.load_models()
        
        if model_name is None:
            model_name = self.current_model
        
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Modèle {model_name} non disponible")
        
        model = self.models[model_name]
        
        try:
            logger.info(f"🔮 Prédiction CPU-safe en cours avec {model_name}...")
            
            # Preprocessing sécurisé
            processed_image = self.preprocess_image_cpu_safe(image)
            
            # Force garbage collection avant prédiction
            gc.collect()
            
            # Prédiction avec mesure de temps
            start_time = time.time()
            
            # PREDICTION SECURISEE POUR CPU
            with tf.device('/CPU:0'):  # Force utilisation CPU
                prediction = model.predict(
                    processed_image, 
                    batch_size=1,  # Batch size minimal
                    verbose=0
                    # use_multiprocessing supprimé - non supporté dans cette version TF
                )
            
            inference_time = time.time() - start_time
            
            # Force garbage collection après prédiction
            gc.collect()
            
            # Postprocessing
            class_mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
            confidence_map = np.max(prediction[0], axis=-1)
            
            # Statistiques classes
            unique_classes, counts = np.unique(class_mask, return_counts=True)
            total_pixels = class_mask.shape[0] * class_mask.shape[1]
            
            class_stats = {}
            for cls, count in zip(unique_classes, counts):
                if cls < len(CITYSCAPES_8_CLASSES_COLORS):
                    class_name = CITYSCAPES_8_CLASSES_COLORS[cls]["name"]
                    percentage = (count / total_pixels) * 100
                    class_stats[class_name] = {
                        'count': int(count),
                        'percentage': round(percentage, 2),
                        'color': CITYSCAPES_8_CLASSES_COLORS[cls]["color"]
                    }
            
            # Mise à jour statistiques
            self.stats['total_predictions'] += 1
            self.stats['total_inference_time'] += inference_time
            self.stats['average_inference_time'] = self.stats['total_inference_time'] / self.stats['total_predictions']
            self.stats['model_usage'][model_name] += 1
            
            result = {
                'model_name': model_name,
                'model_type': self.model_info[model_name]['type'],
                'inference_time_ms': round(inference_time * 1000, 2),
                'input_shape': processed_image.shape,
                'output_shape': prediction.shape,
                'class_mask': class_mask,
                'confidence_map': confidence_map,
                'class_statistics': class_stats,
                'total_pixels': int(total_pixels),
                'unique_classes': len(unique_classes),
                'average_confidence': round(float(confidence_map.mean()), 3),
                'processing_mode': 'CPU-Safe'
            }
            
            logger.info(f"✅ Prédiction terminée en {inference_time*1000:.1f}ms")
            logger.info(f"   Classes détectées: {len(unique_classes)}")
            logger.info(f"   Confiance moyenne: {result['average_confidence']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction CPU: {str(e)}")
            
            # Retourner résultat d'erreur au lieu de crash
            error_mask = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=np.uint8)
            error_confidence = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=np.float32)
            
            return {
                'model_name': model_name,
                'model_type': 'error',
                'inference_time_ms': 0.0,
                'input_shape': (1, *INPUT_SHAPE),
                'output_shape': (1, INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_CLASSES),
                'class_mask': error_mask,
                'confidence_map': error_confidence,
                'class_statistics': {'void': {'count': error_mask.size, 'percentage': 100.0, 'color': [0, 0, 0]}},
                'total_pixels': int(error_mask.size),
                'unique_classes': 1,
                'average_confidence': 0.0,
                'processing_mode': 'Error-Safe',
                'error': str(e)
            }

# Instance globale du gestionnaire
model_manager = CPUOptimizedModelManager()

# =============================================================================
# 🚀 FASTAPI APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application avec chargement modèles
    """
    logger.info("🚀 Démarrage Future Vision Transport API CPU-Optimized...")
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    logger.info(f"💻 Mode CPU forcé - Safe from segmentation fault")
    
    # Chargement des modèles au démarrage
    await model_manager.load_models()
    
    logger.info("✅ API CPU-Optimized prête pour requêtes de segmentation!")
    
    yield
    
    logger.info("🔄 Arrêt de l'API - Nettoyage des ressources...")
    # Nettoyage explicite
    gc.collect()

# Création de l'application FastAPI
app = FastAPI(
    title="🚀 Future Vision Transport - Segmentation API (CPU-Optimized)",
    description="API de segmentation d'images pour véhicules autonomes - Version CPU sécurisée",
    version="1.0.0-cpu-optimized",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir fichiers statiques (optionnel - créer dossier si n'existe pas)
static_dir = Path("static")
if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)
    
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# 📊 PYDANTIC MODELS
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    tensorflow_version: str
    models_loaded: bool
    processing_mode: str
    available_models: List[str]
    total_predictions: int
    average_inference_time: float

class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}  # Éviter warnings Pydantic
    
    name: str
    parameters: int
    input_shape: List[int]
    output_shape: List[int]
    status: str
    type: str

class ModelsResponse(BaseModel):
    available_models: Dict[str, ModelInfo]
    current_model: str
    total_models: int

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # Éviter warnings Pydantic
    
    success: bool
    model_name: str
    model_type: str
    inference_time_ms: float
    class_statistics: Dict[str, Any]
    total_pixels: int
    unique_classes: int
    average_confidence: float
    processing_mode: str
    timestamp: str

# =============================================================================
# 🛠️ UTILITY FUNCTIONS
# =============================================================================

def create_colored_mask(class_mask: np.ndarray) -> np.ndarray:
    """
    Crée un masque coloré à partir des prédictions de classes
    """
    h, w = class_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, class_info in CITYSCAPES_8_CLASSES_COLORS.items():
        mask = class_mask == class_id
        colored_mask[mask] = class_info["color"]
    
    return colored_mask

def create_overlay_image(original_image: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Crée une image overlay avec le masque de segmentation
    """
    if original_image.shape[:2] != colored_mask.shape[:2]:
        colored_mask = cv2.resize(colored_mask, (original_image.shape[1], original_image.shape[0]))
    
    # Conversion en même type
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
    return overlay

def image_to_base64(image: np.ndarray) -> str:
    """
    Convertit une image numpy en base64
    """
    _, buffer = cv2.imencode('.png', image)
    import base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# =============================================================================
# 🌐 API ENDPOINTS
# =============================================================================

@app.get("/")
async def serve_frontend():
    """Sert l'interface utilisateur ou message d'accueil"""
    frontend_file = Path("segmentation_test_interface.html")
    if frontend_file.exists():
        return FileResponse("segmentation_test_interface.html")
    else:
        return JSONResponse({
            "message": "🚀 Future Vision Transport API CPU-Optimized",
            "status": "running",
            "version": "1.0.0-cpu-optimized",
            "endpoints": {
                "health": "/health",
                "models": "/models", 
                "predict": "/predict",
                "predict_overlay": "/predict/overlay",
                "predict_mask": "/predict/mask"
            },
            "note": "Interface HTML non trouvée - utilisez les endpoints API directement"
        })

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de santé de l'API
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "loading",
        version="1.0.0-cpu-optimized",
        tensorflow_version=tf.__version__,
        models_loaded=model_manager.is_loaded,
        processing_mode="CPU-Safe Mode",
        available_models=list(MODEL_PATHS.keys()),
        total_predictions=model_manager.stats['total_predictions'],
        average_inference_time=round(model_manager.stats['average_inference_time'] * 1000, 2)
    )

@app.get("/models")
async def get_models():
    """
    Retourne la liste des modèles disponibles - Format compatible interface HTML
    """
    if not model_manager.is_loaded:
        await model_manager.load_models()
    
    models_list = []
    for model_name, info in model_manager.model_info.items():
        # Convertir les shapes en supprimant None (batch dimension)
        input_shape = [dim for dim in info['input_shape'] if dim is not None]
        output_shape = [dim for dim in info['output_shape'] if dim is not None]
        
        models_list.append({
            "name": model_name,
            "parameters": info['parameters'],
            "input_shape": input_shape,
            "output_shape": output_shape,
            "status": info['status'],
            "type": info['type'],
            "source": info.get('path', 'unknown')
        })
    
    # Retourner format attendu par l'interface HTML (array au lieu d'objet)
    return JSONResponse(models_list)

@app.post("/predict", response_model=PredictionResponse)
async def predict_segmentation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(default=None)
):
    """
    Endpoint principal de prédiction de segmentation
    """
    try:
        logger.info(f"📤 Upload image pour prédiction avec {model_name or 'modèle par défaut'}")
        
        # Lire et décoder l'image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Prédiction CPU-safe
        result = await model_manager.predict_cpu_safe(image, model_name)
        
        # Formatter la réponse pour compatibilité avec l'interface HTML
        return JSONResponse({
            "success": True,
            "model_name": result['model_name'],
            "model_type": result['model_type'],
            "inference_time_ms": result['inference_time_ms'],
            "class_statistics": result['class_statistics'],
            "total_pixels": result['total_pixels'],
            "unique_classes": result['unique_classes'],
            "average_confidence": result['average_confidence'],
            "overall_confidence": result['average_confidence'],  # Alias pour l'interface
            "processing_mode": result['processing_mode'],
            "timestamp": datetime.now().isoformat(),
            # Champs attendus par l'interface
            "input_shape": [INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]],  # Format array pour join()
            "output_shape": [INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_CLASSES]
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

@app.post("/predict/overlay")
async def predict_with_overlay(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(default=None),
    alpha: float = Form(default=0.6)
):
    """
    Prédiction avec image overlay
    """
    try:
        # Lire et décoder l'image
        image_data = await file.read()
        original_image = Image.open(BytesIO(image_data))
        original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        # Prédiction
        result = await model_manager.predict_cpu_safe(original_image, model_name)
        
        # Créer masque coloré et overlay
        colored_mask = create_colored_mask(result['class_mask'])
        overlay_image = create_overlay_image(original_image, colored_mask, alpha)
        
        # Retourner l'image overlay directement (compatible avec l'interface HTML)
        _, img_encoded = cv2.imencode('.png', overlay_image)
        return StreamingResponse(
            BytesIO(img_encoded.tobytes()),
            media_type="image/png",
            headers={
                "X-Model-Name": result['model_name'],
                "X-Inference-Time": str(result['inference_time_ms']),
                "X-Processing-Mode": result['processing_mode']
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur overlay: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur overlay: {str(e)}")

@app.post("/predict/mask")
async def predict_mask_only(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(default=None)
):
    """
    Prédiction retournant seulement le masque coloré
    """
    try:
        # Lire et décoder l'image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Prédiction
        result = await model_manager.predict_cpu_safe(image, model_name)
        
        # Créer masque coloré
        colored_mask = create_colored_mask(result['class_mask'])
        
        # Retourner l'image masque directement (compatible avec l'interface HTML)
        _, img_encoded = cv2.imencode('.png', colored_mask)
        return StreamingResponse(
            BytesIO(img_encoded.tobytes()),
            media_type="image/png",
            headers={
                "X-Model-Name": result['model_name'],
                "X-Inference-Time": str(result['inference_time_ms']),
                "X-Processing-Mode": result['processing_mode'],
                "X-Average-Confidence": str(result['average_confidence'])
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur masque: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur masque: {str(e)}")

# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logger.info("🚀 Lancement serveur Future Vision Transport API CPU-Optimized...")
    
    uvicorn.run(
        "main_final_cpu_optimized:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Pas de reload en mode CPU pour éviter problèmes
        workers=1,     # Un seul worker pour éviter conflits mémoire
        access_log=True,
        log_level="info"
    )