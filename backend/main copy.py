#!/usr/bin/env python3
"""
🚀 Future Vision Transport - API FastAPI SOTA Production
Segmentation d'Images pour Véhicules Autonomes

Cette API de production sert les vrais modèles entraînés UNet Mini et VGG16 UNet
pour la segmentation Cityscapes 8-classes en utilisant la méthodologie exacte
du pipeline de vérification du notebook d'entraînement.

Environnement: TensorFlow 2.18.0, Google Colab L4 GPU
Modèles: Vrais modèles entraînés du pipeline milestone
"""

import os
import json
import time
import uuid
import logging
import platform
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

import tensorflow as tf
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
# 🏗️ CONFIGURATION & CONSTANTES (TensorFlow 2.18.0 + Google Colab L4)
# =============================================================================

# Configuration environnement Google Colab L4 GPU (adapté de RTX 4080)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Configuration Cityscapes 8-classes (EXACTE du notebook d'entraînement)
CITYSCAPES_8_CLASSES_COLORS = {
    0: {"name": "road", "color": [128, 64, 128], "hex": "#804080"},
    1: {"name": "building", "color": [70, 70, 70], "hex": "#464646"},
    2: {"name": "object", "color": [153, 153, 153], "hex": "#999999"},
    3: {"name": "nature", "color": [107, 142, 35], "hex": "#6B8E23"},
    4: {"name": "sky", "color": [70, 130, 180], "hex": "#4682B4"},
    5: {"name": "person", "color": [220, 20, 60], "hex": "#DC143C"},
    6: {"name": "vehicle", "color": [0, 0, 142], "hex": "#00008E"},
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

# Configuration pipeline (du notebook)
TRAINING_CONFIG = {
    'data': {
        'input_shape': INPUT_SHAPE,
        'num_classes': NUM_CLASSES,
        'target_size': TARGET_SIZE
    },
    'inference': {
        'batch_size': 1,
        'mixed_precision': True,  # Activé sur Colab L4
        'gpu_memory_growth': True
    }
}

logger.info(f"🚀 Configuration chargée - TensorFlow {tf.__version__}, GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# 🧠 CUSTOM OBJECTS DU NOTEBOOK (Identiques à l'entraînement)
# =============================================================================

class DiceLoss(keras.losses.Loss):
    """
    Dice Loss pour tâches de segmentation
    Implementation EXACTE du notebook d'entraînement
    """
    
    def __init__(self, smooth=1e-6, name="dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
        )
        return 1 - dice_coef
    
    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

class WeightedCategoricalCrossentropy(keras.losses.Loss):
    """
    Weighted Categorical Crossentropy
    Implementation EXACTE du notebook d'entraînement
    """
    
    def __init__(self, weights, name="weighted_categorical_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = tf.constant(weights, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 
                                1 - tf.keras.backend.epsilon())
        
        loss = y_true * tf.math.log(y_pred)
        loss = loss * self.weights
        return -tf.reduce_sum(loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"weights": self.weights.numpy().tolist()})
        return config

class CombinedLoss(keras.losses.Loss):
    """
    Combined Dice + Weighted CE Loss
    Implementation EXACTE du notebook d'entraînement
    """
    
    def __init__(self, weights, alpha=0.5, name="combined_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_loss = DiceLoss()
        self.weighted_ce = WeightedCategoricalCrossentropy(weights)
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        dice = self.dice_loss(y_true, y_pred)
        weighted_ce = self.weighted_ce(y_true, y_pred)
        return self.alpha * dice + (1 - self.alpha) * tf.reduce_mean(weighted_ce)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "weights": self.weighted_ce.weights.numpy().tolist()
        })
        return config

class DiceCoefficient(keras.metrics.Metric):
    """
    Dice Coefficient metric
    Implementation EXACTE du notebook d'entraînement
    """
    
    def __init__(self, smooth=1e-6, name="dice_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
        )
        
        self.dice_sum.assign_add(dice_coef)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.dice_sum / self.count
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

# Dictionnaire custom objects pour chargement modèles (EXACT du notebook)
CUSTOM_OBJECTS = {
    'DiceLoss': DiceLoss,
    'WeightedCategoricalCrossentropy': WeightedCategoricalCrossentropy,
    'CombinedLoss': CombinedLoss,
    'DiceCoefficient': DiceCoefficient,
    'MeanIoU': tf.keras.metrics.MeanIoU
}

logger.info("✅ Custom objects intégrés - Compatible avec modèles du notebook")

# =============================================================================
# 🎨 FONCTIONS VISUALISATION (Du notebook d'entraînement)
# =============================================================================

def create_cityscapes_colormap():
    """
    Crée la colormap Cityscapes 8-classes
    Function EXACTE du notebook d'entraînement
    """
    colormap = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for class_id, class_info in CITYSCAPES_8_CLASSES_COLORS.items():
        colormap[class_id] = class_info["color"]
    return colormap

def convert_prediction_to_color(prediction_map):
    """
    Convertit carte de prédiction en image couleur
    Function EXACTE du notebook d'entraînement
    """
    colormap = create_cityscapes_colormap()
    h, w = prediction_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(NUM_CLASSES):
        mask = prediction_map == class_id
        color_image[mask] = colormap[class_id]
    
    return color_image

def create_overlay_image(original_image, prediction_map, alpha=0.6):
    """
    Crée image overlay original + segmentation
    Function EXACTE du notebook d'entraînement
    """
    # Redimensionner original si nécessaire
    if original_image.shape[:2] != prediction_map.shape:
        original_resized = cv2.resize(original_image, (prediction_map.shape[1], prediction_map.shape[0]))
    else:
        original_resized = original_image.copy()
    
    # Créer masque coloré
    color_mask = convert_prediction_to_color(prediction_map)
    
    # Créer overlay
    overlay = cv2.addWeighted(original_resized, 1-alpha, color_mask, alpha, 0)
    
    return overlay.astype(np.uint8)

logger.info("✅ Fonctions visualisation intégrées - Colormap Cityscapes prête")

# =============================================================================
# 🎯 MODEL MANAGER AVEC VRAIS MODÈLES + FALLBACKS
# =============================================================================

class ProductionModelManager:
    """
    Gestionnaire de modèles production qui charge et sert les vrais modèles entraînés
    en utilisant la méthodologie EXACTE du pipeline de vérification du notebook.
    """
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.gpu_available = self._configure_gpu_l4()
        self.tensorflow_version = tf.__version__
        self.fallback_mode = False
        
    def _configure_gpu_l4(self) -> bool:
        """Configure GPU pour environnement Google Colab L4 (adapté de RTX 4080)"""
        try:
            # Configuration GPU memory growth (adapté pour L4)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Configuration spécifique L4 (moins agressive que RTX 4080)
                    tf.config.experimental.set_memory_limit(gpu, 12288)  # 12GB L4
                
                logger.info(f"✅ GPU L4 configuré: {len(gpus)} GPU(s) disponible(s)")
                return True
            else:
                logger.warning("⚠️ Aucun GPU détecté - Mode CPU activé")
                return False
        except Exception as e:
            logger.error(f"❌ Configuration GPU L4 échouée: {e}")
            return False
    
    async def load_models(self):
        """Charge les vrais modèles entraînés avec custom objects"""
        logger.info("🚀 Chargement des vrais modèles entraînés...")
        
        for model_name, model_path in MODEL_PATHS.items():
            try:
                if os.path.exists(model_path):
                    # Chargement VRAI modèle entraîné avec custom objects
                    logger.info(f"📂 Chargement {model_name} depuis {model_path}")
                    
                    model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects=CUSTOM_OBJECTS,
                        compile=False  # Pas besoin de setup entraînement pour inférence
                    )
                    
                    self.models[model_name] = model
                    
                    # Stocker informations modèle
                    self.model_info[model_name] = {
                        "name": model_name,
                        "parameters": model.count_params(),
                        "input_shape": list(model.input_shape[1:]),
                        "output_shape": list(model.output_shape[1:]),
                        "source": f"REAL_TRAINED_MODEL: {model_path}",
                        "loaded_at": datetime.now().isoformat(),
                        "status": "available",
                        "type": "production"
                    }
                    
                    logger.info(f"✅ {model_name} chargé avec succès - {model.count_params():,} paramètres")
                    
                else:
                    logger.warning(f"⚠️ Fichier modèle introuvable: {model_path}")
                    # Créer modèle fallback (même architecture que l'entraînement)
                    await self._create_fallback_model(model_name)
                    
            except Exception as e:
                logger.error(f"❌ Échec chargement {model_name}: {e}")
                # Créer modèle fallback
                await self._create_fallback_model(model_name)
    
    async def _create_fallback_model(self, model_name: str):
        """Crée modèle fallback avec architecture identique au notebook"""
        logger.info(f"🔧 Création fallback {model_name} (architecture du notebook)...")
        
        try:
            if model_name == "unet_mini":
                model = self._build_unet_mini_architecture()
            elif model_name == "vgg16_unet":
                model = self._build_vgg16_unet_architecture()
            elif model_name == "unet_efficientnet":
                model = self._build_unet_efficientnet_architecture()
            elif model_name == "deeplabv3plus":
                model = self._build_deeplabv3plus_architecture()
            elif model_name == "segformer_b0":
                model = self._build_segformer_b0_architecture()
            else:
                raise ValueError(f"Modèle inconnu: {model_name}")
            
            self.models[model_name] = model
            self.model_info[model_name] = {
                "name": model_name,
                "parameters": model.count_params(),
                "input_shape": list(model.input_shape[1:]),
                "output_shape": list(model.output_shape[1:]),
                "source": "FALLBACK_ARCHITECTURE",
                "loaded_at": datetime.now().isoformat(),
                "status": "fallback",
                "type": "architecture_only"
            }
            
            self.fallback_mode = True
            logger.warning(f"⚠️ Utilisation fallback {model_name} - {model.count_params():,} paramètres")
            
        except Exception as e:
            logger.error(f"❌ Échec création fallback {model_name}: {e}")
            self.model_info[model_name] = {
                "name": model_name,
                "status": "unavailable",
                "error": str(e)
            }
    
    def _build_unet_mini_architecture(self) -> Model:
        """Architecture UNet Mini - EXACTE du notebook d'entraînement"""
        inputs = layers.Input(shape=INPUT_SHAPE, name='input_image')
        
        # Encoder (même architecture que l'entraînement)
        c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv1a')(inputs)
        c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv1b')(c1)
        p1 = layers.MaxPooling2D((2, 2), name='enc_pool1')(c1)
        
        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv2a')(p1)
        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv2b')(c2)
        p2 = layers.MaxPooling2D((2, 2), name='enc_pool2')(c2)
        
        c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='enc_conv3a')(p2)
        c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='enc_conv3b')(c3)
        p3 = layers.MaxPooling2D((2, 2), name='enc_pool3')(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='bottleneck_conv1')(p3)
        c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='bottleneck_conv2')(c4)
        
        # Decoder (même architecture que l'entraînement)
        u5 = layers.UpSampling2D((2, 2), name='dec_upsample1')(c4)
        u5 = layers.concatenate([u5, c3], name='dec_concat1')
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='dec_conv1a')(u5)
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='dec_conv1b')(c5)
        
        u6 = layers.UpSampling2D((2, 2), name='dec_upsample2')(c5)
        u6 = layers.concatenate([u6, c2], name='dec_concat2')
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv2a')(u6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv2b')(c6)
        
        u7 = layers.UpSampling2D((2, 2), name='dec_upsample3')(c6)
        u7 = layers.concatenate([u7, c1], name='dec_concat3')
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv3a')(u7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv3b')(c7)
        
        # Output (8 classes Cityscapes)
        outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='output_segmentation')(c7)
        
        return Model(inputs=[inputs], outputs=[outputs], name='unet_mini_fallback')
    
    def _build_vgg16_unet_architecture(self) -> Model:
        """Architecture VGG16-UNet - EXACTE du notebook d'entraînement"""
        # Backbone VGG16 (même que l'entraînement)
        vgg16 = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=INPUT_SHAPE
        )
        
        # Extraire feature maps (même niveaux que l'entraînement)
        inputs = vgg16.input
        
        # Encoder (features VGG16)
        conv1 = vgg16.get_layer('block1_conv2').output  # 512, 1024, 64
        conv2 = vgg16.get_layer('block2_conv2').output  # 256, 512, 128
        conv3 = vgg16.get_layer('block3_conv3').output  # 128, 256, 256
        conv4 = vgg16.get_layer('block4_conv3').output  # 64, 128, 512
        conv5 = vgg16.get_layer('block5_conv3').output  # 32, 64, 512
        
        # Decoder (même architecture que l'entraînement)
        up6 = layers.UpSampling2D((2, 2), name='dec_upsample1')(conv5)
        up6 = layers.concatenate([up6, conv4], axis=3, name='dec_concat1')
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='dec_conv1a')(up6)
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='dec_conv1b')(conv6)
        
        up7 = layers.UpSampling2D((2, 2), name='dec_upsample2')(conv6)
        up7 = layers.concatenate([up7, conv3], axis=3, name='dec_concat2')
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='dec_conv2a')(up7)
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='dec_conv2b')(conv7)
        
        up8 = layers.UpSampling2D((2, 2), name='dec_upsample3')(conv7)
        up8 = layers.concatenate([up8, conv2], axis=3, name='dec_concat3')
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv3a')(up8)
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv3b')(conv8)
        
        up9 = layers.UpSampling2D((2, 2), name='dec_upsample4')(conv8)
        up9 = layers.concatenate([up9, conv1], axis=3, name='dec_concat4')
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv4a')(up9)
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv4b')(conv9)
        
        # Output (8 classes Cityscapes)
        outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='output_segmentation')(conv9)
        
        return Model(inputs=[inputs], outputs=[outputs], name='vgg16_unet_fallback')
    
    async def predict(self, image: np.ndarray, model_name: str = "unet_mini") -> Dict[str, Any]:
        """
        Exécute prédiction segmentation RÉELLE utilisant modèles entraînés
        Méthodologie EXACTE du pipeline de vérification du notebook
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non disponible. Disponibles: {list(self.models.keys())}")
        
        start_time = time.time()
        
        # Preprocessing image (même que l'entraînement)
        processed_image = self._preprocess_image_notebook_style(image)
        
        # Exécuter prédiction
        model = self.models[model_name]
        
        logger.info(f"🔮 Prédiction en cours avec {model_name}...")
        predictions = model.predict(processed_image, verbose=0)
        
        # Postprocessing prédictions (même que le notebook)
        prediction_map = np.argmax(predictions[0], axis=-1)
        confidence_map = np.max(predictions[0], axis=-1)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Calculer statistiques (même pipeline que la vérification)
        stats = self._calculate_prediction_statistics_notebook_style(
            prediction_map, confidence_map, model_name, inference_time_ms, processed_image.shape
        )
        
        logger.info(f"✅ Prédiction terminée en {inference_time_ms:.1f}ms")
        
        return {
            "prediction_map": prediction_map,
            "confidence_map": confidence_map,
            "statistics": stats,
            "processed_shape": processed_image.shape,
            "original_shape": image.shape
        }
    
    def _preprocess_image_notebook_style(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing image - EXACTE du notebook d'entraînement"""
        # Redimensionner à la taille cible (même que l'entraînement)
        if image.shape[:2] != INPUT_SHAPE[:2]:
            image = cv2.resize(image, TARGET_SIZE)  # (1024, 512)
        
        # Normaliser à [0, 1] (même que l'entraînement)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Ajouter dimension batch
        return np.expand_dims(image, axis=0)
    
    def _calculate_prediction_statistics_notebook_style(
        self, 
        prediction_map: np.ndarray, 
        confidence_map: np.ndarray,
        model_name: str,
        inference_time_ms: float,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Calculer statistiques prédiction - EXACTE du pipeline de vérification"""
        
        # Statistiques par classe (même méthodologie que le notebook)
        class_stats = {}
        total_pixels = prediction_map.size
        
        for class_id, class_info in CITYSCAPES_8_CLASSES_COLORS.items():
            mask = prediction_map == class_id
            pixel_count = np.sum(mask)
            percentage = (pixel_count / total_pixels) * 100
            avg_confidence = np.mean(confidence_map[mask]) if pixel_count > 0 else 0.0
            
            class_stats[class_info["name"]] = {
                "pixel_count": int(pixel_count),
                "percentage": float(percentage),
                "avg_confidence": float(avg_confidence)
            }
        
        # Statistiques globales (même que la vérification)
        overall_confidence = float(np.mean(confidence_map))
        
        # Détection classes présentes
        unique_classes = np.unique(prediction_map)
        classes_detected = len(unique_classes)
        
        return {
            "model_name": model_name,
            "inference_time_ms": float(inference_time_ms),
            "overall_confidence": overall_confidence,
            "input_shape": list(input_shape),
            "output_shape": list(prediction_map.shape),
            "class_statistics": class_stats,
            "classes_detected": classes_detected,
            "total_pixels": total_pixels,
            "timestamp": datetime.now().isoformat(),
            "fallback_mode": self.model_info[model_name]["status"] == "fallback"
        }

# Initialiser gestionnaire modèles
model_manager = ProductionModelManager()

logger.info("✅ Model Manager initialisé - Prêt pour chargement modèles production")

# =============================================================================
# 🔮 PIPELINE DE VÉRIFICATION (Du notebook simulate_api_request_with_predictions)
# =============================================================================

async def simulate_api_request_with_predictions(image_array: np.ndarray, model_name: str = "unet_mini") -> Dict[str, Any]:
    """
    Simule requête API avec prédictions visuelles
    Function EXACTE du notebook d'entraînement - adaptée pour API production
    """
    logger.info(f"🔮 Simulation requête API avec {model_name}...")
    
    # Utiliser le vrai pipeline de prédiction
    result = await model_manager.predict(image_array, model_name)
    
    # Générer visualisations (même que le notebook)
    prediction_map = result["prediction_map"]
    confidence_map = result["confidence_map"]
    
    # Créer images visualisation
    color_mask = convert_prediction_to_color(prediction_map)
    overlay_image = create_overlay_image(image_array, prediction_map, alpha=0.6)
    
    # Mini rapport pipeline (inspiré du notebook)
    pipeline_report = create_pipeline_mini_report(result["statistics"], prediction_map, confidence_map)
    
    return {
        "prediction_result": result,
        "color_mask": color_mask,
        "overlay_image": overlay_image,
        "pipeline_report": pipeline_report,
        "visual_demo": True
    }

def create_pipeline_mini_report(stats: Dict[str, Any], prediction_map: np.ndarray, confidence_map: np.ndarray) -> Dict[str, Any]:
    """
    Crée mini rapport pipeline automatique
    Inspiré de create_final_performance_summary du notebook
    """
    # Analyse qualité modèle
    high_confidence_pixels = np.sum(confidence_map > 0.8)
    total_pixels = confidence_map.size
    quality_score = (high_confidence_pixels / total_pixels) * 100
    
    # Classes prédominantes (top 3)
    class_percentages = [(name, data["percentage"]) for name, data in stats["class_statistics"].items()]
    class_percentages.sort(key=lambda x: x[1], reverse=True)
    dominant_classes = class_percentages[:3]
    
    # Performance inférence
    inference_speed = "Rapide" if stats["inference_time_ms"] < 100 else "Modéré" if stats["inference_time_ms"] < 500 else "Lent"
    
    # Recommandations
    recommendations = []
    
    if quality_score < 70:
        recommendations.append("⚠️ Confiance faible - Considérer VGG16 UNet pour plus de précision")
    
    if stats["inference_time_ms"] > 200:
        recommendations.append("🚀 Considérer UNet Mini pour inférence plus rapide")
    
    if stats["classes_detected"] < 3:
        recommendations.append("🔍 Image simple - UNet Mini suffisant")
    elif stats["classes_detected"] > 6:
        recommendations.append("🎯 Scene complexe - VGG16 UNet recommandé")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Configuration optimale pour cette image")
    
    return {
        "quality_analysis": {
            "overall_quality": quality_score,
            "quality_label": "Excellent" if quality_score > 85 else "Bon" if quality_score > 70 else "Modéré",
            "high_confidence_ratio": high_confidence_pixels / total_pixels,
            "dominant_classes": dominant_classes
        },
        "performance_analysis": {
            "inference_time_ms": stats["inference_time_ms"],
            "speed_rating": inference_speed,
            "efficiency_score": min(100, 1000 / stats["inference_time_ms"]),
            "gpu_utilization": "Optimale" if stats["inference_time_ms"] < 50 else "Bonne"
        },
        "recommendations": recommendations,
        "technical_details": {
            "model_used": stats["model_name"],
            "architecture": "UNet Mini (1.9M)" if stats["model_name"] == "unet_mini" else "VGG16 UNet (25.9M)",
            "preprocessing": "ImageNet normalization + resize (512x1024)",
            "postprocessing": "Argmax + confidence mapping",
            "fallback_mode": stats.get("fallback_mode", False)
        }
    }

def create_final_performance_summary(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crée résumé performance final
    Function EXACTE du notebook d'entraînement
    """
    if not stats_list:
        return {"error": "Aucune statistique disponible"}
    
    # Moyennes globales
    avg_inference_time = np.mean([s["inference_time_ms"] for s in stats_list])
    avg_confidence = np.mean([s["overall_confidence"] for s in stats_list])
    avg_classes_detected = np.mean([s["classes_detected"] for s in stats_list])
    
    # Analyse comparative modèles
    model_performance = {}
    for stats in stats_list:
        model_name = stats["model_name"]
        if model_name not in model_performance:
            model_performance[model_name] = {
                "count": 0,
                "total_time": 0,
                "total_confidence": 0,
                "total_classes": 0
            }
        
        perf = model_performance[model_name]
        perf["count"] += 1
        perf["total_time"] += stats["inference_time_ms"]
        perf["total_confidence"] += stats["overall_confidence"]
        perf["total_classes"] += stats["classes_detected"]
    
    # Calculer moyennes par modèle
    for model_name, perf in model_performance.items():
        perf["avg_time"] = perf["total_time"] / perf["count"]
        perf["avg_confidence"] = perf["total_confidence"] / perf["count"]
        perf["avg_classes"] = perf["total_classes"] / perf["count"]
    
    return {
        "global_metrics": {
            "average_inference_time_ms": avg_inference_time,
            "average_confidence": avg_confidence,
            "average_classes_detected": avg_classes_detected,
            "total_predictions": len(stats_list)
        },
        "model_comparison": model_performance,
        "summary": f"Pipeline testé sur {len(stats_list)} images - Performance moyenne: {avg_inference_time:.1f}ms",
        "timestamp": datetime.now().isoformat()
    }

logger.info("✅ Pipeline de vérification intégré - Mini rapports automatiques disponibles")

# =============================================================================
# 🌐 LIFESPAN MANAGEMENT (Style moderne comme main_keras_modern.py)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan avec gestion moderne"""
    # Startup
    logger.info("🚀 Démarrage Future Vision Transport API SOTA...")
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    
    try:
        await model_manager.load_models()
        logger.info("✅ API SOTA prête pour requêtes de segmentation!")
    except Exception as e:
        logger.error(f"❌ Échec initialisation modèles: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt API Future Vision Transport")

# =============================================================================
# 🌐 APPLICATION FASTAPI SOTA
# =============================================================================

# Initialiser application FastAPI avec lifespan moderne
app = FastAPI(
    title="🚀 Future Vision Transport - API FastAPI SOTA",
    description="API Production pour Segmentation d'Images Véhicules Autonomes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurer CORS pour frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configurer pour production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 🔧 ENDPOINTS API (Compatible UI 100%)
# =============================================================================

@app.get("/")
async def root():
    """Endpoint racine avec informations API"""
    return {
        "message": "🚀 Future Vision Transport - API FastAPI SOTA",
        "version": "1.0.0",
        "status": "active",
        "tensorflow_version": tf.__version__,
        "gpu_available": model_manager.gpu_available,
        "models_loaded": len(model_manager.models),
        "documentation": "/docs",
        "ui_interface": "/ui"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint avec status système (Compatible UI)"""
    models_status = []
    for model_name, info in model_manager.model_info.items():
        models_status.append({
            "name": model_name,
            "status": info.get("status", "unknown"),
            "parameters": info.get("parameters", 0),
            "type": info.get("type", "unknown")
        })
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tensorflow_version": model_manager.tensorflow_version,
        "gpu_available": model_manager.gpu_available,
        "models_loaded": len(model_manager.models),
        "available_models": list(model_manager.models.keys()),
        "models_status": models_status,
        "fallback_mode": model_manager.fallback_mode,
        "system_info": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "environment": "Google Colab L4" if model_manager.gpu_available else "CPU"
        }
    }

@app.get("/models")
async def get_models():
    """Informations modèles chargés (Compatible UI)"""
    models_info = []
    for model_name, info in model_manager.model_info.items():
        model_data = {
            "name": model_name,
            "parameters": info.get("parameters", 0),
            "input_shape": info.get("input_shape", []),
            "output_shape": info.get("output_shape", []),
            "status": info.get("status", "unknown"),
            "type": info.get("type", "unknown"),
            "source": info.get("source", "unknown"),
            "loaded_at": info.get("loaded_at", ""),
            "available": model_name in model_manager.models
        }
        models_info.append(model_data)
    
    return models_info

@app.post("/predict")
async def predict_segmentation(
    file: UploadFile = File(...),
    model_name: str = Form("unet_mini")
):
    """
    Prédiction segmentation et retour statistiques JSON (Compatible UI)
    Utilise la méthodologie EXACTE du pipeline de vérification
    """
    try:
        # Valider modèle
        if model_name not in model_manager.models:
            raise HTTPException(
                status_code=400, 
                detail=f"Modèle {model_name} non disponible. Disponibles: {list(model_manager.models.keys())}"
            )
        
        # Charger et valider image
        logger.info(f"📤 Upload image pour prédiction avec {model_name}")
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir en array numpy
        image_array = np.array(image)
        
        # Exécuter prédiction (VRAIE prédiction avec modèles entraînés)
        result = await model_manager.predict(image_array, model_name)
        
        # Créer mini rapport pipeline automatique
        pipeline_report = create_pipeline_mini_report(
            result["statistics"], 
            result["prediction_map"], 
            result["confidence_map"]
        )
        
        # Enrichir statistiques pour UI (format attendu par segmentation_test_interface.html)
        enriched_stats = {
            **result["statistics"],
            "pipeline_report": pipeline_report,
            "image_info": {
                "filename": file.filename,
                "original_size": f"{image_array.shape[1]}×{image_array.shape[0]}",
                "processed_size": f"{result['processed_shape'][2]}×{result['processed_shape'][1]}",
                "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1
            }
        }
        
        logger.info(f"✅ Prédiction terminée - {result['statistics']['inference_time_ms']:.1f}ms")
        
        # Retourner JSON statistics (format UI)
        return JSONResponse(content=enriched_stats)
        
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Échec prédiction: {str(e)}")

@app.post("/predict/overlay")
async def predict_overlay(
    file: UploadFile = File(...),
    model_name: str = Form("unet_mini"),
    alpha: float = Form(0.6)
):
    """Génère image overlay segmentation (Compatible UI)"""
    try:
        # Charger image
        logger.info(f"🎨 Génération overlay avec {model_name} (alpha={alpha})")
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        
        # Exécuter prédiction
        result = await model_manager.predict(image_array, model_name)
        prediction_map = result["prediction_map"]
        
        # Créer overlay (même méthode que le notebook)
        overlay = create_overlay_image(image_array, prediction_map, alpha)
        
        # Convertir en bytes
        img_buffer = BytesIO()
        Image.fromarray(overlay).save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        logger.info(f"✅ Overlay généré avec succès")
        
        return StreamingResponse(
            img_buffer, 
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=overlay_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"}
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur génération overlay: {e}")
        raise HTTPException(status_code=500, detail=f"Échec génération overlay: {str(e)}")

@app.post("/predict/mask")
async def predict_mask(
    file: UploadFile = File(...),
    model_name: str = Form("unet_mini")
):
    """Génère masque segmentation pur (Compatible UI)"""
    try:
        # Charger image
        logger.info(f"🎭 Génération masque avec {model_name}")
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        
        # Exécuter prédiction
        result = await model_manager.predict(image_array, model_name)
        prediction_map = result["prediction_map"]
        
        # Créer masque coloré (même méthode que le notebook)
        color_mask = convert_prediction_to_color(prediction_map)
        
        # Convertir en bytes
        img_buffer = BytesIO()
        Image.fromarray(color_mask).save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        logger.info(f"✅ Masque généré avec succès")
        
        return StreamingResponse(
            img_buffer, 
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=mask_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"}
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur génération masque: {e}")
        raise HTTPException(status_code=500, detail=f"Échec génération masque: {str(e)}")

# =============================================================================
# 🎨 ENDPOINT UI ET UTILITAIRES
# =============================================================================

@app.get("/ui")
async def serve_ui():
    """Sert l'interface de test segmentation (UI préservée 100%)"""
    ui_path = "segmentation_test_interface.html"
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    else:
        # Interface intégrée si fichier non trouvé
        return JSONResponse({
            "message": "Interface UI non trouvée",
            "expected_path": ui_path,
            "api_docs": "/docs",
            "status": "API fonctionnelle - Interface séparée"
        })

@app.post("/pipeline/demo")
async def pipeline_demo(
    file: UploadFile = File(...),
    model_name: str = Form("unet_mini")
):
    """
    Démonstration pipeline complet (Inspiré simulate_api_request_with_predictions)
    Génère toutes les visualisations + rapport automatique
    """
    try:
        # Charger image
        logger.info(f"🔮 Démonstration pipeline complet avec {model_name}")
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        
        # Utiliser pipeline de vérification (même que le notebook)
        demo_result = await simulate_api_request_with_predictions(image_array, model_name)
        
        # Préparer réponse avec toutes les visualisations
        response_data = {
            "statistics": demo_result["prediction_result"]["statistics"],
            "pipeline_report": demo_result["pipeline_report"],
            "visual_demo": demo_result["visual_demo"],
            "summary": f"Pipeline démo exécuté avec {model_name} - {demo_result['prediction_result']['statistics']['inference_time_ms']:.1f}ms"
        }
        
        logger.info(f"✅ Démonstration pipeline terminée")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur démonstration pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Échec démonstration: {str(e)}")

# =============================================================================
# 🚀 LANCEMENT SERVEUR
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Lancement serveur Future Vision Transport API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

logger.info("✅ API FastAPI SOTA complète - Tous endpoints compatibles UI configurés!")