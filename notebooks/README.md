# 📊 Notebooks - Projet Vision Embarquée

## 📁 Structure
```
notebooks/
├── figures/                               # Visualisations générées
├── outputs/                               # Fichiers CSV/JSON générés
├── 1_EDA.py                               # Analyse des données
├── 2.1_SOTA_Architecture_Research.py      # Recherche architectures SOTA
├── 2.2_Model_Implementation.py            # Implémentation des modèles
├── 2.3_Advanced_Loss_Functions.py         # Fonctions de perte avancées
├── 2.4_Data_Generator.py                  # Générateur de données optimisé
├── 2.5_Data_Augmentation.py               # Augmentation données Albumentations
├── 2.6_Class_Balancing.py                 # Équilibrage des classes
├── 2.6_Class_Balancing_Enhanced.py        # Équilibrage enhanced + augmentation
├── 2.7_Training_Pipeline.py               # Pipeline d'entraînement coordonné
└── README.md                              # Documentation
```

## 🎯 Phase 1 : Exploration des Données (EDA)

### 📊 Sorties générées

#### Fichiers de données
- `dataset_structure.csv` - Structure du dataset par ville/split
- `class_distribution_sample.csv` - Distribution des classes originales
- `8_categories_distribution_corrected.csv` - Distribution des 8 catégories cibles
- `class_mapping.json` - Mapping complet 30→8 catégories
- `eda_summary.json` - Rapport de synthèse

#### Visualisations
- `sample_annotations.png` - Échantillons d'annotations
- `dataset_structure.html` - Structure interactive du dataset
- `class_distribution.png` - Distribution des classes
- `8_categories_distribution.png` - Distribution des catégories cibles
- `conversion_example.png` - Exemple de conversion d'annotations

### 🎯 Objectifs de l'EDA

1. ✅ **Comprendre la structure** du dataset Cityscapes
2. ✅ **Analyser la distribution** des 30 classes originales
3. ✅ **Créer le mapping** vers 8 catégories principales
4. ✅ **Visualiser les données** pour identifier les patterns
5. ✅ **Préparer les artefacts** pour les phases suivantes

## 🚀 Phase 2 : Développement Modèle IA

### 📊 Conception des Modèles (2.1-2.3)

#### 2.1_SOTA_Architecture_Research.py
- **Objectif** : Recherche et analyse des architectures SOTA
- **Sorties** : Comparatif U-Net, DeepLab, Segformer pour embarqué
- **Artefacts** : Tableau performance vs complexité, justifications techniques

#### 2.2_Model_Implementation.py  
- **Objectif** : Implémentation 3 architectures en Keras
- **Sorties** : Classes modulaires, backbones pré-entraînés
- **Artefacts** : Modèles testés, configurations optimisées

#### 2.3_Advanced_Loss_Functions.py
- **Objectif** : Fonctions de perte adaptées au déséquilibre
- **Sorties** : Dice Loss, Focal Loss, Combined Loss
- **Artefacts** : Benchmark losses, calcul poids automatique

### 📊 Générateur de Données (2.4-2.6)

#### 2.4_Data_Generator.py
- **Objectif** : Générateur tf.keras.utils.Sequence haute performance
- **Sorties** : Pipeline optimisé, gestion mémoire intelligente
- **Artefacts** : `CityscapesSequence` class, benchmarks performance

#### 2.5_Data_Augmentation.py
- **Objectif** : Pipeline Albumentations optimisé >1000 FPS
- **Sorties** : Augmentations cohérentes image+mask, adaptatives
- **Artefacts** : `data_augmentation_config.json`, visualisations

#### 2.6_Class_Balancing.py
- **Objectif** : Stratégies d'équilibrage pour classes déséquilibrées
- **Sorties** : 5 stratégies, calculateur poids, curriculum learning
- **Artefacts** : `comprehensive_balancing_config.json`

#### 2.6_Class_Balancing_Enhanced.py
- **Objectif** : Pipeline coordonné équilibrage + augmentation
- **Sorties** : Intégration validée 2.5, stratégies adaptatives
- **Artefacts** : `enhanced_comprehensive_balancing_config.json`

### 📊 Entraînement et Évaluation (2.7)

#### 2.7_Training_Pipeline.py
- **Objectif** : Pipeline d'entraînement reproductible avec monitoring
- **Sorties** : Entraînement coordonné, callbacks optimisés
- **Artefacts** : Modèles entraînés, logs détaillés

#### 2.8_Evaluation_Metrics.py
- **Objectif** : Métriques avancées pour segmentation déséquilibrée
- **Sorties** : IoU par classe, métriques robustes, visualisations
- **Artefacts** : Rapports d'évaluation, matrices de confusion

#### 2.9_Model_Comparison.py
- **Objectif** : Comparaison et sélection du modèle optimal
- **Sorties** : Critères pondérés, tests robustesse
- **Artefacts** : Rapport technique (10 pages), modèle final

## 🔧 Fonctionnalités Avancées

### 🎯 Innovations Implémentées
- **Pipeline coordonné** : Équilibrage + Augmentation + Entraînement
- **Optimisations temps réel** : >1000 FPS maintenu
- **Stratégies adaptatives** : Ajustement automatique selon contenu
- **Monitoring enhanced** : Métriques en temps réel
- **Compatibilité production** : Contraintes embarquées respectées

### 📈 Améliorations Attendues
- **Classes rares** : +300-600% représentation
- **IoU Person** : +250% (0.30 → 0.75+)
- **IoU Object** : +300% (0.20 → 0.60+)
- **mIoU global** : +15-25% (0.68 → 0.78-0.85)

## 📋 Prérequis

```bash
pip install numpy pandas matplotlib seaborn plotly opencv-python pillow
pip install scikit-image tqdm albumentations tensorflow keras
```

## ⚠️ Notes importantes

- **Framework** : Keras/TensorFlow (requis équipe)
- **Contraintes** : Optimisé embarqué, <100ms inférence, mIoU ≥65%
- **Dataset** : Cityscapes (images + annotations requises pour entraînement)
- **Performance** : Pipeline validé >1000 FPS avec augmentation temps réel

---
