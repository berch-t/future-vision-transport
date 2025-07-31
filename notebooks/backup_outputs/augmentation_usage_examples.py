# Exemples d'usage du syst�me d'augmentation

# EXAMPLE_1_BASIC

# Usage de base - Pipeline d'entra�nement
from notebooks.data_augmentation import ComprehensiveAugmentationPipeline

# Initialisation
aug_pipeline = ComprehensiveAugmentationPipeline(mode='training')

# Application
augmented_image, augmented_mask = aug_pipeline(image, mask)
        

# EXAMPLE_2_GENERATOR

# Int�gration avec g�n�rateur de donn�es
from notebooks.data_augmentation import AugmentedCityscapesSequence

# Cr�er g�n�rateur avec augmentation
train_generator = AugmentedCityscapesSequence(
    data_list=train_data,
    batch_size=16,
    mode='training'
)

# Utilisation avec Keras
model.fit(train_generator, epochs=50)
        

# EXAMPLE_3_CUSTOM

# Configuration personnalis�e
custom_config = AUGMENTATION_CONFIG.copy()
custom_config['augmentation_probabilities']['geometric_p'] = 0.8

custom_pipeline = ComprehensiveAugmentationPipeline(
    config=custom_config, 
    mode='training'
)
        

# EXAMPLE_4_INDIVIDUAL

# Utilisation des modules individuels
from notebooks.data_augmentation import GeometricAugmentations

geo_aug = GeometricAugmentations(AUGMENTATION_CONFIG)
geo_pipeline = geo_aug.create_basic_geometric_pipeline()

result = geo_pipeline(image=image, mask=mask)
        

