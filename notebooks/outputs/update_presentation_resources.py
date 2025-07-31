#!/usr/bin/env python3
"""
🎨 Script de mise à jour des ressources de présentation
Remplace automatiquement les placeholders par les vrais fichiers
"""

import os
import shutil
from pathlib import Path
import json

# Configuration des chemins
PROJECT_ROOT = Path("C:/Tonton/OpenClassrooms/Projet_7_traiter_images_systeme_embarque_voiture_autonome")
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "notebooks" / "outputs"
PRESENTATION_FILE = OUTPUTS_DIR / "presentation_complete.html"

# Mapping des placeholders vers les fichiers réels
FIGURE_MAPPINGS = {
    "📊 figures/8_categories_distribution.png": "8_categories_distribution.png",
    "📊 figures/architecture_comparison.png": "architecture_comparison.png", 
    "📊 figures/loss_comparison.png": "loss_comparison.png",
    "📊 figures/generator_samples.png": "generator_samples.png",
    "📊 figures/conversion_example.png": "conversion_example.png",
    "📊 figures/geometric_augmentations.png": "geometric_augmentations.png",
    "📊 figures/photometric_augmentations.png": "photometric_augmentations.png",
    "📊 figures/weather_augmentations.png": "weather_augmentations.png",
    "📊 figures/augmentation_class_impact.png": "augmentation_class_impact.png",
    "📊 figures/comprehensive_balancing_impact.png": "comprehensive_balancing_impact.png",
    "📊 figures/enhanced_comprehensive_impact.png": "enhanced_comprehensive_impact.png",
    "📊 figures/test_pipeline_enhanced_results.png": "test_pipeline_enhanced_results.png",
    "📊 figures/test_real_cityscapes_enhanced.png": "test_real_cityscapes_enhanced.png",
    "📊 figures/sota_benchmarks.png": "sota_benchmarks.png"
}

CONFIG_MAPPINGS = {
    "📁 outputs/architecture_research_summary.json": "architecture_research_summary.json",
    "📁 outputs/model_implementations.json": "model_implementations.json",
    "📁 outputs/loss_functions_config.json": "loss_functions_config.json",
    "📁 outputs/data_generator_config.json": "data_generator_config.json",
    "📁 outputs/augmentation_config.json": "augmentation_config.json",
    "📁 outputs/enhanced_comprehensive_balancing_config.json": "enhanced_comprehensive_balancing_config.json"
}

def check_file_exists(file_path):
    """Vérifie si un fichier existe"""
    return Path(file_path).exists()

def convert_image_to_base64(image_path):
    """Convertit une image en base64 pour embedding dans HTML"""
    import base64
    
    if not check_file_exists(image_path):
        return None
        
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
        file_ext = Path(image_path).suffix.lower()
        
        if file_ext == '.png':
            mime_type = 'image/png'
        elif file_ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif file_ext == '.svg':
            mime_type = 'image/svg+xml'
        else:
            mime_type = 'image/png'  # default
            
        return f"data:{mime_type};base64,{b64_string}"

def create_image_html(image_b64, alt_text, max_width="600px", max_height="300px"):
    """Crée le HTML pour afficher une image"""
    if image_b64:
        return f'''<img src="{image_b64}" alt="{alt_text}" style="max-width: {max_width}; max-height: {max_height}; width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">'''
    else:
        return f'''<div class="placeholder-img">{alt_text}<br><small>❌ Fichier non trouvé</small></div>'''

def load_config_content(config_path):
    """Charge le contenu d'un fichier de configuration"""
    if not check_file_exists(config_path):
        return None
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix == '.json':
                data = json.load(f)
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return f.read()
    except Exception as e:
        print(f"❌ Erreur lecture {config_path}: {e}")
        return None

def create_config_html(config_content, title):
    """Crée le HTML pour afficher une configuration"""
    if config_content:
        return f'''
        <details style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;">
            <summary style="cursor: pointer; font-weight: bold; color: #FFD700;">{title}</summary>
            <pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 0.8em; margin-top: 10px;"><code>{config_content}</code></pre>
        </details>'''
    else:
        return f'''<div class="placeholder-img">{title}<br><small>❌ Configuration non trouvée</small></div>'''

def update_presentation():
    """Met à jour la présentation avec les vraies ressources"""
    
    print("🔄 Mise à jour de la présentation avec les ressources réelles...")
    
    # Lire le fichier de présentation
    if not check_file_exists(PRESENTATION_FILE):
        print(f"❌ Fichier de présentation non trouvé: {PRESENTATION_FILE}")
        return False
    
    with open(PRESENTATION_FILE, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    updated_count = 0
    
    # Remplacer les placeholders d'images
    print("\n📊 Traitement des images...")
    for placeholder, filename in FIGURE_MAPPINGS.items():
        image_path = FIGURES_DIR / filename
        print(f"  🔍 Recherche: {filename}")
        
        if check_file_exists(image_path):
            print(f"  ✅ Trouvée: {image_path}")
            image_b64 = convert_image_to_base64(image_path)
            if image_b64:
                image_html = create_image_html(image_b64, filename)
                html_content = html_content.replace(
                    f'<div class="placeholder-img">\n                {placeholder}\n            </div>',
                    image_html
                )
                updated_count += 1
                print(f"  🎨 Intégrée dans la présentation")
            else:
                print(f"  ❌ Erreur conversion base64")
        else:
            print(f"  ⚠️ Non trouvée: {image_path}")
    
    # Remplacer les placeholders de configuration
    print("\n📁 Traitement des configurations...")
    for placeholder, filename in CONFIG_MAPPINGS.items():
        config_path = OUTPUTS_DIR / filename
        print(f"  🔍 Recherche: {filename}")
        
        if check_file_exists(config_path):
            print(f"  ✅ Trouvée: {config_path}")
            config_content = load_config_content(config_path)
            if config_content:
                config_html = create_config_html(config_content, filename)
                html_content = html_content.replace(
                    f'<div class="placeholder-img">\n                {placeholder}\n            </div>',
                    config_html
                )
                updated_count += 1
                print(f"  📝 Intégrée dans la présentation")
            else:
                print(f"  ❌ Erreur lecture configuration")
        else:
            print(f"  ⚠️ Non trouvée: {config_path}")
    
    # Sauvegarder la présentation mise à jour
    updated_file = OUTPUTS_DIR / "presentation_with_resources.html"
    with open(updated_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Présentation mise à jour sauvegardée: {updated_file}")
    print(f"📊 {updated_count} ressources intégrées avec succès")
    
    return True

def generate_resource_report():
    """Génère un rapport des ressources disponibles"""
    
    print("\n📋 RAPPORT DES RESSOURCES DISPONIBLES")
    print("=" * 50)
    
    # Vérifier les figures
    print("\n📊 FIGURES:")
    figures_found = 0
    figures_total = len(FIGURE_MAPPINGS)
    
    for placeholder, filename in FIGURE_MAPPINGS.items():
        image_path = FIGURES_DIR / filename
        if check_file_exists(image_path):
            size_mb = image_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.1f} MB)")
            figures_found += 1
        else:
            print(f"  ❌ {filename}")
    
    # Vérifier les configurations
    print("\n📁 CONFIGURATIONS:")
    configs_found = 0
    configs_total = len(CONFIG_MAPPINGS)
    
    for placeholder, filename in CONFIG_MAPPINGS.items():
        config_path = OUTPUTS_DIR / filename
        if check_file_exists(config_path):
            size_kb = config_path.stat().st_size / 1024
            print(f"  ✅ {filename} ({size_kb:.1f} KB)")
            configs_found += 1
        else:
            print(f"  ❌ {filename}")
    
    # Résumé
    print(f"\n📊 RÉSUMÉ:")
    print(f"  Figures: {figures_found}/{figures_total} ({figures_found/figures_total*100:.1f}%)")
    print(f"  Configurations: {configs_found}/{configs_total} ({configs_found/configs_total*100:.1f}%)")
    print(f"  Total: {figures_found + configs_found}/{figures_total + configs_total} ({(figures_found + configs_found)/(figures_total + configs_total)*100:.1f}%)")
    
    return figures_found, configs_found

if __name__ == "__main__":
    print("🎨 MISE À JOUR PRÉSENTATION FUTURE VISION TRANSPORT")
    print("=" * 55)
    
    # Générer le rapport des ressources
    figures_found, configs_found = generate_resource_report()
    
    # Demander confirmation pour la mise à jour
    if figures_found > 0 or configs_found > 0:
        print(f"\n🔄 Mettre à jour la présentation avec {figures_found + configs_found} ressources disponibles ?")
        response = input("Continuer ? (o/N): ").lower().strip()
        
        if response in ['o', 'oui', 'y', 'yes']:
            success = update_presentation()
            if success:
                print("\n🎉 Mise à jour terminée avec succès !")
                print("📱 Ouvrez 'presentation_with_resources.html' pour voir le résultat")
            else:
                print("\n❌ Erreur durant la mise à jour")
        else:
            print("⚠️ Mise à jour annulée")
    else:
        print("\n⚠️ Aucune ressource disponible pour la mise à jour")
        print("💡 Assurez-vous que les notebooks ont été exécutés et ont généré les figures/configurations")
    
    print(f"\n🎯 Script terminé")
