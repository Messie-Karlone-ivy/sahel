"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              🌍 SAHEL.AI CONFIG                               ║
║         Système d'Analyse Hybride pour l'Économie par Intelligence           ║
║                              Artificielle                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Configuration centrale pour le projet SAHEL.AI
Prédiction de l'indice BRVM via données multi-modales
"""

from dataclasses import dataclass, field
from typing import List, Dict
import os

# ═══════════════════════════════════════════════════════════════════════════════
# 🌍 CONFIGURATION GÉOGRAPHIQUE - ZONE UEMOA (8 PAYS)
# ═══════════════════════════════════════════════════════════════════════════════

UEMOA_COUNTRIES = {
    "CIV": {
        "name": "Côte d'Ivoire",
        "capital": "Abidjan",
        "flag": "🇨🇮",
        "gdp_weight": 0.40,  # 40% du PIB UEMOA
        "main_exports": ["cacao", "café", "huile_palme"],
        "coordinates": {"lat": 7.54, "lon": -5.55}
    },
    "SEN": {
        "name": "Sénégal",
        "capital": "Dakar",
        "flag": "🇸🇳",
        "gdp_weight": 0.15,
        "main_exports": ["poisson", "phosphates", "arachides"],
        "coordinates": {"lat": 14.50, "lon": -14.45}
    },
    "MLI": {
        "name": "Mali",
        "capital": "Bamako",
        "flag": "🇲🇱",
        "gdp_weight": 0.12,
        "main_exports": ["or", "coton"],
        "coordinates": {"lat": 17.57, "lon": -4.00}
    },
    "BFA": {
        "name": "Burkina Faso",
        "capital": "Ouagadougou",
        "flag": "🇧🇫",
        "gdp_weight": 0.10,
        "main_exports": ["or", "coton"],
        "coordinates": {"lat": 12.24, "lon": -1.56}
    },
    "BEN": {
        "name": "Bénin",
        "capital": "Porto-Novo",
        "flag": "🇧🇯",
        "gdp_weight": 0.08,
        "main_exports": ["coton", "noix_cajou"],
        "coordinates": {"lat": 9.31, "lon": 2.32}
    },
    "NER": {
        "name": "Niger",
        "capital": "Niamey",
        "flag": "🇳🇪",
        "gdp_weight": 0.08,
        "main_exports": ["uranium", "or"],
        "coordinates": {"lat": 17.61, "lon": 8.08}
    },
    "TGO": {
        "name": "Togo",
        "capital": "Lomé",
        "flag": "🇹🇬",
        "gdp_weight": 0.05,
        "main_exports": ["phosphates", "coton"],
        "coordinates": {"lat": 8.62, "lon": 0.82}
    },
    "GNB": {
        "name": "Guinée-Bissau",
        "capital": "Bissau",
        "flag": "🇬🇼",
        "gdp_weight": 0.02,
        "main_exports": ["noix_cajou"],
        "coordinates": {"lat": 11.80, "lon": -15.18}
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 CONFIGURATION DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration de l'architecture du modèle SAHEL.AI"""

    # Nom du modèle
    model_name: str = "SAHEL_AI_v2.0"

    # Mode d'encodeurs: 'pretrained' ou 'custom'
    encoder_mode: str = "pretrained"
    
    # Dimensions des embeddings
    satellite_embedding_dim: int = 128
    text_embedding_dim: int = 256
    mobile_embedding_dim: int = 64
    commodity_embedding_dim: int = 32
    
    # Architecture principale
    fusion_dim: int = 512
    gnn_hidden_dim: int = 256
    num_gnn_layers: int = 3
    num_attention_heads: int = 8
    
    # Séquences temporelles
    sequence_length: int = 60  # 60 jours d'historique
    prediction_horizon: int = 30  # Prédiction à 30 jours
    
    # Entraînement
    batch_size: int = 32
    learning_rate: float = 1e-4
    dropout_rate: float = 0.2
    num_epochs: int = 100
    
    # Features par source de données
    satellite_features: List[str] = field(default_factory=lambda: [
        "nightlight_intensity",
        "nightlight_change",
        "ndvi_vegetation",
        "port_ship_count",
        "urban_expansion_rate"
    ])
    
    mobile_features: List[str] = field(default_factory=lambda: [
        "transaction_volume",
        "transaction_count",
        "active_users",
        "cross_border_transfers"
    ])
    
    commodity_features: List[str] = field(default_factory=lambda: [
        "cacao_price",
        "coffee_price",
        "cotton_price",
        "gold_price",
        "oil_price"
    ])

MODEL_CONFIG = ModelConfig()

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURATION UI - THÈME AFRO-FUTURISTE
# ═══════════════════════════════════════════════════════════════════════════════

UI_THEME = {
    # Couleurs principales
    "gold_primary": "#FFB800",
    "gold_light": "#FFD54F",
    "gold_dark": "#C79100",
    
    # Fonds
    "navy_deep": "#0A0A1A",
    "navy_medium": "#1A1A2E",
    "navy_light": "#16213E",
    
    # Accents
    "accent_red": "#E94560",
    "accent_cyan": "#00D9A5",
    "accent_purple": "#9D4EDD",
    
    # Texte
    "text_primary": "#FFFFFF",
    "text_secondary": "#A0AEC0",
    
    # Fonts
    "font_display": "Orbitron",
    "font_body": "Space Grotesk"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 SOURCES DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

DATA_SOURCES = {
    "satellite": {
        "nightlights": "NASA VIIRS Nighttime Lights",
        "ndvi": "MODIS Vegetation Index",
        "port_activity": "Sentinel-1 SAR Ship Detection"
    },
    "mobile": {
        "transactions": "Mobile Money Aggregated Data",
        "activity": "Network Activity Index"
    },
    "sentiment": {
        "twitter": "Twitter/X API v2",
        "news": "GDELT Project + Local News"
    },
    "market": {
        "brvm": "BRVM Official Data",
        "commodities": "World Bank Commodity Prices"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 📁 CHEMINS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "data": os.path.join(BASE_DIR, "data"),
    "models": os.path.join(BASE_DIR, "models"),
    "outputs": os.path.join(BASE_DIR, "outputs"),
    "assets": os.path.join(BASE_DIR, "assets")
}

# Créer les répertoires si nécessaire
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 MODÈLES PRÉ-ENTRAÎNÉS
# ═══════════════════════════════════════════════════════════════════════════════

PRETRAINED_MODELS = {
    # CamemBERT - Modèle NLP français pour l'analyse de sentiment
    # Référence: Martin et al. (2020) "CamemBERT: a Tasty French Language Model"
    # https://arxiv.org/abs/1911.03894
    "camembert": {
        "model_name": "camembert-base",
        "description": "Modèle BERT français pré-entraîné sur 138Go de texte",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "use_case": "Analyse de sentiment des tweets/news en français"
    },

    # Amazon Chronos - Foundation model pour séries temporelles
    # Référence: Ansari et al. (2024) "Chronos: Learning the Language of Time Series"
    # https://arxiv.org/abs/2403.07815
    "chronos": {
        "model_name": "amazon/chronos-t5-small",
        "description": "Foundation model pour prédiction de séries temporelles",
        "architecture": "T5-based",
        "training_data": "27 milliards de points de données",
        "use_case": "Prédiction des séries temporelles (satellites, mobile, commodités)"
    },

    # Variantes disponibles (pour référence)
    "chronos_variants": {
        "tiny": "amazon/chronos-t5-tiny",      # 8M params - Ultra léger
        "mini": "amazon/chronos-t5-mini",      # 20M params - Léger
        "small": "amazon/chronos-t5-small",    # 46M params - Défaut
        "base": "amazon/chronos-t5-base",      # 200M params - Précis
        "large": "amazon/chronos-t5-large"     # 710M params - Haute précision
    }
}

# Configuration spécifique pour les encodeurs pré-entraînés
PRETRAINED_ENCODER_CONFIG = {
    "camembert": {
        "freeze_base": True,          # Geler les poids (recommandé)
        "pooling_strategy": "cls",    # Utiliser le token [CLS]
        "max_length": 128,            # Longueur max des séquences
        "output_dim": 256             # Dimension de sortie (compatibilité)
    },
    "chronos": {
        "aggregation": "attention",   # Agrégation par attention
        "output_dim": 256,            # Dimension de sortie
        "prediction_length": 16       # Horizon de prédiction interne
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 📚 CITATIONS ACADÉMIQUES
# ═══════════════════════════════════════════════════════════════════════════════

ACADEMIC_CITATIONS = {
    "camembert": """
    @inproceedings{martin2020camembert,
        title={CamemBERT: a Tasty French Language Model},
        author={Martin, Louis and Muller, Benjamin and Suárez, Pedro Javier Ortiz
                and Dupont, Yoann and Romary, Laurent and de la Clergerie, Éric
                and Seddah, Djamé and Sagot, Benoît},
        booktitle={Proceedings of the 58th Annual Meeting of the Association
                   for Computational Linguistics},
        pages={7203--7219},
        year={2020}
    }
    """,

    "chronos": """
    @article{ansari2024chronos,
        title={Chronos: Learning the Language of Time Series},
        author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner
                and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin
                and Shchur, Oleksandr and Rangapuram, Syama Sundar
                and Pineda Arango, Sebastian and Kapoor, Shubham and others},
        journal={arXiv preprint arXiv:2403.07815},
        year={2024}
    }
    """
}
