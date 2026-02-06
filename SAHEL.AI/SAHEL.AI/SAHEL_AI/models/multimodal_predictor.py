"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 SAHEL.AI v2.0 - MODÈLE MULTI-MODAL                      ║
║                                                                               ║
║   Architecture Innovante avec Modèles Pré-entraînés:                         ║
║   ┌──────────────────────────────────────────────────────────────────────┐   ║
║   │  🛰️ Satellite   ──┐                                                  │   ║
║   │  📱 Mobile      ──┼──► 🔀 Cross-Modal ──► 🕸️ GNN ──► 📈 Prediction  │   ║
║   │  🗣️ Sentiment   ──┤      Attention        (8 pays)                   │   ║
║   │  🌾 Commodities ──┘      (Custom)         (Custom)                   │   ║
║   └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                               ║
║   Modèles Pré-entraînés Intégrés:                                            ║
║   • CamemBERT (NLP français) - Martin et al. 2020                            ║
║   • Chronos (Time Series) - Amazon Science 2024                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import math
import warnings

# Import des encodeurs pré-entraînés
try:
    from .pretrained_encoders import (
        CamemBERTEncoder,
        ChronosEncoder,
        CommodityChronosEncoder,
        TRANSFORMERS_AVAILABLE,
        CHRONOS_AVAILABLE
    )
    PRETRAINED_AVAILABLE = True
except ImportError:
    PRETRAINED_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    CHRONOS_AVAILABLE = False
    warnings.warn(
        "⚠️ Encodeurs pré-entraînés non disponibles. "
        "Utilisation des encodeurs custom uniquement."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 MODULES DE BASE
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal pour séquences temporelles"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalConvBlock(nn.Module):
    """Bloc convolutionnel temporel avec dilatation"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, seq_len]"""
        residual = x if self.residual is None else self.residual(x)
        
        out = self.conv(x)[:, :, :x.size(2)]  # Causal trimming
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


# ═══════════════════════════════════════════════════════════════════════════════
# 🛰️ ENCODEUR SATELLITE
# ═══════════════════════════════════════════════════════════════════════════════

class SatelliteEncoder(nn.Module):
    """
    Encodeur pour données satellites
    - Lumières nocturnes (proxy activité économique)
    - NDVI (végétation/agriculture)
    - Activité portuaire (commerce)
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, 
                 output_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Projection initiale
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # TCN avec dilatations croissantes
        self.tcn = nn.Sequential(
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=1, dropout=dropout),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout),
            TemporalConvBlock(hidden_dim, output_dim, dilation=8, dropout=dropout)
        )
        
        # Attention temporelle
        self.attention = nn.MultiheadAttention(output_dim, num_heads=4, 
                                                dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, output_dim]
        """
        # Projection
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # TCN
        x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        x = self.tcn(x)
        x = x.transpose(1, 2)  # [batch, seq_len, output_dim]
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        # Global pooling (dernière position)
        return x[:, -1, :]


# ═══════════════════════════════════════════════════════════════════════════════
# 📱 ENCODEUR MOBILE ECONOMY
# ═══════════════════════════════════════════════════════════════════════════════

class MobileEconomyEncoder(nn.Module):
    """
    Encodeur pour données de téléphonie mobile
    - Transactions Mobile Money
    - Activité réseau
    - Transferts transfrontaliers
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64,
                 output_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, output_dim]
        """
        x = self.input_proj(x)
        
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.output_proj(context)


# ═══════════════════════════════════════════════════════════════════════════════
# 🗣️ ENCODEUR SENTIMENT NLP
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentEncoder(nn.Module):
    """
    Encodeur pour analyse de sentiment
    - Twitter/X (français + langues locales)
    - News économiques
    Supporte: Français, Wolof, Bambara, Dioula
    """
    
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4,
                 output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Adaptateurs par langue
        self.language_adapters = nn.ModuleDict({
            'fr': nn.Linear(embed_dim, embed_dim),
            'wo': nn.Linear(embed_dim, embed_dim),  # Wolof
            'bm': nn.Linear(embed_dim, embed_dim),  # Bambara
        })
        
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                language: str = 'fr') -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            language: Code langue ('fr', 'wo', 'bm')
        Returns:
            [batch, output_dim]
        """
        x = self.embedding(input_ids)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        mask = None
        if attention_mask is not None:
            mask = (attention_mask == 0)
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Adaptateur langue
        if language in self.language_adapters:
            x = x + self.language_adapters[language](x)
        
        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)
        
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 🌾 ENCODEUR COMMODITÉS
# ═══════════════════════════════════════════════════════════════════════════════

class CommodityEncoder(nn.Module):
    """
    Encodeur pour prix des matières premières
    - Cacao, Café, Coton (exports majeurs)
    - Or (Mali, Burkina)
    - Pétrole (impact économique global)
    """
    
    def __init__(self, num_commodities: int = 5, hidden_dim: int = 64,
                 output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.commodity_embed = nn.Embedding(num_commodities, 16)
        
        self.price_encoder = nn.Sequential(
            nn.Linear(1 + 16, hidden_dim),  # price + embedding
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # TCN par commodité
        self.tcn = nn.Sequential(
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=1),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=2),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=4)
        )
        
        # Cross-commodity attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, 
                                                 dropout=dropout, batch_first=True)
        
        self.output_proj = nn.Linear(hidden_dim * num_commodities, output_dim)
    
    def forward(self, prices: torch.Tensor, commodity_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prices: [batch, seq_len, num_commodities]
            commodity_ids: [num_commodities]
        Returns:
            [batch, output_dim]
        """
        batch, seq_len, num_comm = prices.shape
        
        # Embed commodities
        comm_embed = self.commodity_embed(commodity_ids)  # [num_comm, 16]
        comm_embed = comm_embed.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1)
        
        # Encode each commodity
        encoded = []
        for i in range(num_comm):
            price_i = prices[:, :, i:i+1]  # [batch, seq, 1]
            combined = torch.cat([price_i, comm_embed[:, :, i, :]], dim=-1)
            
            x = self.price_encoder(combined)
            x = x.transpose(1, 2)
            x = self.tcn(x)
            x = x.transpose(1, 2)
            encoded.append(x[:, -1, :])
        
        # Stack and cross-attention
        stacked = torch.stack(encoded, dim=1)  # [batch, num_comm, hidden]
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)
        
        # Flatten
        out = attn_out.reshape(batch, -1)
        return self.output_proj(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔀 FUSION CROSS-MODALE
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalFusion(nn.Module):
    """
    Module de fusion cross-modale avec attention
    Fusionne: Satellite + Mobile + Sentiment + Commodities
    """
    
    def __init__(self, modality_dims: Dict[str, int], fusion_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.modality_names = list(modality_dims.keys())
        
        # Projections
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU()
            ) for name, dim in modality_dims.items()
        })
        
        # Tokens de modalité apprenables
        self.modality_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
            for name in modality_dims.keys()
        })
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * len(modality_dims), len(modality_dims)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dict[modality_name, [batch, feature_dim]]
        Returns:
            [batch, fusion_dim]
        """
        batch_size = list(features.values())[0].size(0)
        
        # Project and add modality tokens
        projected = []
        for name in self.modality_names:
            if name in features:
                feat = self.projections[name](features[name])
                token = self.modality_tokens[name].expand(batch_size, -1, -1)
                feat = feat.unsqueeze(1) + token
                projected.append(feat)
        
        x = torch.cat(projected, dim=1)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        x = self.norm2(x + self.feed_forward(x))
        
        # Gated fusion
        gate_input = x.reshape(batch_size, -1)
        gates = self.gate(gate_input)
        
        x = x.transpose(1, 2)
        fused = torch.bmm(x, gates.unsqueeze(-1)).squeeze(-1)
        
        return fused


# ═══════════════════════════════════════════════════════════════════════════════
# 🕸️ GRAPH NEURAL NETWORK - RELATIONS INTER-PAYS
# ═══════════════════════════════════════════════════════════════════════════════

class RegionalGNN(nn.Module):
    """
    Graph Neural Network pour modéliser les interdépendances
    entre les 8 pays de la zone UEMOA
    
    Nœuds: Pays
    Arêtes: Relations commerciales, géographiques, économiques
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 output_dim: int = 256, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.num_countries = 8
        
        # Embedding pays
        self.country_embed = nn.Embedding(self.num_countries, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph Attention Layers (implémentation simplifiée sans PyG)
        self.gat_layers = nn.ModuleList([
            self._make_gat_layer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # Poids GDP pour agrégation finale
        self.register_buffer('gdp_weights', torch.tensor([
            0.40, 0.15, 0.12, 0.10, 0.08, 0.08, 0.05, 0.02
        ]))  # CIV, SEN, MLI, BFA, BEN, NER, TGO, GNB
        
        # Matrice d'adjacence (relations économiques)
        adj = torch.tensor([
            [0, 1, 1, 1, 1, 1, 1, 0],  # CIV
            [1, 0, 1, 0, 0, 0, 0, 1],  # SEN
            [1, 1, 0, 1, 0, 1, 0, 0],  # MLI
            [1, 0, 1, 0, 1, 1, 1, 0],  # BFA
            [1, 0, 0, 1, 0, 1, 1, 0],  # BEN
            [1, 0, 1, 1, 1, 0, 0, 0],  # NER
            [1, 0, 0, 1, 1, 0, 0, 0],  # TGO
            [0, 1, 0, 0, 0, 0, 0, 0],  # GNB
        ], dtype=torch.float)
        self.register_buffer('adjacency', adj)
    
    def _make_gat_layer(self, in_dim, out_dim, num_heads, dropout):
        """Crée une couche d'attention de graphe"""
        return nn.ModuleDict({
            'query': nn.Linear(in_dim, out_dim * num_heads),
            'key': nn.Linear(in_dim, out_dim * num_heads),
            'value': nn.Linear(in_dim, out_dim * num_heads),
            'output': nn.Linear(out_dim * num_heads, out_dim),
            'dropout': nn.Dropout(dropout)
        })
    
    def _gat_forward(self, layer, x, adj):
        """Forward pass d'une couche GAT"""
        batch, num_nodes, dim = x.shape
        
        Q = layer['query'](x)
        K = layer['key'](x)
        V = layer['value'](x)
        
        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(dim)
        
        # Mask avec adjacence
        mask = adj.unsqueeze(0).expand(batch, -1, -1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = layer['dropout'](attn)
        
        out = torch.bmm(attn, V)
        return layer['output'](out)
    
    def forward(self, regional_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regional_features: [batch, fusion_dim] - Features fusionnées
        Returns:
            [batch, output_dim] - Représentation régionale
        """
        batch = regional_features.size(0)
        
        # Créer les features par pays
        x = self.input_proj(regional_features)  # [batch, hidden_dim]
        
        # Répliquer pour chaque pays + embedding pays
        country_ids = torch.arange(self.num_countries, device=x.device)
        country_emb = self.country_embed(country_ids)  # [8, hidden_dim]
        
        x = x.unsqueeze(1).expand(-1, self.num_countries, -1)  # [batch, 8, hidden]
        x = x + country_emb.unsqueeze(0)
        
        # GAT layers
        for gat, norm in zip(self.gat_layers, self.norms):
            x_new = self._gat_forward(gat, x, self.adjacency)
            x = norm(x + x_new)
        
        # Agrégation pondérée par GDP
        weights = self.gdp_weights.view(1, -1, 1)
        regional_repr = (x * weights).sum(dim=1)
        
        return self.output_proj(regional_repr)


# ═══════════════════════════════════════════════════════════════════════════════
# 📈 MODÈLE PRINCIPAL SAHEL.AI
# ═══════════════════════════════════════════════════════════════════════════════

class SAHELAI(nn.Module):
    """
    🌍 SAHEL.AI v2.0 - Modèle Principal

    Prédiction de l'indice BRVM via fusion multi-modale
    et modélisation des interdépendances régionales.

    Nouveautés v2.0:
    - Support des modèles pré-entraînés (CamemBERT, Chronos)
    - Rétrocompatibilité avec les encodeurs custom
    - Toggle runtime entre modes

    Architecture:
    1. Encodeurs spécialisés (custom ou pré-entraînés)
    2. Fusion cross-modale avec attention (custom - innovation)
    3. GNN pour relations inter-pays (custom - originalité)
    4. Têtes de prédiction multi-tâches

    Args:
        config: Configuration du modèle
        use_pretrained: Utiliser les modèles pré-entraînés (défaut: True)
        pretrained_models: Config des modèles pré-entraînés
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        use_pretrained: bool = True,
        pretrained_models: Optional[Dict] = None
    ):
        super().__init__()

        self.config = config or {
            'satellite_dim': 5,
            'mobile_dim': 4,
            'num_commodities': 5,
            'vocab_size': 50000,
            'hidden_dim': 256,
            'fusion_dim': 512,
            'output_dim': 256,
            'prediction_horizon': 30,
            'dropout': 0.2
        }

        # Déterminer si on utilise les modèles pré-entraînés
        self.use_pretrained = use_pretrained and PRETRAINED_AVAILABLE
        if use_pretrained and not PRETRAINED_AVAILABLE:
            warnings.warn(
                "⚠️ Modèles pré-entraînés demandés mais non disponibles. "
                "Utilisation des encodeurs custom."
            )

        # Configuration des modèles pré-entraînés
        self.pretrained_config = pretrained_models or {
            'camembert': 'camembert-base',
            'chronos': 'amazon/chronos-t5-small'
        }

        # ═══════════════════════════════════════════════════════════════════════
        # ENCODEURS - CHOIX ENTRE PRÉ-ENTRAÎNÉS ET CUSTOM
        # ═══════════════════════════════════════════════════════════════════════

        if self.use_pretrained:
            print("🤖 Initialisation avec modèles PRÉ-ENTRAÎNÉS")
            self._init_pretrained_encoders()
        else:
            print("🔧 Initialisation avec encodeurs CUSTOM")
            self._init_custom_encoders()
        
        # ═══════════════════════════════════════════════════════════════════════
        # FUSION CROSS-MODALE (Custom - Notre Innovation)
        # ═══════════════════════════════════════════════════════════════════════

        # Dimensions des modalités (ajustées selon le mode)
        if self.use_pretrained:
            modality_dims = {
                'satellite': self.config['hidden_dim'],      # 256 (encodeur custom gardé)
                'mobile': self.config['hidden_dim'] // 2,    # 128 (encodeur custom gardé)
                'sentiment': self.config['hidden_dim'],      # 256 (CamemBERT projeté)
                'commodity': self.config['hidden_dim'] // 2  # 128 (Chronos projeté)
            }
        else:
            modality_dims = {
                'satellite': self.config['hidden_dim'],
                'mobile': self.config['hidden_dim'] // 2,
                'sentiment': self.config['hidden_dim'],
                'commodity': self.config['hidden_dim'] // 2
            }

        self.fusion = CrossModalFusion(
            modality_dims=modality_dims,
            fusion_dim=self.config['fusion_dim'],
            dropout=self.config['dropout']
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # GNN RÉGIONAL (Custom - Notre Originalité)
        # ═══════════════════════════════════════════════════════════════════════

        self.regional_gnn = RegionalGNN(
            input_dim=self.config['fusion_dim'],
            output_dim=self.config['output_dim'],
            dropout=self.config['dropout']
        )

        # Log du mode utilisé
        self._log_architecture()
        
        # ═══════════════════════════════════════════════════════════════════════
        # TÊTES DE PRÉDICTION
        # ═══════════════════════════════════════════════════════════════════════
        
        # Prédiction BRVM (30 jours)
        self.brvm_head = nn.Sequential(
            nn.Linear(self.config['output_dim'], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(128, self.config['prediction_horizon'])
        )
        
        # Volatilité
        self.volatility_head = nn.Sequential(
            nn.Linear(self.config['output_dim'], 64),
            nn.GELU(),
            nn.Linear(64, self.config['prediction_horizon']),
            nn.Softplus()
        )
        
        # Tendance (hausse/stable/baisse)
        self.trend_head = nn.Sequential(
            nn.Linear(self.config['output_dim'], 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )
        
        # Confiance
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config['output_dim'], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _init_pretrained_encoders(self):
        """Initialise les encodeurs avec modèles pré-entraînés."""

        # 🛰️ Satellite - Garde l'encodeur custom (pas de modèle pré-entraîné adapté)
        self.satellite_encoder = SatelliteEncoder(
            input_dim=self.config['satellite_dim'],
            output_dim=self.config['hidden_dim'],
            dropout=self.config['dropout']
        )
        print("   🛰️ Satellite: Encodeur custom (TCN + Attention)")

        # 📱 Mobile - Garde l'encodeur custom (données spécifiques)
        self.mobile_encoder = MobileEconomyEncoder(
            input_dim=self.config['mobile_dim'],
            output_dim=self.config['hidden_dim'] // 2,
            dropout=self.config['dropout']
        )
        print("   📱 Mobile: Encodeur custom (Bi-LSTM)")

        # 🗣️ Sentiment - CamemBERT (pré-entraîné)
        self.sentiment_encoder = CamemBERTEncoder(
            model_name=self.pretrained_config.get('camembert', 'camembert-base'),
            output_dim=self.config['hidden_dim'],
            freeze_base=True,
            pooling_strategy='cls'
        )
        print(f"   🗣️ Sentiment: CamemBERT ({self.pretrained_config.get('camembert', 'camembert-base')})")

        # 🌾 Commodités - Chronos (pré-entraîné)
        self.commodity_encoder = CommodityChronosEncoder(
            model_name=self.pretrained_config.get('chronos', 'amazon/chronos-t5-small'),
            output_dim=self.config['hidden_dim'] // 2,
            num_commodities=self.config['num_commodities']
        )
        print(f"   🌾 Commodités: Chronos ({self.pretrained_config.get('chronos', 'amazon/chronos-t5-small')})")

    def _init_custom_encoders(self):
        """Initialise les encodeurs custom (mode legacy)."""

        self.satellite_encoder = SatelliteEncoder(
            input_dim=self.config['satellite_dim'],
            output_dim=self.config['hidden_dim'],
            dropout=self.config['dropout']
        )
        print("   🛰️ Satellite: Encodeur custom")

        self.mobile_encoder = MobileEconomyEncoder(
            input_dim=self.config['mobile_dim'],
            output_dim=self.config['hidden_dim'] // 2,
            dropout=self.config['dropout']
        )
        print("   📱 Mobile: Encodeur custom")

        self.sentiment_encoder = SentimentEncoder(
            vocab_size=self.config['vocab_size'],
            output_dim=self.config['hidden_dim'],
            dropout=self.config['dropout']
        )
        print("   🗣️ Sentiment: Encodeur Transformer custom")

        self.commodity_encoder = CommodityEncoder(
            num_commodities=self.config['num_commodities'],
            output_dim=self.config['hidden_dim'] // 2,
            dropout=self.config['dropout']
        )
        print("   🌾 Commodités: Encodeur TCN custom")

    def _log_architecture(self):
        """Affiche un résumé de l'architecture."""
        print("\n" + "="*60)
        print("   📊 RÉSUMÉ ARCHITECTURE SAHEL.AI v2.0")
        print("="*60)
        mode = "PRÉ-ENTRAÎNÉ" if self.use_pretrained else "CUSTOM"
        print(f"   Mode: {mode}")
        print(f"   Paramètres: {self.get_num_parameters():,}")

        if self.use_pretrained:
            print("\n   Modèles pré-entraînés:")
            if TRANSFORMERS_AVAILABLE:
                print(f"      • CamemBERT: ✅ Chargé")
            else:
                print(f"      • CamemBERT: ⚠️ Fallback")
            if CHRONOS_AVAILABLE:
                print(f"      • Chronos: ✅ Chargé")
            else:
                print(f"      • Chronos: ⚠️ Fallback")

        print("\n   Modules custom (innovation):")
        print("      • Cross-Modal Fusion (Gated Attention)")
        print("      • Regional GNN (8 pays UEMOA)")
        print("="*60 + "\n")

    def get_encoder_mode(self) -> str:
        """Retourne le mode d'encodeur utilisé."""
        return "pretrained" if self.use_pretrained else "custom"

    def get_model_info(self) -> Dict:
        """Retourne les informations détaillées sur le modèle."""
        return {
            'version': '2.0',
            'encoder_mode': self.get_encoder_mode(),
            'num_parameters': self.get_num_parameters(),
            'pretrained_models': {
                'camembert': {
                    'available': TRANSFORMERS_AVAILABLE,
                    'model': self.pretrained_config.get('camembert', 'camembert-base') if self.use_pretrained else None
                },
                'chronos': {
                    'available': CHRONOS_AVAILABLE,
                    'model': self.pretrained_config.get('chronos', 'amazon/chronos-t5-small') if self.use_pretrained else None
                }
            },
            'custom_modules': [
                'CrossModalFusion (Gated Attention)',
                'RegionalGNN (8 UEMOA countries)'
            ],
            'prediction_heads': [
                'BRVM Index (30 days)',
                'Volatility',
                'Trend (up/stable/down)',
                'Confidence'
            ]
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass complet

        Args:
            batch: Dict avec satellite_data, mobile_data, text_ids,
                   text_mask, commodity_prices, commodity_ids
                   (+ raw_texts optionnel pour CamemBERT direct)
        Returns:
            Dict avec brvm_prediction, volatility, trend_logits, confidence
        """

        # 1. Encode chaque modalité
        # 🛰️ Satellite (toujours custom)
        satellite_repr = self.satellite_encoder(batch['satellite_data'])

        # 📱 Mobile (toujours custom)
        mobile_repr = self.mobile_encoder(batch['mobile_data'])

        # 🗣️ Sentiment (CamemBERT ou custom)
        if self.use_pretrained and hasattr(self.sentiment_encoder, 'is_pretrained'):
            # Mode CamemBERT
            sentiment_repr = self.sentiment_encoder(
                batch['text_ids'],
                batch.get('text_mask'),
                batch.get('language', 'fr')
            )
        else:
            # Mode custom
            sentiment_repr = self.sentiment_encoder(
                batch['text_ids'],
                batch.get('text_mask'),
                batch.get('language', 'fr')
            )

        # 🌾 Commodités (Chronos ou custom)
        if self.use_pretrained and hasattr(self.commodity_encoder, 'chronos_encoder'):
            # Mode Chronos
            commodity_repr = self.commodity_encoder(
                batch['commodity_prices'],
                batch.get('commodity_ids')
            )
        else:
            # Mode custom
            commodity_repr = self.commodity_encoder(
                batch['commodity_prices'],
                batch['commodity_ids']
            )
        
        # 2. Fusion cross-modale
        fused = self.fusion({
            'satellite': satellite_repr,
            'mobile': mobile_repr,
            'sentiment': sentiment_repr,
            'commodity': commodity_repr
        })
        
        # 3. GNN régional
        regional_repr = self.regional_gnn(fused)
        
        # 4. Prédictions
        return {
            'brvm_prediction': self.brvm_head(regional_repr),
            'volatility': self.volatility_head(regional_repr),
            'trend_logits': self.trend_head(regional_repr),
            'confidence': self.confidence_head(regional_repr)
        }
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Prédiction en mode évaluation"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            return {
                'brvm_forecast': outputs['brvm_prediction'].cpu().numpy(),
                'volatility': outputs['volatility'].cpu().numpy(),
                'trend_probs': F.softmax(outputs['trend_logits'], dim=-1).cpu().numpy(),
                'confidence': outputs['confidence'].cpu().numpy()
            }
    
    def get_num_parameters(self) -> int:
        """Retourne le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_model(use_pretrained: bool = True):
    """
    Test du modèle avec données synthétiques.

    Args:
        use_pretrained: Tester avec modèles pré-entraînés ou custom
    """
    mode = "PRÉ-ENTRAÎNÉ" if use_pretrained else "CUSTOM"
    print(f"\n🧪 Test SAHEL.AI Model - Mode {mode}")
    print("=" * 60)

    model = SAHELAI(use_pretrained=use_pretrained)
    print(f"✅ Modèle créé: {model.get_num_parameters():,} paramètres")

    # Données test
    batch_size = 4
    seq_len = 60

    batch = {
        'satellite_data': torch.randn(batch_size, seq_len, 5),
        'mobile_data': torch.randn(batch_size, seq_len, 4),
        'text_ids': torch.randint(0, 50000, (batch_size, 128)),
        'text_mask': torch.ones(batch_size, 128),
        'commodity_prices': torch.randn(batch_size, seq_len, 5).abs(),
        'commodity_ids': torch.arange(5),
        'language': 'fr'
    }

    print("\n📥 Forward pass...")
    outputs = model(batch)

    print("\n📊 Outputs:")
    for key, value in outputs.items():
        print(f"   {key}: {value.shape}")

    # Test predict
    print("\n📈 Test prediction...")
    predictions = model.predict(batch)
    print(f"   BRVM forecast shape: {predictions['brvm_forecast'].shape}")
    print(f"   Trend probs shape: {predictions['trend_probs'].shape}")

    # Infos modèle
    info = model.get_model_info()
    print(f"\n📋 Model Info:")
    print(f"   Version: {info['version']}")
    print(f"   Mode: {info['encoder_mode']}")

    print("\n✅ Test réussi!")
    return model


def test_both_modes():
    """Test les deux modes du modèle."""
    print("\n" + "="*70)
    print("   🔬 TEST COMPLET SAHEL.AI v2.0")
    print("="*70)

    # Test mode custom
    print("\n[1/2] Test mode CUSTOM...")
    model_custom = test_model(use_pretrained=False)

    # Test mode pré-entraîné
    print("\n[2/2] Test mode PRÉ-ENTRAÎNÉ...")
    model_pretrained = test_model(use_pretrained=True)

    # Comparaison
    print("\n" + "="*70)
    print("   📊 COMPARAISON DES MODES")
    print("="*70)
    print(f"   Custom:       {model_custom.get_num_parameters():>12,} paramètres")
    print(f"   Pré-entraîné: {model_pretrained.get_num_parameters():>12,} paramètres")
    print("="*70 + "\n")

    return model_custom, model_pretrained


if __name__ == "__main__":
    test_both_modes()
