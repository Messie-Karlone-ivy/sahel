"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    📊 SAHEL.AI - GÉNÉRATEUR DE DONNÉES                        ║
║                                                                               ║
║   Génère des données synthétiques réalistes pour:                            ║
║   - Démonstration du modèle                                                  ║
║   - Tests et validation                                                      ║
║   - Présentation au concours                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 🌍 DONNÉES PAYS UEMOA
# ═══════════════════════════════════════════════════════════════════════════════

COUNTRY_INFO = {
    "CIV": {"name": "Côte d'Ivoire", "flag": "🇨🇮", "gdp_weight": 0.40, "nightlight_base": 85, "mobile_base": 5000},
    "SEN": {"name": "Sénégal", "flag": "🇸🇳", "gdp_weight": 0.15, "nightlight_base": 70, "mobile_base": 3500},
    "MLI": {"name": "Mali", "flag": "🇲🇱", "gdp_weight": 0.12, "nightlight_base": 45, "mobile_base": 2500},
    "BFA": {"name": "Burkina Faso", "flag": "🇧🇫", "gdp_weight": 0.10, "nightlight_base": 40, "mobile_base": 2000},
    "BEN": {"name": "Bénin", "flag": "🇧🇯", "gdp_weight": 0.08, "nightlight_base": 50, "mobile_base": 1500},
    "NER": {"name": "Niger", "flag": "🇳🇪", "gdp_weight": 0.08, "nightlight_base": 30, "mobile_base": 1200},
    "TGO": {"name": "Togo", "flag": "🇹🇬", "gdp_weight": 0.05, "nightlight_base": 55, "mobile_base": 1000},
    "GNB": {"name": "Guinée-Bissau", "flag": "🇬🇼", "gdp_weight": 0.02, "nightlight_base": 20, "mobile_base": 300},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 📈 GÉNÉRATEURS DE SÉRIES TEMPORELLES
# ═══════════════════════════════════════════════════════════════════════════════

class TimeSeriesGenerator:
    """Génère des séries temporelles réalistes"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def brownian_motion(self, n: int, mu: float = 0, sigma: float = 1, 
                        initial: float = 100) -> np.ndarray:
        """Mouvement brownien géométrique"""
        dt = 1/252
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
        prices = initial * np.exp(np.cumsum(returns))
        return prices
    
    def mean_reverting(self, n: int, mean: float, theta: float = 0.1,
                       sigma: float = 1) -> np.ndarray:
        """Processus d'Ornstein-Uhlenbeck"""
        dt = 1/252
        values = np.zeros(n)
        values[0] = mean
        
        for t in range(1, n):
            dW = np.random.normal(0, np.sqrt(dt))
            values[t] = values[t-1] + theta * (mean - values[t-1]) * dt + sigma * dW
        
        return values
    
    def add_seasonality(self, data: np.ndarray, period: int = 365,
                        amplitude: float = 0.1) -> np.ndarray:
        """Ajoute une composante saisonnière"""
        n = len(data)
        seasonal = amplitude * np.sin(2 * np.pi * np.arange(n) / period)
        return data * (1 + seasonal)
    
    def add_trend(self, data: np.ndarray, growth_rate: float = 0.05) -> np.ndarray:
        """Ajoute une tendance linéaire"""
        n = len(data)
        trend = np.linspace(1, 1 + growth_rate, n)
        return data * trend


# ═══════════════════════════════════════════════════════════════════════════════
# 🌍 GÉNÉRATEUR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class SAHELDataGenerator:
    """
    Générateur de données SAHEL.AI
    Crée un dataset cohérent et réaliste
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ts_gen = TimeSeriesGenerator(seed)
        np.random.seed(seed)
    
    def generate_brvm_data(self, n_days: int = 365) -> pd.DataFrame:
        """Génère les données BRVM"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Index avec tendance, saisonnalité et bruit
        base_index = self.ts_gen.brownian_motion(n_days, mu=0.0003, sigma=0.012, initial=200)
        base_index = self.ts_gen.add_seasonality(base_index, amplitude=0.05)
        
        # Calculer les métriques
        returns = np.diff(np.log(base_index), prepend=0)
        volatility = pd.Series(returns).rolling(20).std().fillna(0.01).values * np.sqrt(252) * 100
        
        # Volume
        base_volume = 80_000_000
        volume = self.ts_gen.mean_reverting(n_days, mean=base_volume, theta=0.2, sigma=base_volume * 0.3)
        volume = np.abs(volume) * (1 + np.abs(returns) * 5)
        
        df = pd.DataFrame({
            'date': dates,
            'brvm_composite': base_index,
            'brvm_10': base_index * 1.15 + np.random.randn(n_days) * 2,
            'daily_return': returns * 100,
            'volatility_20d': volatility,
            'volume': volume,
            'market_cap': base_index * 1e9
        })
        
        # Tendance classification
        rolling_return = pd.Series(returns).rolling(5).mean()
        df['trend'] = pd.cut(rolling_return, bins=[-np.inf, -0.002, 0.002, np.inf],
                             labels=['Baisse', 'Stable', 'Hausse'])
        df['trend'] = df['trend'].fillna('Stable')
        
        return df
    
    def generate_commodity_data(self, n_days: int = 365) -> pd.DataFrame:
        """Génère les prix des matières premières"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Prix de base et volatilité
        commodities = {
            'cacao': {'base': 2500, 'vol': 0.025, 'unit': 'USD/t'},
            'cafe': {'base': 1.50, 'vol': 0.02, 'unit': 'USD/lb'},
            'coton': {'base': 0.80, 'vol': 0.015, 'unit': 'USD/lb'},
            'or': {'base': 1900, 'vol': 0.01, 'unit': 'USD/oz'},
            'petrole': {'base': 75, 'vol': 0.025, 'unit': 'USD/bbl'}
        }
        
        data = {'date': dates}
        
        for name, params in commodities.items():
            prices = self.ts_gen.brownian_motion(
                n_days, mu=0.0001, sigma=params['vol'], initial=params['base']
            )
            if name in ['cacao', 'coton']:
                prices = self.ts_gen.add_seasonality(prices, amplitude=0.08)
            data[name] = prices
        
        return pd.DataFrame(data)
    
    def generate_satellite_data(self, n_days: int = 365) -> Dict[str, pd.DataFrame]:
        """Génère les données satellites par pays"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        data = {}
        
        for code, info in COUNTRY_INFO.items():
            # Lumières nocturnes
            nightlight = self.ts_gen.mean_reverting(
                n_days, mean=info['nightlight_base'], theta=0.1, sigma=2
            )
            nightlight = self.ts_gen.add_trend(nightlight, growth_rate=0.03)
            nightlight = self.ts_gen.add_seasonality(nightlight, amplitude=0.1)
            
            # NDVI (végétation)
            ndvi_base = 0.35 if code in ['NER', 'MLI', 'BFA'] else 0.50
            ndvi = self.ts_gen.mean_reverting(n_days, mean=ndvi_base, theta=0.05, sigma=0.05)
            ndvi = ndvi + 0.15 * np.sin(2 * np.pi * np.arange(n_days) / 365 - np.pi/2)
            ndvi = np.clip(ndvi, 0.1, 0.85)
            
            # Activité portuaire (pays côtiers)
            if code in ['CIV', 'SEN', 'TGO', 'BEN', 'GNB']:
                port_base = {'CIV': 45, 'SEN': 35, 'TGO': 25, 'BEN': 20, 'GNB': 5}[code]
                port_activity = np.random.poisson(port_base, n_days)
            else:
                port_activity = np.zeros(n_days)
            
            data[code] = pd.DataFrame({
                'date': dates,
                'country': code,
                'nightlight_intensity': np.clip(nightlight, 10, 100),
                'nightlight_change': np.diff(nightlight, prepend=nightlight[0]) / nightlight * 100,
                'ndvi': ndvi,
                'port_ship_count': port_activity,
                'urban_expansion_rate': np.abs(np.random.randn(n_days) * 0.002 + 0.005)
            })
        
        return data
    
    def generate_mobile_data(self, n_days: int = 365) -> Dict[str, pd.DataFrame]:
        """Génère les données de téléphonie mobile"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        data = {}
        
        for code, info in COUNTRY_INFO.items():
            base = info['mobile_base']
            
            # Volume transactions (millions XOF)
            transactions = self.ts_gen.brownian_motion(
                n_days, mu=0.0005, sigma=0.05, initial=base
            )
            
            # Pattern mensuel (pic fin de mois)
            monthly_pattern = np.array([1 + 0.3 * (i % 30 > 25) for i in range(n_days)])
            transactions = transactions * monthly_pattern
            
            # Croissance Mobile Money
            transactions = self.ts_gen.add_trend(transactions, growth_rate=0.15)
            
            # Utilisateurs actifs
            population_factor = {'CIV': 12, 'SEN': 8, 'MLI': 6, 'BFA': 5, 
                               'BEN': 3, 'NER': 2.5, 'TGO': 2, 'GNB': 0.5}
            users = population_factor[code] * 1e6 * np.linspace(0.3, 0.45, n_days)
            users = users + np.random.randn(n_days) * users * 0.02
            
            # Transferts transfrontaliers
            diaspora_factor = {'SEN': 2, 'MLI': 1.8, 'CIV': 1.5, 'BFA': 1.3,
                             'BEN': 1, 'TGO': 1, 'NER': 0.8, 'GNB': 0.5}
            cross_border = transactions * 0.1 * diaspora_factor[code]
            
            data[code] = pd.DataFrame({
                'date': dates,
                'country': code,
                'transaction_volume': np.maximum(transactions, 0),
                'transaction_count': np.maximum(transactions * 0.5, 0),
                'active_users': np.maximum(users, 0),
                'cross_border_transfers': np.maximum(cross_border, 0)
            })
        
        return data
    
    def generate_sentiment_data(self, n_days: int, brvm_data: pd.DataFrame) -> pd.DataFrame:
        """Génère les données de sentiment"""
        dates = brvm_data['date'].values
        returns = brvm_data['daily_return'].values / 100
        
        # Sentiment Twitter (corrélé aux rendements avec lag)
        twitter_base = self.ts_gen.mean_reverting(n_days, mean=0.5, theta=0.3, sigma=0.15)
        for i in range(2, n_days):
            twitter_base[i] += 0.3 * returns[i-1] + 0.2 * returns[i-2]
        twitter_sentiment = np.clip(twitter_base, 0, 1)
        
        # Sentiment News (plus lissé)
        news_sentiment = self.ts_gen.mean_reverting(n_days, mean=0.5, theta=0.1, sigma=0.08)
        news_sentiment = np.clip(news_sentiment, 0, 1)
        
        # Volume de mentions
        mention_volume = self.ts_gen.mean_reverting(n_days, mean=1000, theta=0.2, sigma=300)
        mention_volume = mention_volume * (1 + brvm_data['volatility_20d'].values / 100)
        
        return pd.DataFrame({
            'date': dates,
            'twitter_sentiment': twitter_sentiment,
            'news_sentiment': news_sentiment,
            'combined_sentiment': 0.6 * twitter_sentiment + 0.4 * news_sentiment,
            'mention_volume': np.maximum(mention_volume, 100)
        })
    
    def generate_full_dataset(self, n_days: int = 365) -> Dict:
        """
        Génère le dataset complet
        
        Returns:
            Dict avec toutes les données
        """
        print(f"🌍 Génération de {n_days} jours de données SAHEL.AI...")
        
        # 1. Données de marché BRVM
        print("   📈 Génération données BRVM...")
        brvm_data = self.generate_brvm_data(n_days)
        
        # 2. Commodités
        print("   🌾 Génération données commodités...")
        commodity_data = self.generate_commodity_data(n_days)
        
        # 3. Satellites
        print("   🛰️ Génération données satellites...")
        satellite_data = self.generate_satellite_data(n_days)
        
        # 4. Mobile
        print("   📱 Génération données mobile...")
        mobile_data = self.generate_mobile_data(n_days)
        
        # 5. Sentiment
        print("   🗣️ Génération données sentiment...")
        sentiment_data = self.generate_sentiment_data(n_days, brvm_data)
        
        print("   ✅ Génération terminée!")
        
        return {
            'brvm': brvm_data,
            'commodities': commodity_data,
            'satellite': satellite_data,
            'mobile': mobile_data,
            'sentiment': sentiment_data,
            'metadata': {
                'n_days': n_days,
                'countries': list(COUNTRY_INFO.keys()),
                'generation_date': datetime.now().isoformat(),
                'seed': self.seed
            }
        }
    
    def generate_predictions(self, brvm_data: pd.DataFrame, 
                            horizon: int = 30, n_simulations: int = 1000) -> Dict:
        """
        Génère des prédictions Monte Carlo simulées
        """
        last_value = brvm_data['brvm_composite'].iloc[-1]
        last_vol = brvm_data['volatility_20d'].iloc[-1] / 100
        
        # Monte Carlo
        predictions = []
        for _ in range(n_simulations):
            path = [last_value]
            for _ in range(horizon):
                drift = 0.0002  # Légère tendance haussière
                shock = np.random.randn() * last_vol / np.sqrt(252)
                path.append(path[-1] * (1 + drift + shock))
            predictions.append(path[1:])
        
        predictions = np.array(predictions)
        
        return {
            'dates': pd.date_range(
                start=brvm_data['date'].iloc[-1] + timedelta(days=1),
                periods=horizon
            ),
            'mean': predictions.mean(axis=0),
            'median': np.median(predictions, axis=0),
            'std': predictions.std(axis=0),
            'lower_5': np.percentile(predictions, 5, axis=0),
            'lower_25': np.percentile(predictions, 25, axis=0),
            'upper_75': np.percentile(predictions, 75, axis=0),
            'upper_95': np.percentile(predictions, 95, axis=0),
            'all_paths': predictions[:100]  # Garder 100 simulations pour visualisation
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_generator():
    """Test du générateur"""
    print("=" * 60)
    print("🧪 Test du Générateur de Données SAHEL.AI")
    print("=" * 60)
    
    generator = SAHELDataGenerator(seed=42)
    data = generator.generate_full_dataset(n_days=365)
    
    print("\n📊 Résumé:")
    print(f"   BRVM: {len(data['brvm'])} jours")
    print(f"   Indice: {data['brvm']['brvm_composite'].iloc[-1]:.2f}")
    print(f"   Commodités: {len(data['commodities'])} jours, {len(data['commodities'].columns)-1} produits")
    print(f"   Satellites: {len(data['satellite'])} pays")
    print(f"   Mobile: {len(data['mobile'])} pays")
    print(f"   Sentiment: {len(data['sentiment'])} jours")
    
    # Test prédictions
    predictions = generator.generate_predictions(data['brvm'])
    print(f"\n📈 Prédictions (30j):")
    print(f"   Moyenne finale: {predictions['mean'][-1]:.2f}")
    print(f"   IC 90%: [{predictions['lower_5'][-1]:.2f}, {predictions['upper_95'][-1]:.2f}]")
    
    print("\n✅ Test réussi!")
    return data, predictions


if __name__ == "__main__":
    test_generator()
