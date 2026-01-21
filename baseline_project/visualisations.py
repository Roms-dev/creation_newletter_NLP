#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
📊 Module Visualisations - Charts et WordClouds

Ce module fournit des fonctions pour créer des visualisations
statiques des données du projet de veille NLP.

Usage:
    from visualisations import create_all_visualizations
    create_all_visualizations()
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Charge toutes les données du projet"""
    data_dir = Path("data")

    def load_jsonl(filepath):
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except:
                        continue
        except FileNotFoundError:
            return pd.DataFrame()
        return pd.DataFrame(data)

    df_raw = load_jsonl(data_dir / "articles_raw.jsonl")
    df_processed = load_jsonl(data_dir / "articles_processed.jsonl")
    df_classified = load_jsonl(data_dir / "articles_classified.jsonl")

    return df_raw, df_processed, df_classified

def create_topic_distribution(df_classified):
    """Crée la visualisation de distribution des topics"""
    if df_classified.empty or 'topic_prediction' not in df_classified.columns:
        print("❌ Données de topics non disponibles")
        return

    topic_counts = df_classified['topic_prediction'].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart
    bars = ax1.bar(topic_counts.index, topic_counts.values,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_title('Distribution des Niveaux de Difficulté', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Niveau')
    ax1.set_ylabel('Nombre d\'Articles')
    ax1.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax2.pie(topic_counts.values, labels=topic_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, shadow=True)
    ax2.set_title('Répartition en Pourcentage', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/topic_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Distribution des topics sauvegardée: output/topic_distribution.png")

def create_sentiment_analysis(df_classified):
    """Crée l'analyse des sentiments"""
    if df_classified.empty or 'sentiment_label' not in df_classified.columns:
        print("❌ Données de sentiments non disponibles")
        return

    sentiment_counts = df_classified['sentiment_label'].value_counts()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values,
                   color=['#4CAF50', '#FFC107', '#F44336'], alpha=0.8)

    plt.title('Analyse des Sentiments dans les Articles', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('Nombre d\'Articles')
    plt.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('output/sentiment_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Analyse des sentiments sauvegardée: output/sentiment_analysis.png")

def create_wordcloud_visualization(df_raw, df_processed):
    """Crée les nuages de mots"""
    # WordCloud des titres
    if not df_raw.empty and 'title' in df_raw.columns:
        titles_text = ' '.join(str(title) for title in df_raw['title'].dropna() if title)

        if titles_text.strip():
            wordcloud = WordCloud(
                width=800, height=400, background_color='white',
                colormap='viridis', max_words=100,
                contour_width=1, contour_color='steelblue'
            ).generate(titles_text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Nuage de Mots - Titres des Articles', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('output/wordcloud_titles.png', dpi=150, bbox_inches='tight')
            plt.close()

            print("✅ WordCloud des titres sauvegardé: output/wordcloud_titles.png")

    # WordCloud des mots-clés traités
    if not df_processed.empty and 'tokens' in df_processed.columns:
        all_tokens = []
        for tokens_list in df_processed['tokens'].dropna():
            if isinstance(tokens_list, list):
                all_tokens.extend(tokens_list)

        if all_tokens:
            tokens_text = ' '.join(all_tokens)

            wordcloud = WordCloud(
                width=800, height=400, background_color='white',
                colormap='plasma', max_words=100,
                contour_width=1, contour_color='steelblue'
            ).generate(tokens_text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Nuage de Mots - Mots-clés Après Preprocessing', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('output/wordcloud_keywords.png', dpi=150, bbox_inches='tight')
            plt.close()

            print("✅ WordCloud des mots-clés sauvegardé: output/wordcloud_keywords.png")

def create_preprocessing_metrics(df_processed):
    """Crée les visualisations des métriques de preprocessing"""
    if df_processed.empty:
        print("❌ Données de preprocessing non disponibles")
        return

    # Distribution des pertes de tokens
    if 'token_loss_pct' in df_processed.columns:
        token_losses = df_processed['token_loss_pct'].dropna()

        plt.figure(figsize=(12, 6))

        # Histogramme
        plt.subplot(1, 2, 1)
        plt.hist(token_losses, bins=20, alpha=0.7, color='#45B7D1', edgecolor='black')
        plt.axvline(token_losses.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne: {token_losses.mean():.1f}%')
        plt.xlabel('Pourcentage de Perte de Tokens')
        plt.ylabel('Nombre d\'Articles')
        plt.title('Distribution des Pertes de Tokens')
        plt.legend()
        plt.grid(alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(token_losses, vert=False)
        plt.xlabel('Pourcentage de Perte de Tokens')
        plt.title('Box Plot des Pertes de Tokens')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('output/preprocessing_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✅ Métriques de preprocessing sauvegardées: output/preprocessing_metrics.png")

def create_source_analysis(df_classified):
    """Crée l'analyse des sources"""
    if df_classified.empty or 'source' not in df_classified.columns:
        print("❌ Données de sources non disponibles")
        return

    source_counts = df_classified['source'].value_counts()

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))

    plt.pie(source_counts.values, labels=source_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90, shadow=True)
    plt.title('Répartition des Sources d\'Articles', fontsize=14, fontweight='bold')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('output/source_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Analyse des sources sauvegardée: output/source_analysis.png")

def create_deduplication_analysis(df_classified):
    """Crée l'analyse de déduplication"""
    if df_classified.empty or 'is_duplicate' not in df_classified.columns:
        print("❌ Données de déduplication non disponibles")
        return

    duplicate_counts = df_classified['is_duplicate'].value_counts()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Articles Uniques', 'Doublons'],
                   [duplicate_counts.get(False, 0), duplicate_counts.get(True, 0)],
                   color=['#4CAF50', '#F44336'], alpha=0.8)

    plt.title('Analyse de Déduplication', fontsize=14, fontweight='bold')
    plt.ylabel('Nombre d\'Articles')
    plt.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('output/deduplication_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Analyse de déduplication sauvegardée: output/deduplication_analysis.png")

def create_summary_report(df_raw, df_processed, df_classified):
    """Crée un rapport de synthèse"""
    print("\n" + "="*60)
    print("📊 RAPPORT DE SYNTHÈSE - VISUALISATIONS")
    print("="*60)

    print(f"\n📈 MÉTRIQUES GÉNÉRALES")
    print(f"   Articles bruts: {len(df_raw)}")
    print(f"   Articles traités: {len(df_processed)}")
    print(f"   Articles classifiés: {len(df_classified)}")

    if not df_classified.empty:
        if 'topic_prediction' in df_classified.columns:
            topic_counts = df_classified['topic_prediction'].value_counts()
            print(f"\n🎯 TOPICS LES PLUS FRÉQUENTS")
            for topic, count in topic_counts.head(3).items():
                pct = (count / len(df_classified)) * 100
                print(f"   {topic}: {count} articles ({pct:.1f}%)")

        if 'sentiment_label' in df_classified.columns:
            sentiment_counts = df_classified['sentiment_label'].value_counts()
            print(f"\n😊 SENTIMENTS PRÉDOMINANTS")
            for sentiment, count in sentiment_counts.items():
                pct = (count / len(df_classified)) * 100
                print(f"   {sentiment}: {count} articles ({pct:.1f}%)")

    if not df_processed.empty and 'token_loss_pct' in df_processed.columns:
        losses = df_processed['token_loss_pct'].dropna()
        print(f"\n🔧 QUALITÉ DU PREPROCESSING")
        print(f"   Perte moyenne: {losses.mean():.1f}%")
        print(f"   Perte médiane: {losses.median():.1f}%")

    print(f"\n📁 VISUALISATIONS CRÉÉES")
    visualizations = [
        "topic_distribution.png",
        "sentiment_analysis.png",
        "wordcloud_titles.png",
        "wordcloud_keywords.png",
        "preprocessing_metrics.png",
        "source_analysis.png",
        "deduplication_analysis.png"
    ]

    for viz in visualizations:
        path = Path("output") / viz
        if path.exists():
            print(f"   ✅ {viz}")
        else:
            print(f"   ❌ {viz} (non créé)")

    print(f"\n💡 PROCHAINES ÉTAPES")
    print(f"   • Ouvrir visualisations.ipynb pour analyses interactives")
    print(f"   • Examiner les visualisations dans le dossier output/")
    print(f"   • Ajuster les paramètres selon les insights")

    print(f"\n" + "="*60)

def create_all_visualizations():
    """Crée toutes les visualisations disponibles"""
    print("🎨 CRÉATION DES VISUALISATIONS...")

    # Créer le dossier output si nécessaire
    Path("output").mkdir(exist_ok=True)

    # Charger les données
    df_raw, df_processed, df_classified = load_data()

    if df_raw.empty and df_processed.empty and df_classified.empty:
        print("❌ Aucune donnée disponible pour les visualisations")
        return

    # Créer toutes les visualisations
    create_topic_distribution(df_classified)
    create_sentiment_analysis(df_classified)
    create_wordcloud_visualization(df_raw, df_processed)
    create_preprocessing_metrics(df_processed)
    create_source_analysis(df_classified)
    create_deduplication_analysis(df_classified)

    # Rapport final
    create_summary_report(df_raw, df_processed, df_classified)

if __name__ == "__main__":
    create_all_visualizations()