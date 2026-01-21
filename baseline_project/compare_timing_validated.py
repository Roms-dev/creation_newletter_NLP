#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparaison des performances temporelles entre spaCy et NLTK - Version avec validation

Ce script compare les temps d'exécution pour des tâches NLP communes avec validation :
- Vérification de la cohérence des résultats
- Statistiques détaillées
- Tests avec différents textes
- Utilisation de timeit pour plus de précision

Usage:
    python compare_timing_validated.py
"""

import time
import timeit
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
import spacy
import numpy as np
from collections import Counter
import statistics

def download_nltk_data():
    """Télécharger les données nécessaires pour NLTK"""
    print("Téléchargement des données NLTK...")
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('maxent_ne_chunker_tab', quiet=True)
        nltk.download('words', quiet=True)
        print("Données NLTK téléchargées avec succès.")
    except Exception as e:
        print(f"Erreur lors du téléchargement NLTK: {e}")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            print("Données NLTK (version ancienne) téléchargées avec succès.")
        except Exception as e2:
            print(f"Erreur lors du téléchargement NLTK (fallback): {e2}")

def load_spacy_model():
    """Charger le modèle spaCy"""
    print("Chargement du modèle spaCy...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Modèle spaCy chargé avec succès.")
        return nlp
    except OSError:
        print("Modèle spaCy 'en_core_web_sm' non trouvé. Installation...")
        import subprocess
        subprocess.run([__import__('sys').executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        print("Modèle spaCy chargé avec succès.")
        return nlp

def prepare_sample_texts():
    """Préparer différents textes d'exemple pour les tests"""
    texts = {
        "court": "Hello world. This is a simple test.",
        "moyen": """
        Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
        It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
        """,
        "long": """
        Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
        It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
        The company designs, manufactures, and markets consumer electronics, computer software, and online services.
        Its best-known products include the iPhone, iPad, Mac computers, and Apple Watch.
        Apple has been one of the world's most valuable companies and is known for its innovation in technology.
        """ * 5
    }
    return texts

def time_function_precise(func, *args, setup="pass", number=100, **kwargs):
    """Mesurer le temps d'exécution avec timeit pour plus de précision"""
    def wrapper():
        return func(*args, **kwargs)

    timer = timeit.Timer(wrapper, setup=setup)
    times = timer.repeat(number=number, repeat=5)
    return times

def validate_results(nltk_result, spacy_result, task_name):
    """Valider que les résultats sont cohérents entre NLTK et spaCy"""
    try:
        if task_name == "Tokenisation":
            # Vérifier que le nombre de tokens est similaire
            nltk_count = len(nltk_result)
            spacy_count = len(spacy_result)
            ratio = min(nltk_count, spacy_count) / max(nltk_count, spacy_count)
            return ratio > 0.8, f"Tokens: NLTK={nltk_count}, spaCy={spacy_count}, ratio={ratio:.2f}"

        elif task_name == "POS Tagging":
            # Vérifier que les tokens correspondent
            nltk_tokens = [token for token, _ in nltk_result]
            spacy_tokens = [token for token, _ in spacy_result]
            if len(nltk_tokens) == len(spacy_tokens):
                matches = sum(1 for a, b in zip(nltk_tokens, spacy_tokens) if a == b)
                ratio = matches / len(nltk_tokens)
                return ratio > 0.9, f"Token alignment: {matches}/{len(nltk_tokens)} ({ratio:.1%})"
            else:
                return False, f"Différent nombre de tokens: NLTK={len(nltk_tokens)}, spaCy={len(spacy_tokens)}"

        elif task_name == "Lemmatisation":
            # Vérifier que les tokens correspondent
            nltk_count = len(nltk_result)
            spacy_count = len(spacy_result)
            if nltk_count == spacy_count:
                return True, f"Lemmes: {nltk_count} tokens"
            else:
                return False, f"Différent nombre de lemmes: NLTK={nltk_count}, spaCy={spacy_count}"

        elif task_name == "NER":
            # Pour NER, on vérifie juste que ça fonctionne
            return True, f"NER exécuté avec succès"

    except Exception as e:
        return False, f"Erreur de validation: {e}"

    return True, "Validation passée"

def run_validated_comparison():
    """Exécuter la comparaison avec validation"""
    print("=" * 70)
    print("COMPARAISON VALIDÉE : spaCy vs NLTK")
    print("=" * 70)

    download_nltk_data()
    nlp = load_spacy_model()
    texts = prepare_sample_texts()

    print(f"\nTextes de test préparés: {list(texts.keys())}")

    tasks = [
        ("Tokenisation", lambda text: word_tokenize(text), lambda text, nlp: [token.text for token in nlp(text)]),
        ("POS Tagging", lambda text: pos_tag(word_tokenize(text)), lambda text, nlp: [(token.text, token.pos_) for token in nlp(text)]),
        ("Lemmatisation", lambda text: lemmatize_nltk(text), lambda text, nlp: [token.lemma_ for token in nlp(text)]),
        ("NER", lambda text: ne_chunk(pos_tag(word_tokenize(text))), lambda text, nlp: [(ent.text, ent.label_) for ent in nlp(text).ents])
    ]

    results = {}

    for text_name, text in texts.items():
        print(f"\n{'='*50}")
        print(f"TEXTE: {text_name.upper()} ({len(text)} caractères)")
        print(f"{'='*50}")

        for task_name, nltk_func, spacy_func in tasks:
            print(f"\n--- {task_name} ---")

            # Exécuter et mesurer NLTK
            nltk_times = time_function_precise(nltk_func, text)
            nltk_result = nltk_func(text)

            # Exécuter et mesurer spaCy
            spacy_times = time_function_precise(spacy_func, text, nlp=nlp)
            spacy_result = spacy_func(text, nlp)

            # Statistiques
            nltk_stats = {
                'mean': statistics.mean(nltk_times),
                'median': statistics.median(nltk_times),
                'stdev': statistics.stdev(nltk_times) if len(nltk_times) > 1 else 0,
                'min': min(nltk_times),
                'max': max(nltk_times)
            }

            spacy_stats = {
                'mean': statistics.mean(spacy_times),
                'median': statistics.median(spacy_times),
                'stdev': statistics.stdev(spacy_times) if len(spacy_times) > 1 else 0,
                'min': min(spacy_times),
                'max': max(spacy_times)
            }

            # Validation
            is_valid, validation_msg = validate_results(nltk_result, spacy_result, task_name)

            # Résultats
            speedup = nltk_stats['mean'] / spacy_stats['mean'] if spacy_stats['mean'] > 0 else float('inf')

            print("NLTK  - Moyenne: .4f s (écart: .4f)")
            print("spaCy - Moyenne: .4f s (écart: .4f)")
            print(f"Rapport: {speedup:.2f}x {'plus rapide' if speedup > 1 else 'plus lent'}")
            print(f"Validation: {'✓' if is_valid else '✗'} {validation_msg}")

            if text_name not in results:
                results[text_name] = {}
            results[text_name][task_name] = {
                'nltk': nltk_stats,
                'spacy': spacy_stats,
                'speedup': speedup,
                'validation': (is_valid, validation_msg)
            }

    # Résumé global
    print(f"\n{'='*70}")
    print("RÉSUMÉ GLOBAL")
    print(f"{'='*70}")

    for text_name in texts.keys():
        print(f"\n{text_name.upper()}:")
        for task_name, data in results[text_name].items():
            winner = "NLTK" if data['speedup'] > 1 else "spaCy"
            valid = "✓" if data['validation'][0] else "✗"
            print(f"  {task_name}: {winner} ({data['speedup']:.2f}x) {valid}")

    print(f"\n{'='*70}")
    print("RECOMMANDATIONS DE VALIDATION:")
    print("- Vérifiez que les résultats sont cohérents entre les bibliothèques")
    print("- Les écarts-types faibles indiquent des mesures stables")
    print("- Testez avec différents textes pour confirmer les tendances")
    print("- Pour plus de précision, utilisez des outils de profiling comme cProfile")
    print(f"{'='*70}")

def lemmatize_nltk(text):
    """Lemmatisation avec NLTK (fonction helper)"""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return 'n'

    lemmas = []
    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        lemmas.append(lemma)
    return lemmas

if __name__ == "__main__":
    run_validated_comparison()