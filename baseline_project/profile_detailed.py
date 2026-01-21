#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profiling détaillé des performances spaCy vs NLTK

Utilise cProfile et pstats pour analyser en détail les performances
"""

import cProfile
import pstats
import io
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import spacy

def profile_function(func, *args, **kwargs):
    """Profiler une fonction avec cProfile"""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Top 10 fonctions
    profile_output = s.getvalue()

    return result, profile_output

def run_detailed_profiling():
    """Exécuter un profiling détaillé"""
    print("=" * 60)
    print("PROFILING DÉTAILLÉ : spaCy vs NLTK")
    print("=" * 60)

    # Télécharger données NLTK
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Charger spaCy
    nlp = spacy.load("en_core_web_sm")

    # Texte de test
    text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    The company designs, manufactures, and markets consumer electronics, computer software, and online services.
    """ * 3

    print(f"Texte de test: {len(text)} caractères")

    # Fonctions à profiler
    def nltk_pipeline(text):
        tokens = word_tokenize(text)
        pos_tags_result = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word, _ in pos_tags_result]
        return tokens, pos_tags_result, lemmas

    def spacy_pipeline(text):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        pos_tags_result = [(token.text, token.pos_) for token in doc]
        lemmas = [token.lemma_ for token in doc]
        return tokens, pos_tags_result, lemmas

    # Profiler NLTK
    print("\n" + "="*40)
    print("PROFILING NLTK")
    print("="*40)
    nltk_result, nltk_profile = profile_function(nltk_pipeline, text)
    print("Résultats NLTK:")
    print(f"Tokens: {len(nltk_result[0])}")
    print(f"POS tags: {len(nltk_result[1])}")
    print(f"Lemmes: {len(nltk_result[2])}")
    print("\nProfile NLTK (top fonctions):")
    print(nltk_profile)

    # Profiler spaCy
    print("\n" + "="*40)
    print("PROFILING SPACY")
    print("="*40)
    spacy_result, spacy_profile = profile_function(spacy_pipeline, text)
    print("Résultats spaCy:")
    print(f"Tokens: {len(spacy_result[0])}")
    print(f"POS tags: {len(spacy_result[1])}")
    print(f"Lemmes: {len(spacy_result[2])}")
    print("\nProfile spaCy (top fonctions):")
    print(spacy_profile)

    # Comparaison temporelle simple
    print("\n" + "="*40)
    print("COMPARAISON TEMPORELLE SIMPLE")
    print("="*40)

    def time_simple(func, text, iterations=10):
        times = []
        for _ in range(iterations):
            start = time.time()
            func(text)
            end = time.time()
            times.append(end - start)
        return sum(times) / len(times)

    nltk_time = time_simple(nltk_pipeline, text)
    spacy_time = time_simple(spacy_pipeline, text)

    print(".4f")
    print(".4f")
    print(".2f")

    print("\n" + "="*60)
    print("ANALYSE DES RÉSULTATS")
    print("="*60)
    print("Points clés à analyser dans les profiles:")
    print("1. Temps passé dans les fonctions principales")
    print("2. Nombre d'appels de fonctions")
    print("3. Temps cumulé vs temps propre")
    print("4. Fonctions les plus coûteuses")
    print("\nPour spaCy:")
    print("- Le chargement du modèle peut être coûteux")
    print("- Le pipeline intégré est optimisé")
    print("- Moins d'appels de fonctions individuelles")
    print("\nPour NLTK:")
    print("- Chaque étape est séparée")
    print("- Plus de flexibilité mais plus d'overhead")
    print("- Téléchargement/initialisation des ressources")

if __name__ == "__main__":
    run_detailed_profiling()