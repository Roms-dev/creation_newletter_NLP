#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparaison des performances temporelles entre spaCy et NLTK

Ce script compare les temps d'exécution pour des tâches NLP communes :
- Tokenisation
- Étiquetage morpho-syntaxique (POS tagging)
- Lemmatisation
- Reconnaissance d'entités nommées (NER)

Usage:
    python compare_timing.py
"""

import time
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
import spacy
import numpy as np

def download_nltk_data():
    """Télécharger les données nécessaires pour NLTK"""
    print("Téléchargement des données NLTK...")
    try:
        # Pour NLTK moderne
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('maxent_ne_chunker_tab', quiet=True)
        nltk.download('words', quiet=True)
        print("Données NLTK téléchargées avec succès.")
    except Exception as e:
        print(f"Erreur lors du téléchargement NLTK: {e}")
        # Fallback pour versions plus anciennes
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
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        print("Modèle spaCy chargé avec succès.")
        return nlp

def prepare_sample_text():
    """Préparer un texte d'exemple pour les tests"""
    text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    The company designs, manufactures, and markets consumer electronics, computer software, and online services.
    Its best-known products include the iPhone, iPad, Mac computers, and Apple Watch.
    Apple has been one of the world's most valuable companies and is known for its innovation in technology.
    """
    return text * 10  # Répéter pour avoir plus de texte

def time_function(func, *args, **kwargs):
    """Mesurer le temps d'exécution d'une fonction"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def nltk_tokenization(text):
    """Tokenisation avec NLTK"""
    return word_tokenize(text)

def spacy_tokenization(text, nlp):
    """Tokenisation avec spaCy"""
    doc = nlp(text)
    return [token.text for token in doc]

def nltk_pos_tagging(text):
    """POS tagging avec NLTK"""
    tokens = word_tokenize(text)
    return pos_tag(tokens)

def spacy_pos_tagging(text, nlp):
    """POS tagging avec spaCy"""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def nltk_lemmatization(text):
    """Lemmatisation avec NLTK"""
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

def spacy_lemmatization(text, nlp):
    """Lemmatisation avec spaCy"""
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def nltk_ner(text):
    """NER avec NLTK"""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return ne_chunk(pos_tags)

def spacy_ner(text, nlp):
    """NER avec spaCy"""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def run_comparison():
    """Exécuter la comparaison"""
    print("=" * 60)
    print("COMPARAISON DES PERFORMANCES : spaCy vs NLTK")
    print("=" * 60)

    # Préparation
    download_nltk_data()
    nlp = load_spacy_model()
    text = prepare_sample_text()

    print(f"\nTexte d'exemple : {len(text)} caractères")
    print(f"Nombre de mots approximatif : {len(text.split())}")

    # Nombre d'itérations pour la moyenne
    num_iterations = 5
    print(f"\nNombre d'itérations pour la moyenne : {num_iterations}")

    tasks = [
        ("Tokenisation", nltk_tokenization, spacy_tokenization),
        ("POS Tagging", nltk_pos_tagging, spacy_pos_tagging),
        ("Lemmatisation", nltk_lemmatization, spacy_lemmatization),
        ("NER", nltk_ner, spacy_ner)
    ]

    results = {}

    for task_name, nltk_func, spacy_func in tasks:
        print(f"\n--- {task_name} ---")

        # Test NLTK
        nltk_times = []
        for i in range(num_iterations):
            if task_name == "NER":
                # Pour NER, on ne mesure que le temps
                _, time_taken = time_function(nltk_func, text)
            else:
                _, time_taken = time_function(nltk_func, text)
            nltk_times.append(time_taken)

        nltk_avg = np.mean(nltk_times)
        nltk_std = np.std(nltk_times)

        # Test spaCy
        spacy_times = []
        for i in range(num_iterations):
            _, time_taken = time_function(spacy_func, text, nlp)
            spacy_times.append(time_taken)

        spacy_avg = np.mean(spacy_times)
        spacy_std = np.std(spacy_times)

        # Résultats
        speedup = nltk_avg / spacy_avg if spacy_avg > 0 else float('inf')

        print(".4f")
        print(".4f")
        print(".2f")

        results[task_name] = {
            'NLTK': {'avg': nltk_avg, 'std': nltk_std},
            'spaCy': {'avg': spacy_avg, 'std': spacy_std},
            'speedup': speedup
        }

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)

    for task, data in results.items():
        winner = "spaCy" if data['spaCy']['avg'] < data['NLTK']['avg'] else "NLTK"
        print(f"{task}: {winner} est plus rapide (x{data['speedup']:.2f})")

    print("\nNote: spaCy est généralement plus rapide grâce à son implémentation Cython optimisée,")
    print("mais NLTK offre plus de flexibilité et de contrôle.")

if __name__ == "__main__":
    import sys
    run_comparison()