# Chatbot de Support Client avec GPT-2

## Description

Ce projet consiste à concevoir un **chatbot capable de répondre automatiquement à des questions courantes du support client**, à l’aide d’un modèle de langage pré-entraîné. L’objectif est de simuler un système de réponse automatisée utilisé dans de nombreuses entreprises.

---

## Dataset

Nous avons utilisé un dataset issu de la plateforme **Hugging Face**, structuré autour de dialogues client / support. Il comporte 26 000 lignes réparties en 5 colonnes principales :

- **Flags** : Indique les variations linguistiques (langage informel, erreurs, synonymes…).
- **Instruction** : La question posée par le client.
- **Category** : Thème général de la question.
- **Intent** : Intention ou sujet précis de la demande.
- **Response** : Réponse attendue.

Pour des raisons de ressources, nous avons travaillé sur un **échantillon de 2 600 lignes**, équilibré par catégorie.

---

## Pipeline de développement

### 1. Exploration & Nettoyage
- Analyse des catégories et intents, vérification de la qualité des données.
- Nettoyage classique : suppression des espaces inutiles, mise en minuscule, uniformisation du texte.

### 2. Préparation & Enrichissement
- Remplacement de variables génériques (`{{Account Type}}`) par des exemples concrets.
- Ajout de phrases hors contexte pour améliorer la robustesse du modèle (ex. *“Hi, how are you?”*).

### 3. Modélisation avec GPT-2
- Modèle : **GPT-2**, pour sa capacité à encoder et générer du texte.
- Tokenisation avec le tokenizer natif de GPT-2.
- Optimisation avec **AdamW** et gestion mémoire avec **GradScaler** (FP16).
- Limitation des epochs à cause de contraintes GPU/RAM, mais convergence partiellement atteinte.

### 4. Génération de texte
- Réglage des paramètres :
  - `max_new_tokens=80` : éviter les réponses trop longues.
  - `top_k=50`, `top_p=0.9` : contrôle de la diversité lexicale.
- Vérification post-traitement pour assurer une **réponse complète et ponctuée**.

### 5. Interface
- Conception d’une **interface Streamlit** avec zone de chat dynamique.
- Déploiement local pour démonstration du chatbot.

---

## Limites & Difficultés

- **Jeu de données réduit** pour l'entraînement → impact sur la qualité des réponses.
- **Contraintes techniques** : manque de RAM et de puissance GPU → limitation du nombre d’epochs.
- Malgré ces contraintes, le chatbot reste capable de répondre de manière cohérente dans des scénarios courants.

---

## Technologies utilisées

- **Python** · **GPT-2** · **Transformers (Hugging Face)**  
- **Streamlit** (interface) · **PyTorch** · **GradScaler** · **AdamW**  
- Dataset : Hugging Face CSV format

---

## À venir

- Entraînement sur un dataset complet
- Amélioration du prétraitement linguistique
