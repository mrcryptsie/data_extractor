# Extraction de Texte d'une Image: (OS: Windows)

Ce projet utilise Streamlit, OpenCV, Pytesseract, et Pandas pour extraire le texte d'une image contenant un tableau et afficher le résultat sous forme de DataFrame.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances suivantes :

- Python 3.6 ou supérieur
- Streamlit
- OpenCV
- Pytesseract
- Pandas
- Tesseract-OCR (doit être installé sur votre système)

Vous pouvez installer les dépendances Python en utilisant pip :

```bash
pip install streamlit opencv-python pytesseract pandas
```

Assurez-vous également que Tesseract-OCR est installé sur votre système. Vous pouvez le télécharger depuis [ce lien](https://github.com/tesseract-ocr/tesseract).

## Installation

1. Clonez ce dépôt sur votre machine locale :

```bash
git clone https://github.com/votre-utilisateur/votre-projet.git
cd votre-projet
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancez l'application Streamlit :

```bash
streamlit run app.py
```

2. Ouvrez votre navigateur et accédez à l'URL indiquée dans le terminal (généralement `http://localhost:8501`).

3. Téléversez une image contenant un tableau via l'interface Streamlit.

4. L'application affichera l'image téléversée et le texte extrait sous forme de DataFrame.

## Fonctionnalités

- **Extraction de Texte** : Utilise Pytesseract pour extraire le texte d'une image.
- **Détection de Tableau** : Utilise OpenCV pour détecter et isoler le tableau dans l'image.
- **Affichage de Résultats** : Affiche le texte extrait sous forme de DataFrame dans l'interface Streamlit.

## Structure du Projet

```
ocr-extract-table-from-image-python/origin2/
│
├── main.py          # Fichier principal de l'application Streamlit
├── requirements.txt # Liste des dépendances Python
└── README.md        # Documentation du projet
```

## Contribution

Les contributions sont les bienvenues ! Voici comment vous pouvez contribuer :

1. Fork le projet.
2. Créez une nouvelle branche (`git checkout -b feature/AmazingFeature`).
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`).
4. Push vers la branche (`git push origin feature/AmazingFeature`).
5. Ouvrez une Pull Request.

## Licence

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

## Contact

Votre Nom - titovlucien@gmail.com

Projet Link: [https://github.com/mrcryptsie/extractor](https://github.com/mrcryptsie/extractor)

---
