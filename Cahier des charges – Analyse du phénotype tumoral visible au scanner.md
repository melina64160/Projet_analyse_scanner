# Cahier des charges – Analyse du phénotype tumoral visible au scanner

Nov 23, 2026 · @Mélina

## 1. Contexte

Le projet fil rouge (4 mois, novembre 2026 à février 2027) porte sur la caractérisation quantitative des tumeurs visibles au scanner (CT) et sur la prédiction de leur évolution. Il est proposé par Jules Dupont, data scientist et doctorant au département d'imagerie médicale de Gustave Roussy (sujet de computer vision).

En oncologie, le scanner sert à suivre la « charge tumorale », c'est-à-dire le volume des lésions, entre deux examens. Aujourd'hui, ce suivi repose surtout sur des mesures manuelles (diamètres, critères de réponse). La radiomique propose d'aller plus loin en extrayant de l'image des descripteurs numériques de forme, d'intensité et de texture, qui décrivent le « phénotype » de chaque lésion.

Les données du projet portent sur le mélanome métastatique, un cancer qui se dissémine dans de nombreux organes (poumon, peau, ganglions, foie, os). Cette diversité en fait un bon terrain pour étudier des lésions très différentes avec une même méthode.

Le sujet original, présenté dans le diaporama de lancement, fixe trois objectifs :

1. Explorer le jeu de données.
2. Prédire l'évolution des lésions entre l'examen initial (t0) et l'examen de suivi (t+1) : diminution, augmentation, etc.
3. Étudier l'explicabilité des modèles (objectif réajustable au cours du projet).

## 2. Redéfinition du sujet

**Question centrale : à partir des seules caractéristiques d'une lésion sur le scanner initial, peut-on prédire si elle va diminuer, rester stable ou augmenter au scanner de suivi ?**

Reformulé simplement, le projet consiste à transformer chaque lésion annotée en un vecteur de descripteurs (radiomiques), puis à entraîner un modèle qui relie ces descripteurs à l'évolution observée de la lésion. L'unité d'analyse est la lésion, pas le patient.

**Objectifs**

1. Explorer et décrire le jeu de données : patients, lésions, organes touchés, volumes, appariement des lésions entre les deux examens.
2. Extraire une représentation quantitative de chaque lésion : radiomique « handcrafted » et, si le temps le permet, radiomique profonde (deep radiomics).
3. Construire une variable cible d'évolution à partir des volumes à t0 et t+1.
4. Entraîner et comparer plusieurs modèles de prédiction, du plus simple au plus complexe.
5. Expliquer les prédictions : quelles caractéristiques d'image pèsent dans la décision.

**Périmètre**

- Inclus : les lésions déjà segmentées manuellement dans le jeu de données, sur deux temps d'examen par patient.
- Exclu : la segmentation automatique des lésions, l'usage clinique réel des modèles, les données hors scanner (biologie, génétique) sauf si elles sont fournies.

**Critères de réussite**

- Un pipeline reproductible, des données brutes jusqu'aux résultats.
- Des performances mesurées sur un jeu de test indépendant et comparées à une référence simple (par exemple prédire la classe majoritaire, ou ne se servir que du volume initial).
- Des résultats interprétables et discutés dans le rapport final.

## 3. Données disponibles

Le jeu de données Longitudinal-CT compte 300 patients atteints de mélanome métastatique, scannés deux fois au cours d'un traitement systémique au CHU de Tübingen (2015-2019), soit 600 examens CT corps entier et 7 182 lésions segmentées à la main. Il est décrit dans [Gatidis et al., Scientific Data, 2026](https://doi.org/10.1038/s41597-026-07466-y) et hébergé sur le [dépôt FDAT de l'université de Tübingen](https://doi.org/10.57754/FDAT.75kj1-64747) ; en attendant que ce site remarche, on utilise le [miroir Hugging Face](https://huggingface.co/datasets/linghs007/Longitudinal-CT).

| Élément | Valeur |
| --- | --- |
| Patients | 300, tous atteints de mélanome métastatique sous traitement systémique ; découpage fourni : 240 train, 30 validation, 30 test |
| Examens | 600 : un examen initial (BL) et un de suivi (FU) par patient ; 9,3 % des examens comportent 2 volumes et 1,2 % en comportent 3 (ex. tête et cou en plus du corps entier) |
| Démographie | 181 hommes (60,3 %), 119 femmes (39,7 %), âge de 19 à 89 ans (moyenne environ 64 ans) |
| Acquisition | Scanners Siemens, phase portale avec produit de contraste iodé, 120 kV, coupes de 3 mm ; résolution native, aucune harmonisation entre scanners |
| Lésions | 4 079 à l'initial, 3 103 au suivi ; médiane de 8 lésions par patient à l'initial (1 à 86) et 6 au suivi (1 à 95) ; aucune taille minimale |
| Organes à l'initial | Poumon 32,1 %, tissus mous et peau 21,3 %, ganglions 20,8 %, foie 14,7 %, autres moins de 5 % chacun |
| Volume des lésions | Moyenne 7,26 mL à l'initial et 8,94 mL au suivi, très dispersé (écart-type supérieur à 60 mL) ; médiane d'environ 0,5 mL (diaporama) |
| Annotation | Segmentation manuelle par un radiologue expert, puis relecture en consensus par deux radiologues ; correspondance des lésions entre BL et FU établie à la main |
| Fichiers | inputsTr/ : scanners BL et FU, masques BL, centres des lésions (JSON), un CSV par patient ; targetsTr/ : masques FU. Format NIfTI (.nii.gz), un identifiant entier par lésion dans les masques |
| Métadonnées par lésion (CSV) | lesion\_id, centres à BL et FU (cog\_bl, cog\_fu), organe (lesion\_type), évolution (UNCHANGED, DISAPPEARING, MERGING, NEWLYAPPEARING), volumes à BL et FU (volume\_bl, volume\_fu) |
| Taille et licence | 51,1 Go sur le miroir ; licence CC BY pour les données |

**Ce que l'article apporte au projet :** l'appariement des lésions et leurs volumes aux deux temps sont déjà fournis dans les CSV, ce qui simplifie la construction de la cible. En revanche, l'article ne donne ni le délai entre les deux examens, ni le traitement exact reçu, ni la réponse clinique : ce sont des limites à garder en tête. Le code de l'article (anonymisation et modèle de segmentation nnU-Net) est disponible sur [GitHub](https://github.com/lab-midas/autoPETCTIV), et le jeu de données a servi au [challenge AutoPET IV 2025](https://autopet-iv.grand-challenge.org/).

## 4. Premières pistes

La démarche envisagée suit cinq étapes, de l'image brute à l'explication des prédictions.

**4.1 Exploration du jeu de données**

- Charger les scanners et les masques NIfTI (bibliothèques nibabel ou SimpleITK) et les visualiser coupe par coupe.
- Décrire la population : nombre de lésions par patient, répartition par organe, distribution des volumes à t0 et t+1.
- Lire les CSV par patient et compter les lésions par type d'évolution fourni (inchangée, disparue, fusionnée, nouvelle), par organe et par volume.

**4.2 Extraction de caractéristiques (radiomique)**

- *Radiomique handcrafted* avec PyRadiomics : descripteurs de forme (volume, sphéricité, surface), de premier ordre (histogramme des intensités en unités Hounsfield : moyenne, minimum, énergie…) et de second ordre ou texture (matrices GLCM, GLRLM, GLSZM). Cette famille est illustrée par El Ayachy et al. (Front. Oncol. 2021).
- *Radiomique profonde* : un auto-encodeur compresse un patch 3D centré sur la lésion en un vecteur latent (les « deep radiomic features », dRFs), approche présentée dans Dupont et al. (Diagn. Interv. Imaging 2026).
- Prétraitement à fixer : rééchantillonnage des voxels à une taille commune, fenêtrage des intensités, normalisation des descripteurs.

**4.3 Définition de la cible**

- Partir des volumes fournis dans les CSV (volume\_bl, volume\_fu) pour calculer la variation relative de volume de chaque lésion : (V\_FU − V\_BL) / V\_BL.
- La découper en classes, par exemple diminution, stabilité, augmentation, avec des seuils à choisir (en s'inspirant des critères RECIST, adaptés au volume).
- Utiliser la classe d'évolution déjà annotée (DISAPPEARING, MERGING…) pour traiter à part les lésions disparues ou fusionnées. Les lésions nouvelles (NEWLYAPPEARING) n'existent pas à t0 et sortent donc de la prédiction par lésion ; elles peuvent servir à une analyse au niveau du patient (progression).
- Variante possible : prédire directement la variation de volume (régression).

**4.4 Modélisation**, par complexité croissante, comme proposé dans le sujet :

1. Modèles de machine learning classiques (scikit-learn) : régression logistique, forêts aléatoires, gradient boosting.
2. Perceptron multicouche (MLP) en PyTorch.
3. Petit Transformer, qui traite l'ensemble des lésions d'un patient comme une séquence pour tenir compte du contexte du patient.

Tous seront comparés à une référence naïve, avec des métriques adaptées à des classes déséquilibrées (AUC, F1 macro, précision équilibrée).

**4.5 Explicabilité**

- Importance des variables (permutation, valeurs SHAP) pour les modèles sur descripteurs handcrafted.
- Cartes de saillance ou d'attention pour les modèles profonds.
- Confronter les variables jugées importantes à l'intuition clinique (taille, organe, hétérogénéité de la lésion).

## 5. Premières problématiques

**Questions scientifiques**

- Les radiomiques d'une lésion au temps t0 contiennent-elles assez d'information pour prédire son évolution, indépendamment du traitement reçu ?
- Les descripteurs profonds (auto-encodeur) apportent-ils plus que les descripteurs handcrafted, ou que le seul volume initial ?
- Faut-il prédire lésion par lésion, ou tenir compte des autres lésions du même patient (une lésion évolue rarement seule) ?
- Comment rendre les prédictions compréhensibles pour un radiologue ?

**Difficultés et risques identifiés**

| Problème | Pourquoi c'est un risque | Piste de réponse |
| --- | --- | --- |
| Volume des données (51 Go) | Téléchargement long, mémoire et calcul limités sur un PC personnel | Travailler d'abord sur un sous-ensemble de patients, extraire les descripteurs une seule fois et les stocker en tableau |
| Fusions et séparations de lésions | Quand plusieurs lésions fusionnent ou qu'une lésion se divise, l'évolution d'une lésion n'est plus définie un pour un | S'appuyer sur la classe d'évolution fournie ; exclure ou regrouper ces cas, et le justifier |
| Lésions disparues ou nouvelles | 4 079 lésions à t0 contre 3 103 au suivi : une partie n'a pas d'équivalent | Traiter les disparues comme une classe à part ; étudier les nouvelles au niveau du patient |
| Définition des classes | Les seuils choisis changent l'équilibre des classes et les résultats | Tester plusieurs seuils, s'appuyer sur RECIST |
| Classes déséquilibrées | Un modèle peut sembler bon en prédisant toujours la classe majoritaire | Métriques équilibrées, pondération des classes |
| Très petites lésions | Aucune taille minimale d'annotation et volume médian d'environ 0,5 mL : peu de voxels, textures peu fiables | Seuil de volume minimal, analyse de sensibilité |
| Résolution des images | Coupes de 3 mm, résolution native variable, aucune harmonisation entre scanners | Rééchantillonner à une taille de voxel commune avant l'extraction |
| Examens en plusieurs volumes | 10,5 % des examens comportent 2 ou 3 volumes (ex. tête et cou séparés) | Associer chaque lésion au bon volume grâce à son masque |
| Fuite de données | Des lésions d'un même patient dans le train et le test gonflent les scores | Séparer par patient, en gardant le découpage 240 / 30 / 30 fourni |
| Facteurs absents | Délai entre examens, traitement exact et réponse clinique ne sont pas fournis, alors qu'ils influencent l'évolution | Le citer comme limite ; demander à l'encadrant si ces informations existent |
| Reproductibilité des radiomiques | Les valeurs varient selon le rééchantillonnage et les paramètres d'extraction | Fixer et documenter la configuration, suivre les recommandations IBSI |
| Beaucoup de variables pour peu d'exemples | Plus de 100 descripteurs par lésion, risque de surapprentissage | Sélection de variables, régularisation, validation croisée |

## 6. Livrables et planning

Le projet compte quatre rendus, du cahier des charges (25 novembre 2026) au rapport final et à la soutenance (26 février 2027). Les phases de travail entre ces dates sont une proposition, à ajuster avec l'encadrant.

&#91;embedded content: planning prévisionnel · 4 rendus, 6 phases de travail\]

| Rendu | Échéance | Contenu attendu |
| --- | --- | --- |
| Cahier des charges | Mercredi 25 novembre 2026 | Redéfinition du sujet, premières pistes, premières problématiques (ce document) |
| Bibliographie | Mi-décembre 2026 | Synthèse des articles sur la radiomique, le suivi longitudinal et la prédiction de l'évolution tumorale |
| Rapport intermédiaire et soutenance | Janvier 2027, date à préciser | Exploration du jeu de données, extraction des descripteurs, premiers modèles |
| Rapport final et soutenance | 26 février 2027 | Rapport complet avec toutes les analyses, modèles comparés et explicabilité |

Le code sera versionné sur le [dépôt GitHub du projet](https://github.com/melina64160/Projet_analyse_scanner) pour que les analyses du rapport final soient reproductibles.

## 7. Ressources et outils

**Stack technique proposée par l'encadrant** (diaporama, diapo 12) : Python avec PyTorch pour les réseaux de neurones, scikit-learn pour le machine learning classique, et uv pour gérer l'environnement et les dépendances. La diapo montre aussi deux autres logos (un serpent vert, probablement Snakemake pour orchestrer le pipeline, et un robot jaune) à identifier avec l'encadrant.

**Bibliothèques complémentaires envisagées** : nibabel ou SimpleITK (lecture des NIfTI), PyRadiomics (descripteurs handcrafted), pandas (tableaux de descripteurs), SHAP (explicabilité), matplotlib (figures).

**Questions ouvertes pour l'encadrant**

- Quelle est la date exacte du rendu intermédiaire de janvier ?
- Faut-il garder le découpage train / validation / test fourni (240 / 30 / 30) ?
- Quels seuils de variation de volume utiliser pour définir les classes d'évolution ?
- Un serveur de calcul ou un GPU est-il disponible pour la radiomique profonde ?

## Sources

- Diaporama de lancement « Analyse du phénotype tumoral visible au scanner », J. Dupont, Gustave Roussy (dossier du projet).
- Gatidis S., Peisen F. et al., [A longitudinal whole-body CT dataset with manually annotated tumor lesions](https://www.nature.com/articles/s41597-026-07466-y), Scientific Data 13, 1139 (2026), lu le 23 novembre 2026.
- Küstner T. et al., [Longitudinal-CT, dépôt FDAT de l'université de Tübingen](https://doi.org/10.57754/FDAT.75kj1-64747) (source officielle des données).
- [Miroir Hugging Face du jeu de données Longitudinal-CT](https://huggingface.co/datasets/linghs007/Longitudinal-CT), consulté le 23 novembre 2026.
- [Code de l'article (lab-midas/autoPETCTIV)](https://github.com/lab-midas/autoPETCTIV) : anonymisation et modèle nnU-Net de référence.
- Dupont J. et al., [article sur la radiomique profonde](https://doi.org/10.1016/j.diii.2026.02.001), Diagnostic and Interventional Imaging (2026). À lire en détail pour la bibliographie.
- El Ayachy R. et al., Frontiers in Oncology (2021), cité dans le diaporama pour la radiomique handcrafted.
- Eisenhauer E. A. et al., New response evaluation criteria in solid tumours: revised RECIST guideline (version 1.1), Eur J Cancer 45(2), 228-247 (2009), cité par l'article du jeu de données.
