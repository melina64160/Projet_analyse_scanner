# Projet fil rouge : Analyse du phénotype tumoral visible au scanner
## Données

Les données (51 Go) ne sont pas versionnées. Pour les récupérer :

1. Télécharger le dataset :
   pip install -U "huggingface_hub[cli]"
   hf download linghs007/Longitudinal-CT --repo-type dataset --local-dir ~/data/Longitudinal-CT

2. Décompresser les archives :
   mkdir -p ~/data/Longitudinal-CT/Patient_unzip
   cd ~/data/Longitudinal-CT/patients
   for f in *.zip; do unzip -q -n "$f" -d ../Patient_unzip/; done

3. Créer le raccourci dans le dépôt (depuis la racine du projet) :
   ln -s ~/data/Longitudinal-CT data