# pytorch-llm-fundamentals
Tutorielles sur les fondement des LLM avec example et NN entrainés.

## Lancer sur VALERIA

Une première fois seulement, pour télécharger les données, rouler les commandes suivantes:
```
module load StdEnv/2020 python/3.10.2 gcc/9.3.0 arrow/13.0.0
pip install -r valeria_requirements.txt
python prepare_data.py
```

Pour lancer la job d'entrainement roulers la commande suivante:
```
sbatch train.slurm
```

Et pour monitorer le progrès effectuer la commande suivante:
```
tail -f slurm-******.out
```

En remplaçant les ****** par le nouveau fichier de log qui vient d'être créé.
