# ai

## Enregistrement d'une démonstration

Lancez le script `import_cv2.py` et choisissez l'option **D** pour "Demo".
Vous pouvez spécifier le nombre d'étapes à capturer. Les actions réalisées
au clavier et à la souris ainsi que les récompenses seront sauvegardées dans
`demo_data.npy`.

Ensuite relancez le script en mode **N** (nouvel apprentissage) ou **C** pour
continuer un entraînement. Si un fichier `demo_data.npy` est présent, il sera
chargé automatiquement pour initialiser la mémoire de l'agent.
