# ai

## Enregistrement d'une démonstration

Lancez le script `import_cv2.py` et choisissez l'option **D** pour "Demo".
Vous pouvez spécifier le nombre d'étapes à capturer. Les actions réalisées
au clavier et à la souris ainsi que les récompenses seront sauvegardées dans
`demo_data.npy`.

Ensuite relancez le script en mode **N** (nouvel apprentissage) ou **C** pour
continuer un entraînement. Si un fichier `demo_data.npy` est présent, il sera
chargé automatiquement pour initialiser la mémoire de l'agent.

## Réglage de la sensibilité de la souris

La classe `GameEnvironment` possède deux paramètres optionnels :
`mouse_dx_threshold` et `mouse_dy_threshold`. Ils contrôlent le déplacement
horizontal et vertical minimal (en pixels) de la souris pour que l'agent
considère un mouvement du regard.

Les valeurs par défaut sont respectivement `15` et `10`, ce qui correspond
au comportement précédent. Vous pouvez ajuster ces seuils lors de la
création de l'environnement :

```python
env = GameEnvironment(mouse_dx_threshold=20, mouse_dy_threshold=12)
```

Augmentez la valeur pour réduire la sensibilité ou diminuez‑la pour rendre
l'IA plus réactive aux petits mouvements de souris.
