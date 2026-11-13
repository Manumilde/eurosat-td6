# TP3 - TD6: Clasificación de uso del suelo con CNNs (EuroSAT RGB)

```
tp3_eurosat/
├── notebooks/TP3_eurosat_cnn.ipynb   # análisis completo (importa src/)
├── src/eurosat_cnn/
│   ├── config.py      # rutas, clases, hiperparámetros globales
│   ├── data.py        # carga/splits, Dataset, data augmentation, DataLoaders
│   ├── models.py      # CNN custom + ResNet18 (transfer learning)
│   ├── training.py    # loop de entrenamiento/validación con tiempos por época
│   ├── gradients.py   # monitor de flujo de gradientes (vanishing/exploding)
│   ├── evaluate.py    # métricas, matriz de confusión, errores
│   ├── latent.py      # extracción de features + PCA / t-SNE / UMAP
│   └── plotting.py    # curvas y comparaciones
├── data/raw/          # acá va el dataset descomprimido
├── outputs/           # figuras, historiales (.csv) y pesos (.pt) generados
└── requirements.txt
```

## Cómo correrlo

### Opción A: Google Colab (recomendado, con GPU)
1. *Entorno de ejecución → Cambiar tipo → GPU (T4)*.
2. Subir la carpeta `tp3_eurosat` (o el zip y `!unzip`), y `%cd tp3_eurosat`.
3. Descargar el dataset:
   ```bash
   !pip install -q umap-learn kaggle
   # subir kaggle.json (Kaggle → Settings → Create New API Token)
   !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d apollo2506/eurosat-dataset -p data/raw --unzip
   ```
4. Abrir `notebooks/TP3_eurosat_cnn.ipynb` y *Ejecutar todo*.

### Opción B: Notebook de Kaggle
Agregar el dataset `apollo2506/eurosat-dataset` con *Add Input*, activar GPU,
subir `src/` y en la celda de setup descomentar
`os.environ["EUROSAT_DIR"] = "/kaggle/input/eurosat-dataset"`.

### Opción C: local
```bash
pip install -r requirements.txt
kaggle datasets download -d apollo2506/eurosat-dataset -p data/raw --unzip
jupyter lab notebooks/TP3_eurosat_cnn.ipynb
```

El loader encuentra solo la carpeta `EuroSAT/` (RGB) y usa los CSVs de
partición del dataset si existen; si no, hace un split estratificado 70/15/15.

**Tiempos orientativos con GPU T4:** CNN custom 30 épocas ≈ 10-15 min;
cada ResNet18 10 épocas a 64×64 ≈ 5-8 min. En CPU es mucho más lento.

**Prueba rápida del pipeline** (pocos datos y épocas, sin pesos preentrenados):
`EUROSAT_SMOKE=1` como variable de entorno antes de abrir el notebook.
