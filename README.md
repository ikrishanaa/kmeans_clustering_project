# K-Means Clustering Project

## Overview
This project demonstrates **unsupervised learning** using **K-Means Clustering**.  
It includes preprocessing, cluster formation, elbow method, silhouette analysis, and 2D visualization.

By default, it runs on the **Mall Customers dataset** (`Mall_Customers.csv`) for customer segmentation.

## Features
- Load dataset from CSV (`--csv`) or use the default Mall Customers dataset.
- Preprocessing: drop ID columns, handle missing values, scale features.
- Automatic selection of feature columns (Age, Annual Income, Spending Score).
- **K-Means clustering** with customizable K (`--clusters`).
- **Elbow Method** for optimal K visualization.
- **Silhouette Score** for cluster quality evaluation.
- 2D scatter plots of clusters (using PCA if features > 2).
- Saves metrics and plots automatically.

## Folder Structure
```
kmeans_clustering_project/
│
├── data/                  # Place your dataset CSV here
│   └── Mall_Customers.csv
├── outputs/               # Generated plots & metrics
├── src/                   # Python code
│   └── kmeans_model.py
├── README.md
└── requirements.txt
```

## Installation
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### 1) Default dataset (Mall Customers)
```bash
python src/kmeans_model.py
```

### 2) Custom dataset
```bash
python src/kmeans_model.py --csv data/your_dataset.csv
```

### 3) Specify number of clusters
```bash
python src/kmeans_model.py --csv data/your_dataset.csv --clusters 5
```

## Outputs
- `outputs/elbow_method.png` – elbow curve to choose optimal K
- `outputs/clusters.png` – cluster visualization (2D with PCA if needed)
- `outputs/silhouette.png` – silhouette visualization
- `outputs/metrics.json` – cluster metrics

## Notes
- By default, clustering uses `Age`, `Annual Income (k$)`, and `Spending Score (1-100)` if present.
- If your dataset has more than 2 features, PCA reduces it to 2D for visualization.
- Missing values are imputed with mean."# kmeans_clustering_project" 
