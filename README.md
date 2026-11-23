# Customer Segmentation Using Clustering

This project explores customer segmentation with unsupervised learning. It compares K-means clustering on (1) scaled numerical features and (2) scaled numerical features combined with one-hot encoded categorical features. It also compares the numerical-only K-means results with Ward hierarchical clustering.

The analysis is in [`customer_segmentation_clustering.ipynb`](customer_segmentation_clustering.ipynb). It includes PCA plots, cluster profiles, silhouette scores, inertia across candidate cluster counts, and comparisons with the dataset's `segment_truth` labels.

## Dataset

The dataset, `customer_behavior.csv`, was created for this project. Place it in the same directory as the notebook before running it. The notebook uses these columns:

- `customer_id`: removed before clustering.
- `segment_truth` and `anomaly_truth`: removed from the clustering features; `segment_truth` is later used to compare the discovered clusters with the supplied labels.
- `preferred_channel` and `region`: one-hot encoded for the mixed-feature K-means model.
- Other columns: treated as numerical features and standardized.

The notebook expects the numeric fields `age`, `annual_income_k`, `spend_score`, `visits_per_month`, and `avg_basket_value` when summarizing cluster profiles. Update the notebook paths or column selections if you change the dataset layout.

## Methods

1. Remove ID and truth-label columns from the clustering inputs.
2. Standardize numerical features and one-hot encode `preferred_channel` and `region`.
3. Fit K-means with three clusters (`random_state=42`, `max_iter=500`) on mixed and numerical-only feature sets.
4. Use PCA to visualize each K-means result in two dimensions.
5. Compare silhouette scores and inspect inertia and silhouette scores for candidate K-means cluster counts from 2 to 9.
6. Fit Ward hierarchical clustering to the standardized numerical features and compare its labels with numerical-only K-means and `segment_truth`.

Cluster IDs are arbitrary labels; cluster 0 does not inherently mean a particular customer type. Use the profile summaries to interpret each cluster.

## Requirements

Python 3.10 or newer is recommended. Install the packages from the included requirements file:

```bash
python -m pip install -r requirements.txt
```

## Run the notebook

1. Clone or download this repository.
2. Make sure `customer_behavior.csv` and `customer_segmentation_clustering.ipynb` are in the same folder.
3. Install the requirements.
4. Open the notebook in Jupyter Notebook, JupyterLab, or VS Code and run the cells from top to bottom.

For example, start Jupyter with:

```bash
python -m pip install notebook
python -m notebook
```

## Project files

```text
.
├── customer_behavior.csv
├── customer_segmentation_clustering.ipynb
├── README.md
└── requirements.txt
```

## Notes

- The notebook uses `OneHotEncoder(sparse_output=False)`, supported by scikit-learn 1.2 and newer.
- `segment_truth` is for post-clustering comparison; it is excluded from the clustering inputs.
- Results can depend on feature selection, scaling, encoding, and the chosen number of clusters.
