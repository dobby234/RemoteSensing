# Image Classification with Google Earth Engine

A Google Colab-based assignment covering supervised classification, accuracy assessment, and unsupervised clustering of satellite imagery using Google Earth Engine (GEE) and `geemap`.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Part 1: Supervised Classification](#part-1-supervised-classification)
  - [Paddy Field Mapping — Philippines](#paddy-field-mapping--philippines)
  - [Land Cover Mapping — Krabi, Thailand](#land-cover-mapping--krabi-thailand)
- [Part 2: Accuracy Assessment](#part-2-accuracy-assessment)
- [Part 3: Unsupervised Classification (Clustering)](#part-3-unsupervised-classification-clustering)
- [Exercises](#exercises)
- [Notes](#notes)

---

## Overview

This notebook demonstrates two major image classification paradigms in remote sensing:

- **Supervised Classification** — training a classifier with labelled ground-truth points and applying it across an image.
- **Unsupervised Classification (Clustering)** — automatically grouping pixels into clusters without labelled training data.

Two study areas are used:

| Study Area | Data Source | Purpose |
|---|---|---|
| Central Luzon, Philippines | Sentinel-1 SAR (GRD) | Binary paddy / non-paddy mapping |
| Krabi, Thailand | Sentinel-2 SR | Multi-class land cover mapping |

---

## Requirements

- Python 3.x
- Google Colab (recommended) or a local Jupyter environment
- A Google Earth Engine account

Python packages:

```
earthengine-api
geemap
pycrs
numpy
matplotlib
pandas
```

Install with:

```bash
pip install earthengine-api geemap pycrs
```

---

## Setup

1. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Authenticate and initialise GEE:

```python
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

---

## Part 1: Supervised Classification

### Paddy Field Mapping — Philippines

Uses a two-band false colour composite (FCC) built from Sentinel-1 VH backscatter at the start and end of the 2019 paddy growing season.

**Workflow:**

1. Export the FCC image (`phip_paddy_fcc.tif`) from GEE.
2. Open it in QGIS and digitise ~10–20 paddy points (`id = 1`) and ~10–20 non-paddy points (`id = 0`) into a shapefile `gt_paddy.shp`.
3. Upload the shapefile to GEE session storage.
4. Sample the FCC image at those points and train a **Minimum Distance** classifier.
5. Apply the classifier across the full image.

```python
my_classifier = ee.Classifier.minimumDistance().train(example_pnts, 'id')
class_result = imageFCC.classify(my_classifier)
```

**How Minimum Distance works:** The algorithm estimates a centroid for each class in feature space. Each new pixel is assigned to the nearest centroid.

---

### Land Cover Mapping — Krabi, Thailand

Uses a cloud-reduced Sentinel-2 median composite (Jan–Apr 2020) with Red (B4), Green (B3), Blue (B2), and NIR (B8) bands.

Four land cover classes:

| ID | Class |
|---|---|
| 1 | Forest |
| 2 | Urban |
| 3 | Water |
| 4 | Agriculture |

Ground-truth training points are loaded from `lc_gt_train.shp`. A **Random Forest** classifier (100 trees) is trained and applied:

```python
my_classifier = ee.Classifier.smileRandomForest(numberOfTrees=100).train(example_pnts, 'id')
KrabiClassImg = KrabiImg.classify(my_classifier)
```

**Why Random Forest?** Random Forest uses an ensemble of decision trees, making it more robust and better suited to multi-class, multi-band classification than Minimum Distance.

---

## Part 2: Accuracy Assessment

Accuracy is evaluated against a separate validation dataset (`lc_gt_val.shp`) that was not used during training.

```python
lc_gt_val = geemap.shp_to_ee('/content/lc_gt_val.shp')
validation_pnts = KrabiImg.sampleRegions(lc_gt_val, scale=30)
validation_class = validation_pnts.classify(my_classifier)

val_accuracy = validation_class.errorMatrix('id', 'classification')
print(val_accuracy.accuracy().getInfo())   # Overall accuracy
print(val_accuracy.kappa().getInfo())      # Kappa coefficient
```

The supervised Random Forest classification achieved an **overall accuracy of ~94%** on the validation set.

In the error matrix, rows correspond to actual (reference) classes and columns to predicted classes.

---

## Part 3: Unsupervised Classification (Clustering)

### Philippines — Paddy Mapping with K-Means

K-Means clustering is applied to the same Sentinel-1 FCC image used in Part 1, without any labelled training data.

```python
imageFCC_sample = imageFCC.sample(numPixels=2500, scale=30)
my_clustering = ee.Clusterer.wekaKMeans(nClusters=5).train(imageFCC_sample)
imageFCC_class = imageFCC.cluster(my_clustering)
```

Paddy areas appear as a distinct cluster (Class 4 in this run) and can be isolated with:

```python
paddy_area = imageFCC_class.eq(4)
```

### Krabi — Land Cover Clustering

K-Means is run on the Sentinel-2 Krabi image with varying numbers of clusters (K = 3, 5, 7, 10) to observe the effect on land cover separation.

```python
Krabi_sample = KrabiImg.sample(region=studyArea, scale=10, numPixels=5000)

my_clustering5  = ee.Clusterer.wekaKMeans(nClusters=5).train(Krabi_sample)
my_clustering7  = ee.Clusterer.wekaKMeans(nClusters=7).train(Krabi_sample)
my_clustering10 = ee.Clusterer.wekaKMeans(nClusters=10).train(Krabi_sample)
```

**Important caveat on accuracy for unsupervised results:** When computing an error matrix for K-Means output, the reported overall accuracy will appear very low (e.g. ~1.7%). This is not a true reflection of the clustering quality — it occurs because K-Means assigns cluster indices arbitrarily, so they rarely align with the integer class IDs in the reference shapefile. Urban areas, for example, were found spread across clusters 0 and 1 due to the spectral variability of built-up surfaces.

---

---

## Notes

- The median reducer over an ImageCollection helps suppress cloud and cloud-shadow artefacts. Experimenting with the minimum reducer will show a different result.
- K-Means cluster labels are not semantically meaningful on their own — use the GEEMap Inspector tool to identify which cluster corresponds to which land cover type.
- Always use a separate validation dataset (not the training dataset) for accuracy assessment to avoid optimistic bias.
