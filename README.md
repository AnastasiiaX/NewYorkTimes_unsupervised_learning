# New York Times Unsupervised Learning

Topic modeling and clustering of **New York Times** articles using unsupervised machine learning techniques.

---

## Project Overview

This repository includes:

-   scripts and notebooks for downloading and preprocessing NY Times articles,
-   implementations of topic modeling (LDA) and clustering (K-means, DBSCAN),
-   data visualization for interpreting results.

The goal is to uncover hidden patterns and topics in news content using unsupervised learning — no manual labeling required.

---

## Getting Started

### 1. Clone the repository:

```bash
git clone https://github.com/AnastasiiaX/NewYorkTimes_unsupervised_learning.git
cd NewYorkTimes_unsupervised_learning
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set up your NY Times API key:

Get a free NY Times API key from developer.nytimes.com

Store the key in a config.py or .env file as needed.

### 4. Fetch articles:

```bash
python fetch_articles.py
```

### 5. Launch Jupyter notebooks:

```bash
Open and run the following notebooks:

preprocessing.ipynb — Text cleaning and tokenization

topic_modeling.ipynb — Clustering and topic modeling

visualization.ipynb — Charts and topic visualizations
```

## Technologies Used

| Task               | Tools / Libraries                   |
| ------------------ | ----------------------------------- |
| Data collection    | `requests`, NY Times API            |
| Text preprocessing | `nltk`, `spaCy`, `pandas`           |
| Topic modeling     | `gensim`, `scikit-learn`            |
| Visualization      | `matplotlib`, `seaborn`, `pyLDAvis` |
| Notebooks          | Jupyter Notebook                    |

## Directory Structure

```
NewYorkTimes_unsupervised_learning/
│
├── data/                         # Raw article data
├── notebooks/                    # Jupyter Notebooks
│     ├── preprocessing.ipynb
│     ├── topic_modeling.ipynb
│     └── visualization.ipynb
├── fetch_articles.py             # Script to collect NYT data
├── requirements.txt              # Python dependencies
└── README.md                     # Project description (this file)
```

## Author: AnastasiiaX
