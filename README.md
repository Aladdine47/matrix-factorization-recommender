# Matrix Factorization Recommender System

**A collaborative filtering model built from scratch using NumPy and Gradient Descent**

Implemented matrix factorization to predict movie ratings on the MovieLens 100k dataset (100,000 ratings, 943 users, 1,682 movies). Achieved **MSE of 0.91** on test set.

## What I Built

This is a **from-scratch implementation** of the core algorithm behind recommender systems like Netflix and Spotify. Instead of using high-level ML libraries (scikit-learn, TensorFlow), I implemented:

- Matrix factorization model (factorizes user-item matrix into latent feature matrices)
- Custom gradient descent optimization loop
- MSE loss function
- Training/evaluation pipeline with visualizations

## Key Technical Highlights

- **Pure NumPy**: No ML frameworks - demonstrates understanding of the underlying math
- **Gradient Descent**: Custom implementation of stochastic gradient descent
- **80/20 Train-Test Split**: Stratified by user to prevent data leakage
- **5 Latent Factors**: Learns hidden user preferences and movie characteristics

## Results

The model successfully predicts movie ratings with good accuracy:

| Metric | Value |
|--------|-------|
| Test MSE | 0.91 |
| Training Epochs | 100 |
| Learning Rate | 0.001 |
| Dataset | MovieLens 100k |

**Visualizations included:**
- Training loss curve showing convergence over epochs
- Error distribution histogram for test set predictions
- Statistical analysis of prediction accuracy

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Dan-Ofri/matrix-factorization-recommender.git
cd matrix-factorization-recommender
pip install -r requirements.txt

# Run the notebook
jupyter notebook Matrix_Factorization_Recommender.ipynb
```

The notebook will download the MovieLens dataset, train the model, and display visualizations.

## Project Structure

```
├── Matrix_Factorization_Recommender.ipynb   # Main implementation
├── README.md                                 
├── requirements.txt                          # pandas, numpy, matplotlib
└── .gitignore                               
```

## Tech Stack

**Python 3.7+** | **NumPy** | **Pandas** | **Matplotlib** | **Jupyter**

---

*Built to demonstrate machine learning fundamentals and algorithm implementation skills*
