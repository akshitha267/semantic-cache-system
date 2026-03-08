from sklearn.mixture import GaussianMixture
import numpy as np

def build_clusters(embeddings):

    print("Training GMM clustering...")

    gmm = GaussianMixture(
        n_components=20,
        covariance_type="full"
    )

    gmm.fit(embeddings)

    probs = gmm.predict_proba(embeddings)

    return gmm, probs