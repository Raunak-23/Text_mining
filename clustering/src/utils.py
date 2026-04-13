import pickle
import os

def save_clusters(clusterer, filepath="models/narrative_state.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'centroids': clusterer.centroids,
        'meta': clusterer.cluster_meta
    }
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

def load_clusters(clusterer, filepath="models/narrative_state.pkl"):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            clusterer.centroids = state['centroids']
            clusterer.cluster_meta = state['meta']
        return True
    return False