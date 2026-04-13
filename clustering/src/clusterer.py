import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class IncrementalClusterer:
    def __init__(self, threshold=0.7):
        self.centroids = {}      
        self.cluster_meta = {}   
        self.threshold = threshold

    def update(self, vectors, metadata_df):
        for i, vec in enumerate(vectors):
            assigned = False
            if self.centroids:
                c_ids = list(self.centroids.keys())
                c_vecs = np.array([self.centroids[id] for id in c_ids])
                
                # Compare new article vector against current cluster centroids
                sims = cosine_similarity(vec.reshape(1, -1), c_vecs)[0]
                best_idx = np.argmax(sims)
                
                if sims[best_idx] > self.threshold:
                    cid = c_ids[best_idx]
                    # Incremental update: shift centroid slightly toward the new article
                    self.centroids[cid] = (self.centroids[cid] * 0.95) + (vec * 0.05)
                    self.cluster_meta[cid]['count'] += 1
                    assigned = True

            if not assigned:
                # Start a new narrative cluster
                new_id = f"NARRATIVE_{len(self.centroids) + 1}"
                self.centroids[new_id] = vec
                self.cluster_meta[new_id] = {
                    'count': 1, 
                    'label': metadata_df.iloc[i]['Title']
                }
        return len(self.centroids)