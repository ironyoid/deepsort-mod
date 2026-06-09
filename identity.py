import numpy as np


def normalize(vector):
    return vector / (np.linalg.norm(vector) + 1e-12)


class IdentityDatabase:
    def __init__(self, threshold):
        self.threshold = threshold
        self.centroids = []
        self.counts = []

    def query(self, descriptor):
        descriptor = normalize(descriptor)
        if self.centroids:
            sims = np.array([centroid @ descriptor for centroid in self.centroids])
            best = int(np.argmax(sims))
            if 1.0 - sims[best] <= self.threshold:
                count = self.counts[best]
                self.centroids[best] = normalize(self.centroids[best] * count + descriptor)
                self.counts[best] = count + 1
                return best
        self.centroids.append(descriptor.copy())
        self.counts.append(1)
        return len(self.centroids) - 1
