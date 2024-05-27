import faiss
import numpy as np


d = 64
nb = 1000
ncent = 16
K = 2

# Generate random vectors
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.

# Generate centroids
centroids = np.random.random((ncent, d)).astype('float32')

# Assign vectors to clusters
cluster_assignments = np.random.randint(K, size=nb)

# Create the quantizer
quantizer = faiss.IndexFlatL2(d)

# Create and train the index
index = faiss.IndexIVFPQ(quantizer, d, ncent, K, 8)  # Assuming L2 distance metric

index.train(xb)

# Add vectors to the index
index.add(xb)
print(f"Number of vectors in the index: {index.ntotal}")

# Search for the 5 nearest neighbors of a random vector
xq = np.random.random((1, d)).astype('float32')
D, I = index.search(xq, 5)  # Search

print("Distances:", D)
print("Indices:", I)
