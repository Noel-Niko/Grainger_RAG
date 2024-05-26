import faiss
import numpy as np

# Create a simple FAISS index
d = 64  # Dimensionality of the vectors
nb = 1000  # Number of database vectors

# Generate random vectors
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.

# Create the index
index = faiss.IndexFlatL2(d)
print(f"Is trained: {index.is_trained}")

# Add vectors to the index
index.add(xb)
print(f"Number of vectors in the index: {index.ntotal}")

# Search for the 5 nearest neighbors of a random vector
xq = np.random.random((1, d)).astype('float32')
D, I = index.search(xq, 5)  # Search

print("Distances:", D)
print("Indices:", I)
