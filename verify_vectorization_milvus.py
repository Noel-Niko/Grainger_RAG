import pytest
from rag_application.modules.vector_index_faiss import VectorIndex
import numpy as np

@pytest.fixture(scope='module')
def vector_index():
    """Fixture to create a VectorIndex instance for testing."""
    collection_name = 'test_collection'
    dimension = 128  # Example dimension size
    vi = VectorIndex(collection_name, dimension)
    yield vi
    # Cleanup after tests run
    vi.drop_collection()  # Make sure to implement this method in your VectorIndex class

def test_create_collection(vector_index):
    """Test creating a collection."""
    assert vector_index.collection_name == 'test_collection'
    assert vector_index.get_collection().name == 'test_collection'

def test_insert_vectors(vector_index):
    """Test inserting vectors."""
    vectors = np.random.rand(5, 128).astype('float32')  # Generate random vectors
    ids = list(range(5))  # Example IDs
    vector_index.insert_vectors(vectors, ids)
    fetched_ids = [entity.id for entity in vector_index.get_collection().fetch_all()]
    assert len(fetched_ids) == 5
    assert set(ids) <= set(fetched_ids)

def test_search(vector_index):
    """Test searching for vectors."""
    vectors = np.random.rand(5, 128).astype('float32')  # Generate random vectors
    ids = list(range(5))  # Example IDs
    vector_index.insert_vectors(vectors, ids)
    query_vector = vectors[0]  # Search for the first inserted vector
    results = vector_index.search_index(query_vector, k=2)
    assert len(results) == 2
    assert results[0] == ids[0]
    assert results[1] == ids[1]

def test_update_product_description(vector_index):
    """Test updating a product description."""
    product_id = 1
    original_description = "Original description"
    new_description = "Updated description"
    vector_index.update_product_description(product_id, new_description)
    # Verify the update by fetching the entity and checking the description
    entity = vector_index.get_collection().fetch_one({"id": product_id})
    assert entity.description == new_description

def test_remove_product_by_id(vector_index):
    """Test removing a product by ID."""
    product_id = 1
    vector_index.remove_product_by_id(product_id)
    # Verify the removal by attempting to fetch the entity
    with pytest.raises(Exception) as exc_info:
        _ = vector_index.get_collection().fetch_one({"id": product_id})
    assert "not found" in str(exc_info.value)

# Run the tests with pytest
# $ pytest test_vector_index.py
