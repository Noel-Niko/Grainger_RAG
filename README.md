# grainger_rag
![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/cb599178-5400-4ce8-984b-bec9e6d4e869)

![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/8a125cdb-e533-42a8-903c-8337132e9f86)



TROUBLE SHOOTING
If - FAISS vector index build failed
./start.sh: line 43:   174 Killed                  python -m rag_application.modules.vector_index_faiss
THEN - increase memory limits in Docker to handle the required large shopping_queries_dataset_products.parquet
