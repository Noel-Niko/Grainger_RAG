# A Simple Relevance-Aware Generation (RAG) Application

![Starting Up Web Interface](https://github.com/Noel-Niko/grainger_rag/assets/83922762/cc674a06-bd70-4f21-932d-b352789d154a)

Proposed Plan for Managing FAISS Index with AWS Services

Objective: Efficiently manage updates and distribution of a FAISS index across containers in a scalable and fault-tolerant manner.

Architecture Overview:

S3 as Index Repository: Utilize Amazon S3 to serve as the primary repository for the FAISS index. Store the complete index as an object in S3, ensuring durability and availability.

Lambda for Index Updates: Implement AWS Lambda functions to handle index updates. These functions will be triggered by events such as product modifications or deletions. Upon triggering, Lambda will retrieve the latest index from S3, perform the necessary updates using FAISS update functions, and store the updated index back to S3.

Green/Blue Deployment with Containerization: Adopt a Green/Blue deployment strategy with container orchestration (e.g., Kubernetes). During updates, new container instances will be provisioned with the latest FAISS index fetched from S3. This ensures seamless updates without impacting the availability of the application.

Detailed Workflow:

 - Index Initialization:

 - Use a Lambda function to generate the initial FAISS index from the source data.
 - Store the generated index in S3 as an object.
 
 - Index Updates:
    - Trigger Lambda functions in response to events (e.g., product updates).
    - Retrieve the current index from S3.
    - Apply necessary updates using FAISS update functions.
    - Store the updated index back to S3.

Container Deployment:
    - During deployments, retrieve the latest index from S3. 
    - Provision new container instances with the fetched index.
    - Gradually switch traffic to the new containers (Green/Blue deployment).
    - Scale up or down the number of containers as needed.

Advantages:

Scalability: S3 provides extensive storage capacity, allowing the index to scale with the growth of data.
Fault Tolerance: S3 ensures high availability and durability of the index, minimizing the risk of data loss.
Efficient Updates: Lambda enables efficient and timely updates to the index, ensuring that containers always operate with the latest data.
Seamless Deployments: Green/Blue deployment strategy ensures zero downtime during updates, maintaining uninterrupted service availability.




![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/cb599178-5400-4ce8-984b-bec9e6d4e869)

![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/8a125cdb-e533-42a8-903c-8337132e9f86)



LOCAL INSTALL AND RUN
  - Create a local conda env named: rag_env
  - Download https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet
      - Place it in BOTH:
          - rag_application/modules/shopping_queries_dataset
          - rag_application/test_shopping_queries_dataset
  - Update local_start.sh with your path: export PYTHONPATH="
  - Run local_start.sh

  
LOCAL UNIT TESTING  
  Note: Due to the use of conda to help manage library version compatibility, the packages to install are listed primarily in the start shell and only those requiring a pip install in requirements.txt To set up a local env 
  1. Create a conda env named: rag_env
  2. Install the packages as listed in both requirements.txt and local_start.sh AND INCLUDE the commented-out test packages:
      - #conda install -y pytest==8.2.1  <<< testing pkg
      - #conda install -y Faker==25.2.0  <<< testing pkg
  4. In Pycharm system settings set your new conda env as the python interpreter.
  5. Ensure you have downloaded and added shopping_queries_dataset_products.parquet as described above.




TROUBLE-SHOOTING

If - FAISS vector index build failed ./start.sh: line 43:   174 Killed    python -m rag_application.modules.vector_index_faiss
THEN - increase memory limits e.g. in Docker to handle the required large shopping_queries_dataset_products.parquet 
   OR use self.products_df.dropna().drop_duplicates().sample(frac=0.001) in preprocess_data.py

IF - your Docker build fails with unfound url's
THEN - you are likely running on a Grainger computer with restrictions circumventing the wget

IF - you are running unit tests, and the self._index.add(embeddings_np) causes infinite hanging or you receive a SIGABRT
THEN - re-run the application in a NON-apple silicone device

IF - you continue to experience 'hanging' or infinite looping, or receive a segmentation fault error.
THEN - the cause is likely the mismatch between faiss-cpu, intel mkl, pytorch, python, numbpy, and or using apple silicon  (For example see [here](https://numpy.org/devdocs/user/troubleshooting-importerror.html).)
  - be aware: Faiss 1.7.3 is not compatible with Python >=3.10 or the corresponding pytorch for 3.9
  - ensure your local is running in a conda env with the versions as directed above




