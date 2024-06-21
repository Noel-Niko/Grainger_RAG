# A Simple Retrieval-Augmented Generation (RAG) Application
![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/5ceebba5-8680-4237-96e4-3ad7e8022faa)

![Womens clothing](https://github.com/Noel-Niko/simple_retrieval_augmented_generation/assets/83922762/0b51fc63-d731-4150-89a5-ace25dc66246)





![image](https://github.com/Noel-Niko/simple_retrieval_augmented_generation/assets/83922762/c78dde55-8b86-45e7-8f73-e393ebdb816a)

### Architecture Overview Utilizing AWS Infrastructure and Services:

**S3 as Index Repository**: Utilize Amazon S3 to serve as the primary repository for the FAISS index. Store the complete index as an object in S3, ensuring durability and availability. Additionally, partitioning the index into smaller segments if possible to facilitate parallel processing and faster updates.

**Elastic Map Reduce (EMR) for Data Preprocessing**: Integrate AWS EMR to preprocess and normalize the product data before generating or updating the FAISS index, leveraging its distributed computing capabilities.

**Lambda for Index Updates**: Implement AWS Lambda functions to handle index updates. These functions are triggered by events such as product modifications or deletions. Upon triggering, Lambda retrieves the latest index from S3, applies the necessary updates using FAISS update functions, and stores the updated index back to S3.

**ECS for Application Deployment**: Deployment with container orchestration using Amazon Elastic Container Service (ECS). ECS manages the deployment of application containers, ensuring they have access to the latest FAISS index from S3. This approach maintains high availability and scalability, supporting dynamic scaling based on demand.

### Detailed Workflow:

**Index Initialization**:

- Use a combination of EMR and Lambda to generate the initial FAISS index from the source data. EMR processes the data normalization and preparation, while Lambda handles the final conversion to a FAISS index.
- Store the generated index in S3 as an object.

**Index Updates**:

- Trigger Lambda functions in response to events (e.g., product updates).
- Retrieve the current index from S3.
- Apply necessary updates using FAISS update functions.
- Store the updated index back to S3.

**Application Deployment with ECS**:

- Define a task definition in ECS that specifies the Docker image for your application, along with the necessary environment variables and volume mounts to access the FAISS index from S3.
- Create a service in ECS to manage the deployment of application containers. Configure the service to pull the latest version of the application image and to dynamically scale based on demand.
- During deployments, the ECS service ensures that new container instances are provisioned with the latest FAISS index fetched from S3. This setup supports seamless updates without impacting the availability of the application.

### Advantages:

- **Scalability**: Leveraging S3 for storage, EMR for data processing, and ECS for application deployment allows the system to scale with the growth of data. Partitioning the index and utilizing ECS's dynamic scaling capabilities facilitate handling larger datasets efficiently.
- **Fault Tolerance**: S3 ensures high availability and durability of the index, minimizing the risk of data loss. The integration of EMR and ECS adds layers of resilience through distributed computing models and managed container orchestration.
- **Efficient Updates**: Combining EMR for data preprocessing with Lambda for index updates ensures that the system can efficiently handle both large-scale data transformations and timely index updates.
- **Seamless Deployments**: The use of ECS for application deployment ensures zero downtime during updates, maintaining uninterrupted service availability. ECS's support for rolling updates and blue/green deployments to ensure reliability and availability.



![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/cb599178-5400-4ce8-984b-bec9e6d4e869)




### LOCAL INSTALL AND RUN
  1. Obtain and add api key and email address to env variables (see constants.py), as well as export TOKENIZERS_PARALLELISM=false for the huggingface version.
  2. Create a local conda env with python 3.10
  3. Download https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet
      - Place it in BOTH:
          - rag_application/modules/shopping_queries_dataset
          - rag_application/shopping_queries_dataset
  4. Install the required libraries into your conda environment
      - pip install faiss-cpu
      - conda install pandas nltk numpy transformers pytorch scikit-learn langchain langchain-openai langsmith streamlit langdetect pyyaml packaging
      - python -m spacy download ja_core_news_sm && \python -m spacy download es_core_news_sm && \python -m spacy download en_core_web_sm
  5. The size of the products_df data frame can be reduced for speed of processing for test and demo. It can be returned adjusted in preprocess_data.py line 44 under  # Data Cleaning
  6. Run start_local.sh

    NOTE: Pickle, Singleton design pattern, and streamlit state annotation are used so that while the program is running the creation of the initial faiss index is persisted and reused to prevent the need to recreate. Updates can be made to that existing index through the methods included and the pickle file replace with the updated version.

  
### LOCAL UNIT TESTING  
  Note: Due to the use of conda to help manage library version compatibility, the packages to install are listed primarily in the start shell and only those requiring a pip install in requirements.txt To set up a local env 
  1. Create a conda env named: simple_retrieval_augmented_generation_env
  2. Install the packages as listed in both requirements.txt and local_start.sh AND INCLUDE the commented-out test packages:
      - #conda install -y pytest==8.2.1  <<< testing pkg
      - #conda install -y Faker==25.2.0  <<< testing pkg
  4. In Pycharm system settings set your new conda env as the python interpreter.
  5. Ensure you have downloaded and added shopping_queries_dataset_products.parquet as described above.




### TROUBLE-SHOOTING

If - FAISS vector index build failed ./start.sh: line 43:   174 Killed    python -m rag_application.modules.vector_index_faiss

THEN - increase memory limits e.g. in Docker to handle the required large shopping_queries_dataset_products.parquet 
   OR use self.products_df.dropna().drop_duplicates().sample(frac=0.001) in preprocess_data.py


IF - your Docker build fails with un-found url's

THEN - you are likely running on a corporate (i.e. Grainger) computer with restrictions circumventing the wget


IF - you are running unit tests, and the self._index.add(embeddings_np) causes infinite hanging or you receive a SIGABRT or SIGSEGV

THEN - re-run the application on a NON-apple silicone device


IF - you continue to experience 'hanging' or infinite looping, or receive a segmentation fault error.

THEN - the cause is likely the mismatch between faiss-cpu, intel mkl, pytorch, python, numbpy, and or using apple silicon  (For example see [here](https://numpy.org/devdocs/user/troubleshooting-importerror.html).)
  - be aware: Faiss 1.7.3 is not compatible with Python >=3.10 or the corresponding pytorch for 3.9
  - ensure your local is running in a conda env as directed above




