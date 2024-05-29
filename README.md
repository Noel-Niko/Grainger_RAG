# grainger_rag

Proposed Plan for Managing FAISS Index with AWS Services

Objective: Efficiently manage updates and distribution of a FAISS index across containers in a scalable and fault-tolerant manner.

Architecture Overview:

S3 as Index Repository: Utilize Amazon S3 to serve as the primary repository for the FAISS index. Store the complete index as an object in S3, ensuring durability and availability.

Lambda for Index Updates: Implement AWS Lambda functions to handle index updates. These functions will be triggered by events such as product modifications or deletions. Upon triggering, Lambda will retrieve the latest index from S3, perform the necessary updates using FAISS update functions, and store the updated index back to S3.

Green/Blue Deployment with Containerization: Adopt a Green/Blue deployment strategy with container orchestration (e.g., Kubernetes). During updates, new container instances will be provisioned with the latest FAISS index fetched from S3. This ensures seamless updates without impacting the availability of the application.

Detailed Workflow:

Index Initialization:

Use a Lambda function to generate the initial FAISS index from the source data.
Store the generated index in S3 as an object.
Index Updates:

Trigger Lambda functions in response to events (e.g., product updates).
Retrieve the current index from S3.
Apply necessary updates using FAISS update functions.
Store the updated index back to S3.
Container Deployment:

During deployments, retrieve the latest index from S3.
Provision new container instances with the fetched index.
Gradually switch traffic to the new containers (Green/Blue deployment).
Scale up or down the number of containers as needed.
Advantages:

Scalability: S3 provides virtually unlimited storage capacity, allowing the index to scale with the growth of data.
Fault Tolerance: S3 ensures high availability and durability of the index, minimizing the risk of data loss.
Efficient Updates: Lambda enables efficient and timely updates to the index, ensuring that containers always operate with the latest data.
Seamless Deployments: Green/Blue deployment strategy ensures zero downtime during updates, maintaining uninterrupted service availability.
Conclusion:


![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/cb599178-5400-4ce8-984b-bec9e6d4e869)

![image](https://github.com/Noel-Niko/grainger_rag/assets/83922762/8a125cdb-e533-42a8-903c-8337132e9f86)



TROUBLE SHOOTING

If - FAISS vector index build failed ./start.sh: line 43:   174 Killed    python -m rag_application.modules.vector_index_faiss
THEN - increase memory limits in Docker to handle the required large shopping_queries_dataset_products.parquet

IF - your Docker build fails with unfound url's
THEN - you are likely running on a Grainger computer with restrictions circumventing the wget

IF - you are running unit tests, cna the self._index.add(embeddings_np) causes infinite hanging or you receive a SIGABRT
THEN - re-run the application an a NON apple silicone device



