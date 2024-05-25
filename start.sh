#!/bin/bash

python preprocess_data.py

python vector_index.py

# Start Streamlit
exec streamlit run rag_application/modules/user_interface.py --server.port=8505 --server.address=0.0.0.0
