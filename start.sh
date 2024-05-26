#!/bin/bash

# Navigate to the directory containing your scripts and data
cd /app

# Run the preprocessing script
python preprocess_data.py || { echo "Preprocessing failed"; exit 1; }


# Run the FAISS vector index script
python vector_index_faiss.py || { echo "FAISS vector index build failed"; exit 1; }

# Start Streamlit
exec streamlit run rag_application/modules/user_interface.py --server.port=8505 --server.address=0.0.0.0 || { echo "Streamlit failed to start"; exit 1; }

##!/bin/bash
#
## Check if the script is being sourced or executed directly
#if [[ "${BASH_SOURCE}"!= "${0}" ]]; then
#    echo "This script cannot be sourced."
#    exit 1
#fi
#
## Navigate to the directory containing your scripts and data
#cd /path/to/your/project || { echo "Failed to change directory"; exit 1; }
#
## Export environment variables securely
#echo "Setting environment variables..."
#export LANGCHAIN_TRACING_V2=true
#export LANGCHAIN_API_KEY="YOUR_API_KEY_HERE"
#export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
#
## Verify environment variables are set
#if [[ -z "$LANGCHAIN_API_KEY" ]] || [[ -z "$OPENAI_API_KEY" ]]; then
#    echo "One or more required environment variables are missing."
#    exit 1
#fi
#
## Run the preprocessing script
#python preprocess_data.py || { echo "Preprocessing failed"; exit 1; }
#
## Assuming vector_index.py builds the vector index and saves it to a file
## Ensure this script is correctly placed and executable
#python vector_index.py || { echo "Vector index build failed"; exit 1; }
#
## Start Streamlit
#exec streamlit run rag_application/modules/user_interface.py --server.port=8505 --server.address=0.0.0.0 || { echo "Streamlit failed to start"; exit 1; }


#
#
##!/bin/bash
#
## Set -e option to exit immediately if a command exits with a non-zero status
#set -e
#
## Custom error handling function
#function error_exit {
#    # Print the name of the script and the error message
#    echo "$(basename $0): ${1:-"Unknown Error"}" 1>&2
#    exit 1
#}
#
## Trap command to catch signals and perform cleanup
#trap 'error_exit "Caught signal"' SIGINT SIGTERM
#
## Navigate to the directory containing your scripts and data
#cd /path/to/your/project || error_exit "Failed to change directory"
#
## Run the preprocessing script
#python preprocess_data.py || error_exit "Preprocessing failed"
#
## Assuming vector_index.py builds the vector index and saves it to a file
## Ensure this script is correctly placed and executable
#python vector_index.py || error_exit "Vector index build failed"
#
## Start Streamlit
#exec streamlit run rag_application/modules/user_interface.py --server.port=8505 --server.address=0.0.0.0 || error_exit "Streamlit failed to start"
