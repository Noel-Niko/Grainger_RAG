#!/bin/bash

cd /app

# Run the Preprocessing file
echo "********************************************************* Preprocessing Data...:"
python rag_application/modules/preprocess_data.py || { echo "Preprocessing failed"; exit 1; }

# Serialize VectorIndex instance
echo "********************************************************* Serializing VectorIndex instance:"
if python rag_application/modules/serialize_vector_index.py; then
    echo "Serialization completed successfully."

    # Define the range of ports to try
    port_range=(8505)

    for port in "${port_range[@]}"; do
        # Use Python to check if the port is available
        python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('localhost', $port)); s.close(); print('Port $port is available')" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            # Port is available, start Streamlit with this port
            echo "Starting Streamlit Application..."
            exec streamlit run rag_application/modules/user_interface.py --server.port=$port --server.address=0.0.0.0 || { echo "Streamlit failed to start"; exit 1; }
            break
        fi
    done

    echo "No available port found in the specified range."
else
    echo "Serialization failed. Exiting."
    exit 1
fi