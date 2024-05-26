# base image
FROM python:3.10


## Set environment variables
#ENV PYTHONDONTWRITEBYTECODE=1
#ENV PYTHONUNBUFFERED=1
#ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORKDIR /app

# Update package lists and install protobuf development files and compiler
RUN apt-get update && \
    apt-get install -y libprotobuf-dev protobuf-compiler && \
    apt-get update && \
    apt-get -y install vim


# copy over source code
COPY . /app
# TODO: UPDATE PERMISSIONS TO 755 FOR PROD
RUN chmod -R 777 /app
RUN mkdir -p /rag_application/model/huggingface/hub

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Install pip dependencies
COPY requirements.txt /app/requirements.txt
# TODO: consider using --no-cache-dir if concerned about cache size growth or conflicts with existing cached packages
RUN pip install -r /app/requirements.txt


# Expose port
EXPOSE 8505
CMD ["./start.sh"]