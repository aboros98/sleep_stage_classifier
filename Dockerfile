FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt-get update && \
    apt-get install -y wget unzip git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
