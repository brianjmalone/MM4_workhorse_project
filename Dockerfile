FROM python:3.11-slim

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY notebooks/ ./notebooks/
COPY src/ ./src/

CMD ["bash"]
