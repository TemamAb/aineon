# Streamlit Dashboard Dockerfile for Render
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard and core modules
COPY dashboard/ ./dashboard/
COPY core/ ./core/

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV API_BASE_URL=http://localhost:8080
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run Streamlit dashboard
CMD ["streamlit", "run", "dashboard/monitoring_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
