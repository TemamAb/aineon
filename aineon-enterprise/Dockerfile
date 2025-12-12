# Multi-stage build: Node.js + Python
FROM node:18-alpine as node-base
WORKDIR /app
COPY package*.json ./
RUN npm install

# Python stage for Streamlit
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for Streamlit dashboard
RUN pip install streamlit pandas plotly numpy websockets

# Copy Node.js application from previous stage
COPY --from=node-base /app /app
COPY . .

# Install Node.js dependencies
RUN npm install

# Expose ports: Node.js on 3000, Streamlit on 8501
EXPOSE 3000 8501

# Start both services
CMD ["sh", "-c", "npm start & streamlit run /app/../aineon-enterprise/dashboard/monitoring_dashboard.py --server.port 8501 --server.address 0.0.0.0"]
