# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt from the root folder
COPY ../../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all frontend source files into the container
COPY . .

# Expose port for the frontend
EXPOSE 3000

# Run the frontend application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

