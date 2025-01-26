# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt from the root folder
COPY ../../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source files into the container
COPY . .

# Expose port for the backend
EXPOSE 5000

# Run the backend application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

