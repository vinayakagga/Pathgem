FROM pathwaycom/pathway:latest

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create data directory
RUN mkdir -p data

# Expose the port FastAPI will run on
EXPOSE 8000

# Install gunicorn for multi-process web server
RUN pip install gunicorn==21.2.0

# Use gunicorn with uvicorn workers for multi-processing
CMD ["gunicorn", "app:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]