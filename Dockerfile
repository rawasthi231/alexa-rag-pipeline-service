# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5500

# Command to run the app
CMD ["python", "entrypoint.py"]
