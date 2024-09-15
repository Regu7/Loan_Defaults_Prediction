# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the src directory contents into the container at /app
RUN mkdir -p /app/src
COPY src /app/src

# Create the artifacts directory and copy files into it
RUN mkdir -p /app/artifacts
COPY artifacts/model.pkl /app/artifacts
COPY artifacts/preprocessor.pkl /app/artifacts
COPY artifacts/cat_unique_values.json /app/artifacts

# Copy the deploy directory contents into the container at /app
RUN mkdir -p /app/artifacts
COPY deploy /app/deploy

# Copy the app.py file into the container at /app
COPY app.py /app

COPY setup.py /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME LoanDefaultPrediction

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
