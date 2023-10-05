FROM python:3.10.13

# Set the working directory
WORKDIR /app

# Copy the content of the local src directory to the working directory
COPY ./ /app

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /var/cache/apk/*

# Specify the command to run on container start
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]