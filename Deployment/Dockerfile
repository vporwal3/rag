# Use the official Azure Functions base image for Python.
FROM mcr.microsoft.com/azure-functions/python:4-python3.9

# Create a directory for the function app code
WORKDIR /home/site/wwwroot

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install pip dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the function app code
COPY . .
 
# If you store your OpenAI key as an env var, you can set it here (not recommended in plaintext)
# ENV OPENAI_API_KEY="replace_me"

# Final instruction to start Azure Functions (the entry point is provided by the base image)
