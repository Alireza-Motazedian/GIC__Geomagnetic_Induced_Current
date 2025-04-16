#!/bin/bash

# Get the parent directory name
PROJECT_NAME=$(basename "$(pwd)")

# Convert the project name to lowercase
PROJECT_NAME_LOWER=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]')

# Export the lowercase project name as an environment variable
export PROJECT_NAME=$PROJECT_NAME_LOWER

# Build and run the Docker Compose in detached mode
docker-compose up --build -d