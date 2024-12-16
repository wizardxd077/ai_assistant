#!/bin/bash

# Project Setup Script for Multimodal AI Assistant

# Create project root and directory structure
mkdir -p multimodal-ai-assistant/{frontend,backend,models,data,docs}
cd multimodal-ai-assistant

# Create subdirectories
mkdir -p backend/{app,tests,services}
mkdir -p frontend/{src,public}
mkdir -p models/{nlp,vision,audio}
mkdir -p data/{raw,processed}

# Initialize Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Create initial project files
touch README.md
touch .gitignore
touch requirements.txt
touch backend/app/main.py
touch backend/app/services/multimodal_service.py
touch frontend/src/App.js

# Populate .gitignore
cat << EOF > .gitignore
venv/
__pycache__/
*.pyc
.env
*.log
.DS_Store
node_modules/
build/
dist/
*.sqlite3
EOF

# Initial requirements.txt
cat << EOF > requirements.txt
# Core AI/ML Libraries
torch==2.1.2
transformers==4.36.0
accelerate==0.25.0
datasets==2.15.0

# Web Framework
flask==2.3.2
flask-cors==4.0.0
streamlit==1.29.0

# Computer Vision
opencv-python==4.8.1.78
PIL==9.5.0

# Audio Processing
librosa==0.10.1

# Speech Recognition
speechrecognition==3.10.0

# Natural Language Processing
nltk==3.8.1

# Utility
numpy==1.26.2
scipy==1.11.4
pandas==2.1.4

# API and Deployment
requests==2.31.0
gunicorn==21.2.0

# Testing
pytest==7.4.4
EOF

# Create a basic README
cat << EOF > README.md
# Multimodal AI Assistant

## Project Overview
A comprehensive AI assistant integrating natural language processing, computer vision, and audio recognition.

## Setup Instructions
1. Clone the repository
2. Create virtual environment: \`python3 -m venv venv\`
3. Activate environment: \`source venv/bin/activate\`
4. Install dependencies: \`pip install -r requirements.txt\`

## Project Structure
- \`/backend\`: Flask API and core services
- \`/frontend\`: React/Streamlit interface
- \`/models\`: ML model configurations
- \`/data\`: Training and inference data

## Key Features
- Natural Language Understanding
- Image Recognition
- Speech Processing
- Context-Aware Interactions

## TODO
- [ ] Train base models
- [ ] Implement multimodal integration
- [ ] Build frontend interfaces
EOF

# Initialize git
git init
git add .
git commit -m "Initial project setup for Multimodal AI Assistant"

echo "Project structure created successfully!"
