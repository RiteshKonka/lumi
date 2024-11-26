# Lumi - Your AI Mental Health Therapist

A compassionate AI-powered mental health companion built with modern technology and a soothing user interface.

![Lumi Interface](purple_gradient.jpg)

## Overview

Lumi is an intelligent chatbot designed to provide mental health support through empathetic conversations and evidence-based guidance. Using RAG (Retrieval-Augmented Generation) technology, Lumi offers personalized responses while maintaining a warm, supportive presence.

## Features

- 💬 Real-time chat interface with a calming purple gradient design
- 🤖 Advanced AI responses powered by Groq's LLama3-70B model
- 📚 Knowledge-based responses using RAG technology
- 🧠 Mental health-focused conversation capabilities
- 💜 Modern, accessible user interface

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI Model**: Groq LLama3-70B
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace sentence-transformers
- **Memory**: ConversationBufferMemory

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lumi-therapist.git
cd lumi-therapist

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # For Unix/macOS
# or
.\venv\Scripts\activate  # For Windows

# Install dependencies
pip install -r requirements.txt

