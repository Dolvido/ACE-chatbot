# ACE Chatbot
This project implements an chatbot powered by an Autonomous Cognitive Entity (ACE) to answer questions about its internal state.

# Overview
The chatbot is built using the LangChain library and consists of the following main components:

ACE Layers - Implemented in ace.py, this defines the different layers of the ACE entity based on the ACE framework. Each layer processes incoming messages and generates responses using a large language model.
Logging - The conversation between the layers is logged to YAML files in the logs folder.
Vectorstore - The log files are vectorized using sentence transformers and indexed in a FAISS vectorstore (vectorstore.pkl). This allows quick nearest neighbor search.
RetrievalQA - query_data.py defines LangChain RetrievalQA chains powered by the vectorstore to retrieve relevant log context and generate a response to the question.
Web UI - A simple Gradio web UI is implemented in gradio_app.py to interact with the chatbot.
Data Ingestion - ingest_data.py uses Watchdog to monitor changes to the logs and automatically re-index them.

# Usage
To start the chatbot:

Run ace.py to initialize the ACE layers and simulate some sample conversations. This will populate the logs/ folder.
Run ingest_data.py to index the log files into the vectorstore.
Run gradio_app.py to launch the web UI.
Go to http://localhost:7860 to interact with the chatbot.
The chatbot can now answer questions about the internal state of the ACE based on the logged conversations. Whenever new logs are added, re-run ingest_data.py to update the indexed data.

# Customization
The ACE layers in ace.py can be modified to customize the internal ACE architecture.
Different retrieval pipelines can be defined in query_data.py by using different prompt templates.
The web UI can be customized by editing gradio_app.py.
Requirements
Python 3.7+
LangChain
Transformers, FAISS, Gradio, Watchdog (see requirements.txt for versions)
Large language model like GPT-4All (see ace.py)
