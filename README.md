🏥 Medical RAG Assistant

The Medical RAG Assistant is an AI-powered chatbot designed to answer medical questions using information stored inside a PDF book. It works by reading your PDF, breaking it into small pieces, storing them in Pinecone, and then using Google Gemini to generate accurate answers based on that information. This project helps you build your own smart medical question-answering system using RAG (Retrieval-Augmented Generation).

📌 What this project does

This assistant allows you to:

Upload a medical book (PDF)

Break it into meaningful text chunks

Convert those chunks into embeddings

Store them inside Pinecone for fast searching

Retrieve relevant content when you ask questions

Generate final answers using Google Gemini AI

Chat using a clean and modern web interface

The goal is to give you a chatbot that does not guess answers, but instead uses your actual medical data.

🧠 How it works (in simple words)

You add your medical PDF into the data/ folder.

The script store_index.py reads the PDF and sends the processed text into Pinecone.

When you ask a question, the chatbot searches Pinecone for the best matching information.

Gemini takes that information and creates a meaningful answer.

The response is shown in your chatbot interface in the browser.

🚀 How to set everything up

You need Python and a virtual environment.

Step 1: Create environment
conda create -n medibot python=3.10 -y
conda activate medibot

Step 2: Install the required packages
pip install -r requirements.txt

Step 3: Add your API keys in a .env file
PINECONE_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

Step 4: Upload your PDF

Place your book inside:

data/Medical_book.pdf

Step 5: Send the PDF to Pinecone

Run:

python store_index.py




Step 6: Start the chatbot
python app.py


Now open the browser and visit:

http://127.0.0.1:8080


You will see your chatbot ready to use.
