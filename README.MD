# AI Agent - CSV FAQ

An Agentic AI application that allows users to upload CSV files and ask natural language questions. The system answers strictly based on the uploaded data using a LangChain-powered Pandas DataFrame agent.

## Features

* Upload multiple CSV files
* Ask questions in natural language
* Answers grounded strictly in data
* No hallucinations (controlled via prompt)
* Simple Streamlit UI

## Tech Stack

* Python
* Streamlit
* Pandas
* LangChain
* OpenAI GPT-4o-mini

## Setup Instructions

```bash
git clone https://github.com/yourusername/ai-agent-csv-faq.git
cd ai-agent-csv-faq
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

Run the app:

```bash
streamlit run app.py
```

## Example Questions

* What is the return policy?
* What are hospital visiting hours?
* What is API rate limit?

## Learning Outcome

This project demonstrates how to build a simple Agentic AI system that uses tools (Pandas) to answer questions from structured data instead of relying on general knowledge.
