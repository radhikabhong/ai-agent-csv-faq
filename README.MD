# 🤖 AI CSV FAQ Agent

A conversational AI app that answers natural language questions from uploaded CSV files — powered by LangChain, OpenAI GPT, and Streamlit.

🔗 **[Live Demo](https://ai-agent-csv-faq-mhhtkae5bgaycdycdhkrg7.streamlit.app)**

---

## 📌 What It Does

Upload any CSV file and ask questions in plain English. The AI agent reads your data and answers intelligently — no SQL, no formulas, just conversation.

**Example questions:**
- "What is the return policy?"
- "What are the visiting hours?"
- "Summarize the key terms of the credit card agreement"

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| AI Agent | LangChain CSV Agent |
| LLM | OpenAI GPT |
| Language | Python 3.11 |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/radhikabhong/ai-agent-csv-faq.git
cd ai-agent-csv-faq
```

**2. Install dependencies**
```bash
pip3 install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file:
```text
OPENAI_API_KEY=your_openai_api_key_here
```

**4. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure
```
ai-agent-csv-faq/
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .env                 # API key (not committed)
├── .gitignore
├── README.md
└── csv-files/           # Sample CSV datasets
    ├── ecommerce_faqs.csv
    ├── hospital_policy.csv
    ├── credit_card_terms.csv
    └── saas_docs.csv
```

---

## 🔐 Security

- API key stored in `.env` locally (excluded from GitHub via `.gitignore`)
- On Streamlit Cloud, secrets managed via the Secrets manager

---

## 👩‍💻 Author

**Radhika Bhong** — Senior Software Engineer  
[GitHub](https://github.com/radhikabhong) · [LinkedIn](https://linkedin.com/in/radhikabhong)

---

## ⭐ If you found this useful, give it a star!
