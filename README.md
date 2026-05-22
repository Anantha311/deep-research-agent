# Setup and Run Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/Anantha311/deep-research-agent.git
cd deep-research-agent
```

---

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
```

---

## 5. Run the Streamlit Application

```bash
streamlit run ui/app.py
```

The application will start locally at:

```txt
http://localhost:8501
```

---

# Running the Evaluation Harness

Run the evaluation suite:

```bash
python evaluation/harness.py
```

Evaluation outputs will be written to:

```txt
evaluation/results.json
```

---

# System Requirements

- Python 3.11+
- Internet connection required for Tavily and Gemini APIs
- Recommended RAM: 8 GB+
- Linux/macOS recommended (Windows also supported)

---

# Notes

- The embedding model (`all-MiniLM-L6-v2`) runs locally on CPU.
- GPU acceleration is optional and not required.
- SQLite is used for persistent session storage.
- The app streams intermediate research steps in real time through the Streamlit UI.