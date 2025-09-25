# ⚡ Smart Energy Forecaster

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web application built for the **AI & Sustainability Hackathon 2025**.
This project forecasts household energy consumption using multiple models and provides **AI-generated, actionable tips** to promote energy efficiency and sustainable living, directly contributing to **SDG 7 (Affordable and Clean Energy)**.

---

## 🎯 Key Features

- 📈 **Historical Data Visualization:** Interactive charts of past household energy usage.
- 🔮 **Next-Hour Forecasting:** Predicts consumption for the next 24–72 hours with **Prophet** and **LightGBM**.
- 🔍 **Automated Peak Detection:** Highlights peak consumption hours and estimates their intensity.
- ⚡ **Model Comparison & Overlay:** Toggle between models or overlay both forecasts to compare trends.
- 💡 **Eco-Advice:** Generates rule-based and **LLM-paraphrased tips** (English / Myanmar / French) for sustainable living.
- 🔬 **Model Insights:**
  - Prophet: Seasonal components (daily/weekly trends).
  - LightGBM: Feature importances & partial dependence (hour/day-of-week).

---

## 🛠 Technology Stack

- **Language:** Python
- **Framework:** [Streamlit](https://streamlit.io)
- **Data Manipulation:** Pandas, NumPy
- **Forecasting Models:**
  - [Prophet](https://facebook.github.io/prophet/) (by Meta)
  - [LightGBM](https://lightgbm.readthedocs.io/) (gradient boosting with lag features)
- **Visualization:** Plotly
- **Eco-Advice LLM:** Hugging Face Inference API + [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/soneeee22000/AI-Energy-Peak-Forecaster-Eco-Advice.git
cd AI-Energy-Peak-Forecaster-Eco-Advice
```

## 2. Create and Activate the Virtual Environment

Create and activate a dedicated virtual environment for the project.

### On macOS / Linux

```Bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows

```bash
python -m venv venv
venv\Scripts\activate
Note: If VS Code prompts you to select the new (venv) as the workspace interpreter, choose "Yes".
```

# 3. Install Dependencies

Install all the required libraries from the requirements.txt file.

```Bash
pip install -r requirements.txt
```

# ▶️ How to Run the App

To run the app on your local server, use the following command:

```Bash
streamlit run streamlit_app.py
```

Your web browser will automatically open a new tab at http://localhost:8501.

## 📊 Dataset

We use the [UCI Household Power Consumption Dataset](), preprocessed into **hourly averages** for faster loading and clearer patterns.

* Repository includes `data/hourly_power.csv` (ready-to-use).
* Optionally, you can upload the raw UCI file (`.txt` or `.csv`) and the app will process it automatically

# 🤝 Team Workflow & Contributing

To ensure a smooth collaboration, please follow the workflow below.

### Adding a New Library

If any team member needs to add a new library, please follow these three steps every time:

1. Install the library:

```Bash
pip install <new-library-name>
```

2. Update requirements.txt:

```Bash
pip freeze > requirements.txt
```

3. Commit and Push: Commit the updated requirements.txt file and push it to the repository.

```Bash
git add requirements.txt
git commit -m "feat: Add <new-library-name> for X feature"
git push
Important: Always run git pull to get the latest updates before you start working.
```

---

## ☁️ Deployment

This project can be deployed on  **Streamlit Community Cloud** :

1. Link your GitHub repo.
2. Set secrets in  **Streamlit Cloud → Settings → Secrets** :
   <pre class="overflow-visible!" data-start="3716" data-end="3948"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-toml"><span><span>DATA_URL</span><span> = </span><span>"https://raw.githubusercontent.com/soneeee22000/AI-Energy-Forecasting/main/data/hourly_power.csv"</span><span>
   </span><span>HF_TOKEN</span><span> = </span><span>"hf_xxx..."</span><span></span><span># optional, for LLM advice</span><span>
   </span><span>MODEL_ID</span><span> = </span><span>"mistralai/Mistral-7B-Instruct-v0.2"</span><span>
   </span></span></code></div></div></pre>
3. Deploy — the app auto-updates whenever you push changes to `main`.

**Live App URL (after deploy):**

👉 [https://ai-energy-forecasting.streamlit.app/](https://ai-energy-forecasting.streamlit.app/) *(replace with your Streamlit Cloud URL once deployed)*

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

### ✨ Acknowledgements

* Dataset: [UCI Machine Learning Repository]()
* Models: [Meta Prophet](), [LightGBM](https://github.com/microsoft/LightGBM)
* LLM: [Mistral AI](), via Hugging Face Inference API
* Built during **AI & Sustainability Hackathon FTL ML Bootcamp Myanmar UNDP 2025**
