# ⚡ AI-Powered Energy Forecaster

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)](https://streamlit.io) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced web application built for the **AI & Sustainability Hackathon**. This project forecasts household energy consumption using multiple time-series models (**Prophet** and **LightGBM**) and provides **AI-generated, actionable tips** to promote energy efficiency and sustainable living, directly contributing to **SDG 7 (Affordable and Clean Energy)**.

---

## 🎯 Key Features

- 📈 **Historical Data Visualization:** Interactive charts of past household energy usage.
- 🔮 **Dual-Model Forecasting:** Predicts consumption for the next 24–72 hours using both Facebook's **Prophet** and a high-performance **LightGBM** model with lag features.
- 🆚 **Model Comparison & Overlay:** Allows users to dynamically switch between models or overlay both forecasts on a single chart to compare their trends and predictions.
- 🔍 **Automated Peak Detection:** Automatically identifies and highlights upcoming peak consumption hours based on a user-defined threshold.
- 💡 **Multilingual AI Eco-Advice:** Generates rule-based tips and uses an LLM (**Mistral-7B**) to paraphrase them into user-friendly advice in English, Myanmar, or French.
- 🔬 **Advanced Model Insights:**
  - **Prophet:** Visualizes seasonal components (daily/weekly trends).
  - **LightGBM:** Displays feature importances and partial dependence plots to explain what drives the model's predictions.

---

## 🛠️ Technology Stack

- **Language:** Python
- **Web Framework:** [Streamlit](https://streamlit.io)
- **Data Manipulation:** Pandas, NumPy
- **Forecasting Models:**
  - [Prophet](https://facebook.github.io/prophet/) (by Meta)
  - [LightGBM](https://lightgbm.readthedocs.io/) (Gradient Boosting)
- **Visualization:** Plotly
- **Eco-Advice LLM:** Hugging Face Inference API + [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Environment:** Docker & VS Code Dev Containers

---

## 🚀 Getting Started

This project uses a **Dev Container** to ensure a 100% consistent development environment for all team members.

### Prerequisites
1.  **Docker Desktop:** Make sure it is installed and running on your system.
2.  **VS Code:** Install Visual Studio Code.
3.  **Dev Containers Extension:** Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code.

### Recommended Setup (Dev Container)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/kyawphyoaung/AI-Energy-Forecasting.git
    cd AI-Energy-Forecasting
    ```

2.  **Reopen in Container:**
    - Open the cloned folder in VS Code.
    - A pop-up will appear in the bottom-right corner asking: **"Reopen in Container"**.
    - Click that button. VS Code will automatically build the Docker image, install all dependencies from `requirements.txt`, and set up the entire environment for you. This may take a few minutes the first time.

### Manual Setup (venv - Fallback)
If you cannot use Docker, you can set up a local environment manually.
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


# ▶️ How to Run the App

Once your environment is set up (either in the Dev Container or a local venv), run the app with:

```Bash
streamlit run streamlit_app.py
```
The application will be available at http://localhost:8501.

# 📊 Dataset
We use the UCI Household Power Consumption Dataset, preprocessed into hourly averages for faster performance.

The repository includes data/hourly_power.csv which is ready to use.
Optionally, you can upload the raw UCI file (.txt or .csv) directly in the app's sidebar, and it will be processed automatically.

# ☁️ Deployment
This application is designed for deployment on Streamlit Community Cloud.

Link your GitHub repository.
Set the following in Streamlit Cloud → Settings → Secrets:
Ini, TOML

Optional, for LLM-powered advice
```bash
HF_TOKEN = "hf_xxx..."
```
Deploy. 

The app will automatically update whenever you push changes to the main branch.

Live App URL (after deploy): 👉 https://ai-energy-forecasting.streamlit.app/


## 📜 License

This project is licensed under the [MIT License](LICENSE).

---