# 🛡️ ML Network Intrusion Detector

A machine-learning powered **Network Intrusion Detection System (NIDS)** built with Python and Streamlit. It classifies network traffic connections as **normal** or one of four attack types using supervised ML classifiers trained on a synthetic NSL-KDD-style dataset.

> ⚠️ **Demo mode only** — All traffic records are synthetically generated for educational purposes. This is not a production security tool.

---

## 📋 Project Description

This project demonstrates how machine learning can be applied to cybersecurity for automated threat detection. Three classifiers are trained and compared on synthetic network traffic data, and results are presented through an interactive dark-themed Streamlit dashboard with four pages:

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | Traffic distribution charts, attack breakdown, best-model metrics, sample data |
| 🔍 **Detect Traffic** | Enter 21 connection features → get a live classification with confidence score |
| 📈 **Model Performance** | Confusion matrix, classification report, feature importance chart |
| ℹ️ **About** | Background on NIDS, dataset details, limitations, future work |

### Attack Classes

| Label | Description |
|-------|-------------|
| `normal` | Legitimate network traffic |
| `dos` | Denial of Service — flood attacks |
| `probe` | Port scanning and reconnaissance |
| `r2l` | Remote-to-Local — unauthorized remote access |
| `u2r` | User-to-Root — privilege escalation |

---

## 🗂️ Dataset

**Inspired by:** [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — University of New Brunswick

The dataset used in this project is **synthetically generated at runtime** to mirror the statistical distributions of NSL-KDD. No external download is required. 5,000 records are created with the following class split:

| Class | Proportion | Count |
|-------|-----------|-------|
| Normal | 45% | 2,250 |
| DoS | 35% | 1,750 |
| Probe | 12% | 600 |
| R2L | 5% | 250 |
| U2R | 3% | 150 |

To use the **real NSL-KDD dataset**, download it from:  
🔗 https://www.unb.ca/cic/datasets/nsl.html

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Web Framework | [Streamlit](https://streamlit.io/) ≥ 1.35.0 |
| Machine Learning | [Scikit-learn](https://scikit-learn.org/) ≥ 1.4.0 |
| Data Manipulation | [Pandas](https://pandas.pydata.org/) ≥ 2.1.0 · [NumPy](https://numpy.org/) ≥ 1.26.0 |
| Visualisation | [Matplotlib](https://matplotlib.org/) ≥ 3.8.0 |

### ML Models

- **Random Forest** — 100 estimators; handles class imbalance well; provides feature importances
- **Decision Tree** — max depth 12; fast, interpretable
- **Logistic Regression** — linear baseline with StandardScaler pipeline

---

## ⚙️ Setup & Run Instructions

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd ml-network-intrusion-detector
```

Or simply place `app.py` and `requirements.txt` in the same folder.

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
ml-network-intrusion-detector/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📊 Key Results (on synthetic data)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~99% | ~99% | ~99% | ~99% |
| Decision Tree | ~99% | ~99% | ~99% | ~99% |
| Logistic Regression | ~85–90% | ~85–90% | ~85–90% | ~85–90% |

> High accuracy is expected because the synthetic data has clearly separated per-class distributions. Results on real-world data would differ.

---

## ⚠️ Limitations

- Synthetic data only — does not represent real network captures
- No live packet sniffing or network interface access
- NSL-KDD patterns are based on 1999 traffic — modern attacks differ significantly
- Not suitable for production deployment

---

## 🚀 Future Enhancements

- [ ] Integrate real NSL-KDD or CIC-IDS2017 dataset
- [ ] CSV batch upload for bulk predictions
- [ ] SQLite prediction history logging
- [ ] SHAP explainability charts
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add LSTM / Autoencoder for anomaly-based detection

---

## 📄 License

This project is for **academic and educational purposes only**.

---

**Project:** BCA Data Science — ML Network Intrusion Detector  
**Author:** RAJASHREE C  
**Stack:** Python · Streamlit · Scikit-learn · Pandas · Matplotlib
