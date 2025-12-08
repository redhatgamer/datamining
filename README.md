# Machine Learning Analysis – Product Sales

A complete data mining workflow including:

- Data preprocessing  
- K-means clustering  
- Linear & polynomial regression  
- Visualizations & actionable insights

---

# 🚀 Quick Start (Automatic)

Run the setup script (macOS / Linux / Windows supported):

```bash
python3 setup.py
```

This will:

- Create a virtual environment  
- Install all dependencies from `requirements.txt`  
- Launch Jupyter Notebook automatically  

---

# 🧠 Manual Setup (Optional)

### 1. Create & activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook ML_Analysis.ipynb
```

---

# 📁 Project Structure

```
datamining/
│
├── ML_Analysis.ipynb      # Main notebook
├── product_sales.csv      # Dataset
│
├── preprocessing.py       # Data preprocessing utilities
├── kmeans.py              # Clustering logic
├── regression.py          # Regression models
├── visualization.py       # Charts & plotting helpers
│
├── requirements.txt
└── setup.py               # Automatic installer
```

---

# 📊 Output

- Cleaned dataset (missing values, outliers handled)
- Normalized features
- K-means clustering with optimal K chosen via elbow method
- Regression models (linear + polynomial)
- Matplotlib & Seaborn visualizations
- Business insights:
  - Revenue clusters
  - Profit trends
  - Actionable recommendations

