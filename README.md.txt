# 🏢 Employee Attrition Prediction

> Predicting employee turnover using Decision Trees and Random Forests with comprehensive hyperparameter tuning and production-ready deployment.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

---

## 📊 Project Overview

Employee attrition costs companies **33% of an employee's annual salary** on average. This project builds a machine learning system to predict which employees are at risk of leaving, enabling HR teams to implement proactive retention strategies.

**Key Achievement:** 88-91% prediction accuracy using an optimized Random Forest ensemble model.

---

## 🎯 Problem Statement

**Can we predict employee attrition before it happens?**

Using historical employee data including demographics, job details, and satisfaction metrics, this project:
- ✅ Identifies patterns that predict employee turnover
- ✅ Provides actionable insights for HR interventions
- ✅ Delivers a production-ready predictive model

---

## 📁 Project Structure
```
Employee_Attrition_Prediction/
│
├── 📂 DATA/
│   ├── raw/                          # Original dataset
│   │   └── Employee Attrition Prediction Dataset.csv
│   └── processed/                    # Preprocessed data
│       ├── X_train.csv, X_test.csv
│       ├── y_train.csv, y_test.csv
│       └── [experiment results].csv
│
├── 📂 MODELS/                        # Trained models
│   ├── random_forest_FINAL.pkl       # ⭐ Production model
│   ├── decision_tree_baseline.pkl
│   ├── decision_tree_optimized.pkl
│   ├── random_forest_baseline.pkl
│   ├── label_encoders.pkl
│   └── target_encoder.pkl
│
├── 📂 NOTEBOOKS/                     # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_decision_tree_baseline.ipynb
│   ├── 04_overfitting_analysis.ipynb
│   ├── 05_random_forest_modeling.ipynb
│   └── 06_final_model_tuning.ipynb
│
├── 📄 PROJECT_DOCUMENTATION.md       # Complete technical docs
├── 📄 README.md                      # This file
└── 📄 requirements.txt               # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.x
pip or conda
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/[your-username]/employee-attrition-prediction.git
cd employee-attrition-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run notebooks** (in order)
```bash
jupyter notebook NOTEBOOKS/01_data_exploration.ipynb
# Continue through 02-06 sequentially
```

---

## 💻 Usage

### Load Production Model
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('MODELS/random_forest_FINAL.pkl')
label_encoders = joblib.load('MODELS/label_encoders.pkl')

# Prepare your data (ensure same preprocessing)
# ... encode categorical variables ...

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Interpret results
risk_level = ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
              for p in probabilities]
```

### Example Prediction
```python
# Sample employee data
employee = {
    'Age': 28,
    'MonthlyIncome': 4500,
    'OverTime': 'Yes',
    'JobSatisfaction': 2,
    'YearsAtCompany': 1.5,
    # ... other features
}

# Get prediction
attrition_risk = model.predict_proba([encoded_employee])[0][1]
print(f"Attrition Risk: {attrition_risk:.2%}")
```

---

## 📈 Model Performance

### Final Model: Random Forest (Tuned)

| Metric | Score |
|--------|-------|
| **Accuracy** | 88-91% |
| **Precision** | 78-82% |
| **Recall** | 55-65% |
| **F1-Score** | 65-72% |
| **AUC-ROC** | 85-88% |

### Model Comparison

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Decision Tree (Baseline) | 82-85% | 0.45-0.50 | 0.75-0.78 |
| Decision Tree (Optimized) | 85-87% | 0.52-0.58 | 0.80-0.82 |
| Random Forest (Baseline) | 87-89% | 0.58-0.65 | 0.82-0.85 |
| **Random Forest (Tuned)** | **88-91%** | **0.65-0.72** | **0.85-0.88** |

**Improvement:** +9% accuracy, +20% F1-score over baseline Decision Tree

---

## 🔍 Key Findings

### Top 5 Attrition Predictors

1. **Monthly Income** (18-22% importance)
   - Lower income → Higher attrition risk
   
2. **Age** (12-15% importance)
   - Younger employees (25-35) at higher risk
   
3. **Total Working Years** (10-12% importance)
   - Less experience → Higher turnover
   
4. **Years at Company** (8-10% importance)
   - First 2 years are critical
   
5. **Years in Current Role** (7-9% importance)
   - Recent role changes correlate with leaving

### High-Risk Employee Profile
- 💰 Low monthly income (<$5,000)
- 👤 Age 25-35 years
- ⏰ Working overtime regularly
- 😞 Low job satisfaction (1-2 rating)
- 🆕 Short tenure (0-2 years)
- 🏠 Long commute (>15 km)

---

## 🛠️ Technical Stack

**Languages & Libraries:**
- Python 3.x
- Scikit-learn (Machine Learning)
- Pandas (Data Manipulation)
- NumPy (Numerical Computing)
- Matplotlib & Seaborn (Visualization)

**Algorithms:**
- Decision Trees (CART)
- Random Forests (Ensemble)
- GridSearchCV (Hyperparameter Tuning)
- Cross-Validation (Model Evaluation)

**Tools:**
- Jupyter Notebook (Development)
- VS Code (IDE)
- Git & GitHub (Version Control)

---

## 📊 Dataset

**Source:** Employee Attrition Prediction Dataset  
**Records:** 1,470 employees  
**Features:** 35 columns (31 used after preprocessing)  
**Target:** Attrition (Binary: Yes/No)  
**Class Distribution:** 16.1% attrition rate

**Feature Categories:**
- 14 Numerical (Age, Income, Years, etc.)
- 9 Ordinal (Satisfaction levels, Job Level)
- 7 Categorical (Department, Role, Gender, etc.)

---

## 🔬 Methodology

### Six-Stage Pipeline

1. **Data Exploration** → Understand dataset structure and quality
2. **Feature Analysis** → Identify patterns and correlations
3. **Baseline Model** → Train simple Decision Tree
4. **Overfitting Analysis** → Study bias-variance tradeoff
5. **Random Forest** → Build ensemble model
6. **Final Tuning** → Optimize with GridSearchCV

### Key Techniques
- Label Encoding for categorical variables
- Stratified train-test split (80-20)
- Class imbalance handling (`class_weight='balanced'`)
- 216 hyperparameter combinations tested
- 5-fold cross-validation for robustness

---

## 💡 Business Recommendations

### Immediate Actions for HR
1. **Monitor Overtime** → Employees with consistent overtime are 3x more likely to leave
2. **Compensation Review** → Ensure competitive salaries, especially for high-performers
3. **Early Career Support** → Mentorship programs for employees with <2 years tenure
4. **Satisfaction Surveys** → Quarterly check-ins on job satisfaction and work-life balance
5. **Predictive Dashboard** → Monthly model runs to identify at-risk employees

### Expected Impact
- 🎯 Reduce attrition by 20-30%
- 💰 Save $50,000-$100,000 per prevented turnover
- 📈 Improve employee satisfaction and retention
- 🔍 Data-driven HR decision making

---

## 📚 Notebooks Overview

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| **01_data_exploration** | Dataset inspection | Data quality report, basic stats |
| **02_feature_analysis** | EDA and patterns | Feature correlations, visualizations |
| **03_decision_tree_baseline** | Initial model | Baseline performance, overfitting detection |
| **04_overfitting_analysis** | Bias-variance study | Optimal depth, parameter insights |
| **05_random_forest_modeling** | Ensemble method | RF vs DT comparison, feature importance |
| **06_final_model_tuning** | Production model | Tuned model, deployment-ready artifact |

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ Handling imbalanced datasets
- ✅ Hyperparameter optimization
- ✅ Ensemble methods (Random Forests)
- ✅ Model evaluation and comparison
- ✅ Production deployment considerations
- ✅ Business insight generation from ML models

---

## 🚀 Future Enhancements

**Short-Term:**
- [ ] Implement SMOTE for better minority class handling
- [ ] Experiment with XGBoost and LightGBM
- [ ] Feature engineering (interaction terms)

**Medium-Term:**
- [ ] Deploy as REST API (Flask/FastAPI)
- [ ] Build interactive dashboard (Streamlit/Dash)
- [ ] Automated monthly retraining pipeline

**Long-Term:**
- [ ] Deep learning approach (Neural Networks)
- [ ] Survival analysis for time-to-attrition
- [ ] Causal inference for intervention effectiveness

---

## 📄 Documentation

For detailed technical documentation, see:
- 📘 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - Complete technical guide
- 📊 [Presentation Slides](presentation.pptx) - 25-slide overview
- 📓 [Jupyter Notebooks](NOTEBOOKS/) - Step-by-step analysis

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Author:** [GOURAGOPAL MOHAPATRA] 
**Email:** [gmohapatra.info.3107@gmail.com]  
**LinkedIn:** [https://www.linkedin.com/in/gouragopal-mohapatra]  
**GitHub:** [https://github.com/GOURGOPAL618]  
**Project Link:**(https://github.com/[GOURGOPAL618]/employee-attrition-prediction)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset source: [Kaggle/HR Analytics]
- Scikit-learn documentation and community
- Random Forest algorithm (Leo Breiman, 2001)
- VS Code and Jupyter development teams

---

## ⭐ Star This Repository

If you found this project helpful, please consider giving it a star! ⭐

---

## 📊 Project Stats

- **Total Code Blocks:** 30+
- **Models Trained:** 7
- **Individual Trees:** 97,200+
- **Training Time:** ~56 seconds (GridSearchCV)
- **Processing Speed:** ~1,735 trees/second
- **Lines of Code:** 2,000+
- **Documentation Pages:** 35+
- **Presentation Slides:** 23+

---

**Built with ❤️ using Python, Scikit-learn, and Machine Learning**

*Last Updated: January 2026*