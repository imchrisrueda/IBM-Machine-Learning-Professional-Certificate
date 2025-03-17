# Employee Attrition Prediction Analysis

## 🚀 Project Overview
This repository contains a comprehensive analysis aimed at predicting employee attrition using advanced classification models. By leveraging machine learning, this project helps HR departments proactively identify employees at risk, enabling strategies to improve retention and organizational stability.

## 🎯 Objective
The main goal is to build and evaluate predictive models that accurately identify potential employee attrition, providing actionable insights into key attrition factors.

## 📂 Dataset
- **Source:** IBM Employee Attrition Dataset (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset?resource=download) 
- **Size:** 1,470 records
- **Features:** Includes demographic information, job-related details, compensation metrics, and employee satisfaction indicators.

## 🛠️ Methodology
The analysis covers:
- Data exploration and preprocessing (cleaning, scaling, encoding)
- Feature engineering to enhance predictive capabilities
- Training and evaluation of multiple classifiers:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (XGBoost)

## 📈 Results
| Classifier             | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 88%      | 71%       | 42%    | 0.53     |
| Random Forest          | 84%      | 52%       | 17%    | 0.26     |
| XGBoost                | 85%      | 59%       | 27%    | 0.37     |

- Recommended Model: **Logistic Regression** (high interpretability and balanced performance).

## 🔑 Key Insights
- Major factors driving attrition include:
  - Monthly income and financial incentives
  - Employee age, tenure, and total working years
  - Departmental roles (e.g., Human Resources) and overtime policies
- Attrition is more likely among employees with shorter tenure or in specific job roles.

## ⚙️ Repository Structure
```
project/
├── data/                           # Dataset files
├── report_employees.ipynb          # Jupyter notebooks with analysis
└── README.md                       # Project overview
```

## 🛠️ How to Run
### Clone Repository
```bash
git clone <your_repository_url>
cd employee-attrition-prediction
```

### Set Up Environment
Ensure you have Anaconda installed:
```bash
conda env create -f environment.yml
conda activate attrition_analysis_env
```

### Execute Analysis
Run the notebooks or Python scripts provided:
```bash
jupyter notebook
```

## 📝 Next Steps
- Address class imbalance using advanced techniques (SMOTE, undersampling)
- Integrate qualitative insights from surveys or employee feedback
- Continuously update and retrain models with fresh data

## 📬 Contact
For further information or collaboration, feel free to reach me out.


## 🚧 Contributions
Contributions, issues, and feature requests are welcome!

---

🌟 Made with passion for Data Science and HR Analytics 🌟
