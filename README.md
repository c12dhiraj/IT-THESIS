# 🧠 Optimized AI-Based Threat Detection System Using Artificial Neural Networks

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Overview
This project presents an **AI-driven Intrusion Detection System (IDS)** that leverages **Artificial Neural Networks (ANNs)** to identify and mitigate cyber threats.  
By analyzing network flow data, the system classifies activities as benign or malicious.  
The ANN model was optimized through **feature selection, normalization, and hyperparameter tuning**, leading to high accuracy and generalization.

---

## 🚀 Features
- End-to-end data preprocessing (encoding, normalization, balancing)
- Artificial Neural Network (ANN) model implementation and optimization
- Performance comparison with traditional ML classifiers
- Visualization of Accuracy, Precision, Recall, and F1-score
- Feature importance analysis for interpretability
- Modular and scalable notebook design for deployment

---

## 🧩 Project Workflow
1. **Data Preparation**
   - Data cleaning and transformation  
   - Encoding categorical features and normalization  
   - Handling class imbalance using resampling techniques  

2. **Model Development**
   - ANN architecture with optimized hidden layers and activation functions  
   - Hyperparameter tuning (epochs, batch size, learning rate)  

3. **Evaluation**
   - Computed Accuracy, Precision, Recall, F1-score  
   - Visualized Confusion Matrix and ROC Curve  

4. **Optimization**
   - Fine-tuned model for better generalization  
   - Identified key features influencing intrusion detection  

---

## 📊 Results
The optimized ANN achieved:
- **High Accuracy and F1-score**
- **Strong detection of minority attack classes**
- **Top contributing features:** `source bytes`, `destination port`, `duration`, `service`, and `protocol`

---

## 📸 Sample Output
Below are example visualizations generated during training and evaluation:

| Confusion Matrix | Feature Importance |
|------------------|--------------------|
| ![Confusion Matrix](outputs/confusion_matrix.png) | ![Feature Importance](outputs/feature_importance.png) |

## 🧠 Technologies Used
- **Language:** Python  
- **Libraries:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 📁 File Structure
📂 Optimized-AI-Based-Threat-Detection
│
├── 📜 Optimized AI-Based Threat Detection System Using Artificial Neural Networks.ipynb
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📊 dataset.csv # (if applicable)
└── 📈 outputs/ # model performance charts and graphs


---

## ⚙️ How to Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/Optimized-AI-Based-Threat-Detection.git
   cd Optimized-AI-Based-Threat-Detection


Install Dependencies

pip install -r requirements.txt


Run the Notebook

jupyter notebook "Optimized AI-Based Threat Detection System Using Artificial Neural Networks.ipynb"


Reproduce Results

Execute all notebook cells sequentially

View metrics, plots, and evaluation results
