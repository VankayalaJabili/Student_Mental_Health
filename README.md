# 🧠 Predictive and Proactive Student Mental Health Monitoring

## 📌 Overview

An AI-based web application that monitors student behavioral patterns and predicts mental health risk levels using machine learning and Explainable AI.

---

# 🎯 Objectives

- Monitor student behavioral patterns.
- Predict mental health risk.
- Explain predictions using SHAP.
- Provide personalized recommendations.
- Track historical progress.

---

# ✨ Features

## 📋 Daily Check-In

Collects information about:

- Sleep
- Phone usage
- Physical activity
- Social interaction
- Study/workload
- Mood and stress

## 🤖 Risk Prediction

Predicts four risk categories:

- Normal
- Mild
- Moderate
- Severe

## 🔍 Explainable AI

Uses SHAP to identify the behavioral factors influencing predictions.

## 💡 Recommendations

Provides personalized wellness suggestions based on the predicted risk and behavioral factors.

## 📊 Dashboard

Displays:

- Risk level
- Progress
- SHAP explanations
- Recommendations

---

# 🏗️ System Architecture

```text
User
  ↓
Next.js Frontend
  ↓
Node.js / API
  ↓
SQLite Database
  ↓
Flask AI Service
  ↓
XGBoost / DNN
  ↓
Prediction + SHAP
  ↓
Recommendations
