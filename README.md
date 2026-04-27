# Neural Digit Engine 🧠⚡

A machine learning project that combines Python-based model training with a custom inference workflow, demonstrating applied AI concepts and system-level understanding.

---

## 🚀 Overview

This project focuses on building a digit recognition system using a trained model and executing predictions efficiently.

### Key Components:

1. **Model Training**

   * Built using Python and deep learning libraries
   * Trained on digit datasets
   * Weights saved for reuse

2. **Inference Execution**

   * Model inference executed through optimized pipeline
   * Demonstrates how trained models are used in real applications

3. **System Design**

   * Organized modular structure (engine, backend, frontend)
   * Separation of training and inference logic

---

## 🏗️ Project Structure

```
neural-digit-engine/
│── backend/
│── frontend/
│── engine/
│── data/
│── weights/
│── train.py
│── requirements.txt
│── Dockerfile
```

---

## ⚙️ Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/aadity-dev/neural-digit-engine.git
cd neural-digit-engine
```

---

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Training Script

```bash
python train.py
```

---

## 🐳 Docker Setup

### Build Docker Image

```bash
docker build -t digit-engine .
```

### Run Container

```bash
docker run digit-engine
```

---

## 📸 Screenshots

(Add your Docker screenshots here if needed)

---

## 🧠 Key Learnings

* Machine Learning workflow (training → inference)
* Handling dependencies using Docker
* Containerizing ML applications
* Debugging real-world environment issues

---

## 🎯 Conclusion

This project demonstrates how machine learning models can be structured, executed, and containerized for reproducibility and deployment.

---

## 👨‍💻 Author

Aditya
