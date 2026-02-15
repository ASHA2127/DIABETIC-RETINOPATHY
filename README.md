# 🩺 Diabetic Retinopathy Detection Using Deep Learning

## 🎓 MCA Final Year Project

---

## 📖 Abstract

Diabetic Retinopathy (DR) is one of the leading causes of blindness among diabetic patients worldwide. Early detection and treatment can prevent severe vision loss. 

This project presents a Deep Learning-based system for automatic detection and classification of Diabetic Retinopathy using retinal fundus images. A Convolutional Neural Network (CNN) model is trained to classify images into different severity stages of the disease.

The system is integrated into a Flask-based web application to allow users to upload retinal images and receive instant predictions.

---

## 🎯 Objectives

- To develop a Deep Learning model for detecting Diabetic Retinopathy.
- To classify retinal images into different severity levels.
- To build a user-friendly web interface for image upload and prediction.
- To assist in early diagnosis using AI-based automation.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Flask (Web Framework)
- HTML, CSS
- SQLite (Database)

---

## 🏗️ System Architecture

1. Image Upload through Web Interface  
2. Image Preprocessing (Resizing, Normalization)  
3. CNN Model Prediction  
4. Display Result to User  

---

## 📂 Project Structure

```
DIABETIC-RETINOPATHY/
│
├── app.py              # Main Flask application
├── cnn.py              # CNN model architecture
├── static/             # CSS and image assets
├── templates/          # HTML templates
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Execution

### Step 1: Clone Repository

```
git clone https://github.com/ASHA2127/DIABETIC-RETINOPATHY.git
```

### Step 2: Navigate to Project Folder

```
cd DIABETIC-RETINOPATHY
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

### Step 4: Run Application

```
python app.py
```

### Step 5: Open in Browser

```
http://127.0.0.1:5000/
```

---

## 🏥 Dataset Description

The dataset consists of retinal fundus images categorized into:

- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

(Note: Dataset is not included in the repository due to large file size.)

---

## 📊 Model Details

- Model Type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Evaluation Metric: Accuracy

---

## 🚀 Key Features

- Automated disease detection
- Multi-class classification
- Web-based interface
- AI-assisted diagnosis support

---

## 📚 Future Enhancements

- Improve accuracy using transfer learning (ResNet / VGG)
- Deploy the model on cloud platform
- Add real-time image capture support
- Mobile application integration

---

## 👩‍💻 Developed By

Asha  
Master of Computer Applications (MCA)  
AI & Machine Learning Enthusiast  
Academic Year: 2025–2026
