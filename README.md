# CLASSIFICATION-OF-FRACTURED-BONES-USING-MACHINE-LEARNING
Fractured Bones Using Machine Learning
Here’s a clean and professional **README.md** file for your GitHub project based on your uploaded files:

---

# 🦴 Classification of Fractured Bones Using Machine Learning

## 📌 Overview

This project focuses on classifying different types of bone images using a **Machine Learning algorithm (Random Forest)**. The system helps in identifying bone types such as chest, elbow, finger, hand, head, shoulder, and wrist, which can assist in fracture detection.

---

## 🎯 Objective

To build a simple and efficient system that:

* Classifies bone types from X-ray images
* Assists in identifying possible fractures
* Reduces manual effort in medical diagnosis

---

## 🧠 Technology Used

* Python
* Machine Learning (Random Forest)
* OpenCV (Image Processing)
* NumPy
* Scikit-learn
* Tkinter (GUI)

---

## 📂 Dataset

* The project uses a dataset containing **7 types of bone images**:

  * Chest
  * Elbow
  * Finger
  * Hand
  * Head
  * Shoulder
  * Wrist

* Each folder contains images of a specific bone type.

---

## ⚙️ Modules in the Project

### 1. Dataset Collection / Upload

* Upload dataset folder into the application.

### 2. Feature Extraction

* Extract RGB pixel values from images
* Resize images to **32×32**
* Convert images into feature vectors

### 3. Train-Test Split

* 80% data used for training
* 20% data used for testing

### 4. Model Building

* Random Forest Classifier is used
* Model is trained and saved

### 5. Prediction

* Upload a test image
* System predicts the bone type

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/bone-classification.git
cd bone-classification
```

### Step 2: Install Dependencies

```bash
pip install numpy opencv-python matplotlib scikit-learn
```

### Step 3: Run the Application

* Double click on `run.bat`
  **OR**

```bash
python BoneClassification.py
```

---

## 🖥️ Application Workflow

1. Upload Dataset
2. Click on **Feature Extraction**
3. Click on **Train & Test Data Generator**
4. Click on **Build Random Forest Model**
5. Upload test image to classify

---

## 📊 Output

* Displays:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
* Shows classified image with predicted bone label

---

## 📈 Results

* Achieved approximately **84% accuracy** on test dataset

---

## 📌 Features

✔ Simple GUI using Tkinter
✔ Fast training with Random Forest
✔ Easy dataset handling
✔ Real-time image classification

---

## ⚠️ Limitations

* Works on limited dataset
* Uses basic image features (RGB only)
* Accuracy can be improved with deep learning

---

## 🔮 Future Enhancements

* Use CNN (Deep Learning) for better accuracy
* Add fracture detection (not just classification)
* Improve GUI design
* Use larger medical datasets

---

## 👨‍💻 Author

* Your Name

---

## 📜 License

This project is for educational purposes only.

---

If you want, I can also:
✅ Customize it with your **GitHub username & screenshots**
✅ Add **badges (accuracy, Python version, etc.)**
✅ Convert it into a **professional README with images & GIFs**
