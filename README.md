# **AlzheimerClassifierTL: ResNet50-Based Alzheimer’s MRI Classifier**

AlzheimerClassifierTL is a deep learning project that uses **transfer learning with ResNet50** to classify **brain MRI scans** into four cognitive health categories:  
🟢 **NonDemented** • 🟡 **Very Mild Demented** • 🟠 **Mild Demented** • 🔴 **Moderate Demented**  

It demonstrates a complete **end-to-end computer vision workflow** including **data exploration, augmentation, ResNet50 fine-tuning, evaluation, and deployment with Streamlit & Hugging Face**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-transferlearning-resnet50-activity-7374182255985946624-P-8j?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://alzheimerclassifiertl-otxrv2bgjegphrzmwmupqb.streamlit.app/)  
- 🤗 [Explore on Hugging Face](https://huggingface.co/RawanAlwadeya/AlzheimerClassifierTL)  
- 📓 [Kaggle Notebook](https://www.kaggle.com/code/rawanalwadeya/alzheimerclassifiertl-resnet50-transfer-learning)

![App Demo](./Alzheimer%20Detection%20App.png)  
![No Dementia Example](./No%20Dementia.png)  
![Very Mild Dementia Example](./Very%20Mild%20Dementia.png)  
![Mild Dementia Example](./Mild%20Dementia.png)  
![Moderate Dementia Example](./Moderate%20Dementia.png)

---

## **Project Overview**

**AlzheimerClassifierTL** is a deep learning project that classifies **brain MRI images** into four cognitive health categories:  
- 🟢 **NonDemented**  
- 🟡 **Very Mild Demented**  
- 🟠 **Mild Demented**  
- 🔴 **Moderate Demented**  

The workflow includes **data exploration, visualization, and transfer learning** using **ResNet50**, followed by deployment as an interactive **Streamlit application** and publication on **Hugging Face**.

---

## **Objective**

Develop and deploy a reliable **transfer learning model** to support the early detection of **Alzheimer’s disease**, a progressive neurological disorder that impacts **memory and cognitive function**.  
Early identification of dementia stages can help guide timely medical consultation and care.

---

## **Dataset**

- **Source**: [Kaggle – Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data)  
- **Preprocessing**:  
  - Images resized to **224×224 (RGB)**  
  - Normalized using **ResNet50 preprocess_input**  
  - Split into **train, validation, and test sets** with balanced distribution  

---

## **Modeling Approach**

- **Base Model**: **ResNet50** pre-trained on ImageNet, with top layers removed and a global max pooling layer added  
- **Custom Layers**: Batch normalization, fully connected dense layer (256 units, L1/L2 regularization), and dropout (0.45)  
- **Output Layer**: Dense layer with softmax activation for 4-class prediction  
- **Training Setup**:  
  - Loss: **Categorical Crossentropy**  
  - Optimizer: **Adamax** (learning rate 0.001)  
  - Data augmentation (rotation, width/height shift, zoom, nearest-neighbor fill) applied to training set only  
  - EarlyStopping and ModelCheckpoint callbacks to prevent overfitting  

---

## **Performance**

The final model achieved strong results on the test set:  
- **Accuracy**: `98.3%`  
- **Precision**: `98.9%`  
- **Recall**: `93.8%`  
- **F1 Score**: `96.1%`

---

## **Deployment**

Users can upload an MRI image to get **real-time, color-coded predictions** for dementia stage.

- **Streamlit App**: [AlzheimerClassifierTL](https://alzheimerclassifiertl-otxrv2bgjegphrzmwmupqb.streamlit.app/)  
- **Hugging Face Repo**: [AlzheimerClassifierTL](https://huggingface.co/RawanAlwadeya/AlzheimerClassifierTL)

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- TensorFlow / Keras, scikit-learn  
- Matplotlib, Seaborn  
- Streamlit (Deployment)

**Techniques**:  
- Transfer Learning (ResNet50 fine-tuning)  
- Data Augmentation (rotation, width/height shift, zoom)  
- EarlyStopping & ModelCheckpoint  
- Real-time deployment with **Streamlit** & **Hugging Face**
