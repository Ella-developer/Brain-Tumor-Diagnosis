````markdown
# 🧠 Brain Tumor Detection with CNN & PyTorch

A deep learning-based system to automatically detect brain tumors from MRI scans using a custom Convolutional Neural Network (CNN) built in PyTorch.

---

## 🚀 Overview

This project leverages the power of CNNs to classify brain MRI images as **tumor** or **no tumor** with high accuracy. Manual diagnosis of brain tumors can be time-consuming and prone to human error. This project aims to support early diagnosis and improve efficiency using deep learning.

---

## 🧠 Model Architecture

The CNN model has the following structure:

- 4 Convolutional layers with increasing filter sizes (8 → 16 → 32 → 64)
- MaxPooling after each convolutional block
- Fully connected (dense) layers:
  - FC1: 12,544 → 100 neurons with ReLU and Dropout
  - FC2: 100 → 2 output classes
- Final layer: `LogSoftmax` for classification

![CNN Model Visualization](cnn_model_visualization.png)

---

## 📊 Performance

- ✅ **F1 Score**: 0.98  
- ✅ **Accuracy**: 98%

Evaluated on a validation set of labeled MRI scans.

---

## 📁 Project Structure

```text
brain-tumor-detection/
├── brain-tumor-detection-by-cnn-pytorch.ipynb   # Jupyter notebook with full training pipeline
├── cnn_model_visualization.png                  # Model architecture graph
├── Presentation.pdf                             # Final presentation slides
├── dataset/                                     # Folder containing brain MRI images
│   ├── yes/                                     # MRI images with tumor
│   └── no/                                      # MRI images without tumor
└── README.md
````

---

## ⚙️ Tech Stack

* `PyTorch` - Deep learning framework
* `Matplotlib`, `Seaborn` - Visualizations
* `sklearn.metrics` - Evaluation metrics
* `Google Colab` or `Jupyter Notebook`

---

## 📦 Dependencies

Install via pip:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

---

## 📈 How to Use

1. Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Run the notebook:

```bash
jupyter notebook brain-tumor-detection-by-cnn-pytorch.ipynb
```

3. Or run on Google Colab:

> Upload the notebook and dataset, and execute all cells.

---

## 📌 Results & Insights

* Model is effective in distinguishing tumor vs non-tumor MRI with very few misclassifications.
* The simplicity of the architecture makes it ideal for fast inference on low-resource systems.
* Further enhancement could involve segmentation or multi-class tumor classification.

---

## 🙋 Author

👩‍💻 **Ella K.**


---

## 📜 License

MIT License - Free for educational and commercial use.

```

