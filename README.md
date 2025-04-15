# Image-Sentiment-Analysis-With-PyTorch
We plan to train a model that can recognize a person's emotion by viewing an image of their face. In addition, we will fine-tune its weights using a pre-trained model in PyTorch to adapt the model to our dataset.

# 🎭 Facial Emotion Recognition with ResNet18

This project implements a deep learning pipeline to classify facial expressions into seven emotions using a **pre-trained ResNet18** model. The model is built using **PyTorch** and trained on grayscale facial images (48x48 pixels).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Preprocessing & Augmentation](#-preprocessing--augmentation)
- [Training Procedure](#-training-procedure)
- [Evaluation & Visualization](#-evaluation--visualization)
- [How to Use](#-how-to-use)
- [Installation](#-installation)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

Facial expression recognition is a key component of many modern AI systems, including:

- Human-computer interaction
- Security systems
- Mental health analysis

This project recognizes **7 emotion categories**:

- Anger 😠  
- Disgust 🤢  
- Fear 😨  
- Happiness 😄  
- Sadness 😢  
- Surprise 😲  
- Neutral 😐  

---

## 📁 Dataset

The dataset consists of flattened grayscale facial images, stored in compressed `.parquet.gzip` files:

- `df_train.parquet.gzip`: Labeled images for training and validation  
- `df_test.parquet.gzip`: Unlabeled test images for final predictions  

Each image is:

- 48x48 pixels  
- Stored as a 1D NumPy array  
- Labeled using integer class indices (`0–6`)  

Emotion labels are mapped as:

| Label Index | Emotion     |
|-------------|-------------|
| 0           | Anger       |
| 1           | Disgust     |
| 2           | Fear        |
| 3           | Happiness   |
| 4           | Sadness     |
| 5           | Surprise    |
| 6           | Neutral     |

---

## 🧠 Model Architecture

The model uses a modified version of **ResNet18**:

- Pretrained on ImageNet  
- Final layers replaced with custom fully connected layers:  
  - `512 → 100 → 7` (softmax output)

### Forward Pass

```
Input → ResNet18 Backbone → FC(512 → 100) → ReLU → FC(100 → 7) → Softmax
```

---

## 🧪 Preprocessing & Augmentation

- Images are reshaped to 48×48 and converted to `PIL` format.  
- Training images are augmented with:
  - Random rotation (±30°)
  - Normalization (mean & std computed from dataset)  
- All images are resized to **224×224** (ResNet18 input size).  
- Grayscale images are repeated across **3 channels**.

---

## 🏋️ Training Procedure

- **Optimizer**: Adam  
- **Learning Rate**: `1e-4`  
- **Loss Function**: CrossEntropyLoss  
- **Epochs**: `30`  
- **Batch Size**: `128`  

Each training epoch:

- Computes forward and backward pass  
- Tracks running loss  
- Evaluates on validation set  

Device selection is automatic: uses **GPU if available**.

---

## 📊 Evaluation & Visualization

After training:

- Validation **accuracy and loss** are reported.  
- A **confusion matrix** shows class-wise performance.  
- A bar chart visualizes emotion distribution in the dataset.  
- Random image predictions (with ground truth) are displayed.

---

## 🧰 How to Use

1. Place the following files in your working directory:
    - `df_train.parquet.gzip`
    - `df_test.parquet.gzip`

2. Run the Python script to:
    - Load and preprocess data  
    - Train and validate the model  
    - Generate test predictions  

**Outputs**:
- Evaluation accuracy/loss logs  
- Confusion matrix plot  
- Final test predictions (`test_prediction` list)

---

## 🛠️ Installation

Make sure you have Python 3.x and install the required libraries:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow tqdm
```

---

## 📈 Results

- Training and validation logs show model performance over time.  
- Confusion matrix highlights which emotions are most/least confused.  
- Sample visualization demonstrates prediction quality.

You can improve accuracy by:

- Using a larger or more diverse dataset  
- Fine-tuning deeper models  
- Applying more sophisticated augmentation techniques

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Fork the repository  
- Add features or improve performance  
- Fix bugs or update documentation  
- Submit a pull request


