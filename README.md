# Fashion MNIST Image Classifier with CNN
## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify clothing items from the Fashion MNIST dataset. The model achieves **~91% test accuracy** with capabilities for:
- Image classification
- Training visualization
- Performance metrics generation
- Model checkpointing

## 📊 Dataset: Fashion MNIST

### Dataset Description
Fashion MNIST is a benchmark dataset containing **70,000 grayscale images** (28x28 pixels) across **10 fashion categories**:

| Label | Class       | Description          |
|-------|-------------|----------------------|
| 0     | T-shirt/top | 👕 T-shirts, tops     |
| 1     | Trouser     | 👖 Pants, trousers   |
| 2     | Pullover    | 🧥 Sweaters          |
| 3     | Dress       | 👗 Dresses           |
| 4     | Coat        | 🧥 Jackets, coats    |
| 5     | Sandal      | 👡 Sandals           |
| 6     | Shirt       | 👔 Button-up shirts  |
| 7     | Sneaker     | 👟 Athletic shoes    |
| 8     | Bag         | 👜 Handbags          |
| 9     | Ankle boot  | 👢 Boots             |

**Dataset Split:**
- **Training:** 60,000 images
- **Testing:** 10,000 images

### Key Characteristics
- **Grayscale** (1 channel)
- **Low-resolution** (28×28 pixels)
- **Balanced classes** (6,000 images per class in training set)
- **Pre-processed** (centered, normalized)
