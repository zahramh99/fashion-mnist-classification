# Fashion MNIST Classifier

CNN-based image classifier for Fashion MNIST using TensorFlow/Keras.

## Features
- 2-layer CNN with dropout
- Early stopping & model checkpointing
- Training visualization
- Classification report generation

## Usage
```bash
python train.py --epochs 20 --batch_size 64

Results
Test Accuracy: ~91%

Includes confusion matrix and classification reports