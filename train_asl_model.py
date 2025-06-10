import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from tqdm import tqdm
import logging
from collections import Counter
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleASLPreprocessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def extract_hand_landmarks(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks, dtype=np.float32)
            return np.zeros(63, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error processing {image_path}: {e}")
            return None
    
    def normalize_landmarks(self, landmarks):
        if landmarks is None or len(landmarks) == 0 or np.all(landmarks == 0):
            return np.zeros(63, dtype=np.float32)
        reshaped = landmarks.reshape(21, 3)
        wrist = reshaped[0]
        normalized = reshaped - wrist
        return normalized.flatten()

class KaggleASLDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.preprocessor = KaggleASLPreprocessor()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, limit_per_class=None):
        X, y = [], []
        train_path = os.path.join(self.dataset_path, 'asl_alphabet_train')
        classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        logger.info(f"Found {len(classes)} classes: {classes}")
        
        for class_name in classes:
            class_path = os.path.join(train_path, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if limit_per_class:
                image_files = image_files[:limit_per_class]
            
            logger.info(f"Processing class '{class_name}': {len(image_files)} images")
            
            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                image_path = os.path.join(class_path, image_file)
                landmarks = self.preprocessor.extract_hand_landmarks(image_path)
                
                if landmarks is not None and not np.all(landmarks == 0):
                    normalized_landmarks = self.preprocessor.normalize_landmarks(landmarks)
                    X.append(normalized_landmarks)
                    y.append(class_name)
                else:
                    logger.warning(f"Skipped {image_path}: No valid landmarks detected")
        
        if len(X) == 0:
            raise ValueError("No valid data loaded. Check dataset or preprocessing.")
        
        return np.array(X), np.array(y)
    
    def prepare_sequences(self, X, sequence_length=10):
        X_sequences = []
        for landmarks in X:
            sequence = []
            reshaped = landmarks.reshape(21, 3)
            for _ in range(sequence_length):
                scale = np.random.uniform(0.95, 1.05)
                scaled = reshaped * scale
                theta = np.random.uniform(-0.1, 0.1)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                rotated = np.dot(scaled, rotation_matrix)
                noise = np.random.normal(0, 0.005, rotated.shape)
                augmented = rotated + noise
                sequence.append(augmented.flatten())
            X_sequences.append(sequence)
        return np.array(X_sequences)

class ASLClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.4),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.4),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
            ModelCheckpoint('model_outputs/best_asl_model.keras', save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss')
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def evaluate_model(self, X_test, y_test, class_names):
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes, target_names=class_names)
        logger.info("Classification Report:")
        logger.info(f"\n{report}")
        
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('model_outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        return report, cm
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train ASL Alphabet model")
    parser.add_argument('--dataset_path', default='asl_alphabet_dataset', type=str)
    parser.add_argument('--sequence_length', default=10, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--limit_per_class', default=1000, type=int, nargs='?')
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    SEQUENCE_LENGTH = args.sequence_length
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LIMIT_PER_CLASS = args.limit_per_class
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path {DATASET_PATH} does not exist")

    os.makedirs('model_outputs', exist_ok=True)
    logger.info("Starting ASL Alphabet training pipeline...")

    logger.info("Loading and preprocessing dataset...")
    loader = KaggleASLDataLoader(DATASET_PATH)
    X, y = loader.load_data(limit_per_class=LIMIT_PER_CLASS)

    logger.info(f"Loaded {len(X)} samples")
    logger.info(f"Feature shape: {X[0].shape}")
    logger.info(f"Classes: {sorted(set(y))}")

    class_counts = Counter(y)
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('model_outputs/class_distribution.png')
    plt.show()

    y_encoded = loader.label_encoder.fit_transform(y)
    class_names = loader.label_encoder.classes_

    with open('model_outputs/label_encoder.pkl', 'wb') as f:
        pickle.dump(loader.label_encoder, f)

    logger.info("Creating sequence data...")
    X_sequences = loader.prepare_sequences(X, SEQUENCE_LENGTH)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_sequences, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE, stratify=y_temp)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")

    if X_train.shape[2] != 63:
        raise ValueError(f"Expected 63 features, got {X_train.shape[2]}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(class_names)

    classifier = ASLClassifier(input_shape, num_classes)
    model = classifier.build_model()
    logger.info("Model architecture:")
    model.summary()

    logger.info("Starting training...")
    history = classifier.train(
        X_train, y_train, X_val, y_val, 
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    classifier.plot_training_history(history)
    model.load_weights('model_outputs/best_asl_model.keras')

    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    report, cm = classifier.evaluate_model(X_test, y_test, class_names)

    model.save('model_outputs/final_asl_model.keras')
    model.save('model_outputs/final_asl_model.h5')

    config = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'sequence_length': SEQUENCE_LENGTH,
        'class_names': class_names.tolist(),
        'test_accuracy': float(test_accuracy),
        'feature_count': X_train.shape[2]
    }

    with open('model_outputs/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Training completed successfully!")
    logger.info(f"Final test accuracy: {test_accuracy:.4f}")
    logger.info("Model saved in 'model_outputs/' directory")

if __name__ == "__main__":
    main()