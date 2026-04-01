import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

# Early check for PIL (Pillow)
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is not installed. Please install it using `pip install pillow`")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'datasets'  # Must be 'datasets' based on your folder structure inside mango
MODEL_DIR = 'model'
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'mango_model.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'mango_labels.json')

def prepare_data():
    """Sets up data generators, augmentation, and computes class weights."""
    print("Preparing data generators...")
    
    # Using ImageDataGenerator with 80/20 train/val split and augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Save Class Labels mappings
    os.makedirs(MODEL_DIR, exist_ok=True)
    class_indices = train_gen.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    
    with open(LABELS_PATH, 'w') as f:
        json.dump(class_labels, f, indent=4)
    print(f"Saved class labels mapping to {LABELS_PATH}")
    
    # Compute Class Weights due to potentially imbalanced datasets
    classes = train_gen.classes
    class_weight_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights = {i: weight for i, weight in enumerate(class_weight_arr)}
    
    return train_gen, val_gen, class_weights, len(class_indices)

def build_model(num_classes):
    """Builds the main deep learning model using transferring learning (EfficientNetB0)."""
    print("Building model...")
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    # Freeze the base model for initial training phase
    base_model.trainable = False 
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, base_model, train_gen, val_gen, class_weights):
    """Executes the training and fine-tuning phases."""
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    INITIAL_EPOCHS = 15
    print("--- Phase 1: Training top layers ---")
    history = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_list
    )
    
    print("--- Phase 2: Fine-Tuning base model ---")
    # Unfreeze the base model
    base_model.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Use a very low learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    FINE_TUNE_EPOCHS = 10
    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    
    history_fine = model.fit(
        train_gen,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_list
    )
    
    print("Training finished! Final model saved to: ", CHECKPOINT_PATH)
    return model

def predict_disease(img_path):
    """Utility function to predict on a single image (Production ready for FastAPI)."""
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("Model or labels not found. Ensure the model is trained first.")
        
    loaded_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    with open(LABELS_PATH, 'r') as f:
        labels_dict = json.load(f)
        
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Apply the same rescale factor used in training
    
    predictions = loaded_model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)
    confidence = float(predictions[predicted_class_index])
    
    predicted_class_name = labels_dict[str(predicted_class_index)]
    
    return {
        "disease": predicted_class_name,
        "confidence": round(confidence * 100, 2)
    }

if __name__ == '__main__':
    # 1. Setup Data
    train_gen, val_gen, class_weights, num_classes = prepare_data()
    
    # 2. Build Model
    model, base_model = build_model(num_classes)
    
    # 3. Train Model
    trained_model = train_model(model, base_model, train_gen, val_gen, class_weights)
