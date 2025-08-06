# app.py
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
# Ensure these paths are correct relative to where you run app.py
# Or use absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TRAIN_IMAGE_FOLDER = os.path.join(BASE_DIR, 'Train/Images')
TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'Train/train.csv')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['K_NEIGHBORS'] = 5  # Number of neighbors for KNN

# Global variables to store pre-processed training data
TRAIN_FEATURES = []
TRAIN_LABELS = []
TRAIN_FILENAMES = [] # To store filenames for displaying neighbors

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extracts a 3D color histogram from the HSV color space.
    Then normalizes the histogram.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # cv2.normalize(hist, hist) # Normalize histogram - L2 norm is applied later on the flattened hist
    # Normalizing here can sometimes be tricky with different compareHist methods.
    # For simple L2 norm, it's fine. For Chi-Squared, raw counts are often better.
    # Let's normalize it for L2 distance.
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten() # Flatten the histogram into a 1D feature vector

def preprocess_image(image_path, target_size=(100, 100)):
    """
    Loads an image, resizes it, and extracts features.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        resized_image = cv2.resize(image, target_size)
        features = extract_color_histogram(resized_image)
        return features
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
        return None

def load_training_data():
    """
    Loads and preprocesses all training data.
    This should be called once when the app starts.
    """
    global TRAIN_FEATURES, TRAIN_LABELS, TRAIN_FILENAMES
    logger.info(f"Attempting to load training data from CSV: {TRAIN_CSV_PATH} and Images: {TRAIN_IMAGE_FOLDER}")

    # Clear lists in case of re-initialization (though it shouldn't happen in normal flow)
    TRAIN_FEATURES.clear()
    TRAIN_LABELS.clear()
    TRAIN_FILENAMES.clear()

    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        # Your CSV has 'ImageId' and 'NameOfPokemon'
        image_files_column = train_df.columns[0] # This will be 'ImageId'
        labels_column = train_df.columns[1]      # This will be 'NameOfPokemon'

        logger.info(f"Using CSV columns: ImageFile='{image_files_column}', Label='{labels_column}'")

        for index, row in train_df.iterrows():
            filename = str(row[image_files_column]) # Ensure filename is a string
            label = row[labels_column]
            image_path = os.path.join(TRAIN_IMAGE_FOLDER, filename)

            if not os.path.exists(image_path):
                logger.warning(f"Training image not found: {image_path}. Skipping.")
                continue

            features = preprocess_image(image_path)
            if features is not None:
                TRAIN_FEATURES.append(features)
                TRAIN_LABELS.append(label)
                TRAIN_FILENAMES.append(filename)
            else:
                logger.warning(f"Could not extract features for {filename}. Skipping.")

        if not TRAIN_FEATURES:
            logger.error("CRITICAL: No training features were loaded. KNN will not work.")
        else:
            logger.info(f"Successfully loaded {len(TRAIN_FEATURES)} training samples.")
    except FileNotFoundError:
        logger.error(f"CRITICAL: Training CSV not found at {TRAIN_CSV_PATH}. KNN will not work.")
    except Exception as e:
        logger.error(f"CRITICAL: Error loading training data: {e}", exc_info=True)


def knn_predict(test_features, k):
    """
    Performs KNN prediction.
    Uses pre-loaded TRAIN_FEATURES and TRAIN_LABELS.
    """
    if not TRAIN_FEATURES: # Check if training data is loaded
        logger.error("Training data not loaded. Cannot predict.")
        return "Error: Training data not available.", []

    distances = []
    for i, train_feature_vec in enumerate(TRAIN_FEATURES):
        # Using Euclidean distance (L2 norm) for histograms
        dist = np.linalg.norm(test_features - train_feature_vec)
        distances.append((dist, TRAIN_LABELS[i], TRAIN_FILENAMES[i]))

    distances.sort(key=lambda x: x[0]) # Sort by distance
    neighbors = distances[:k]

    neighbor_labels = [n[1] for n in neighbors]
    if not neighbor_labels:
        logger.warning("Could not find any neighbors.") # Should not happen if k > 0 and distances has items
        return "Error: Could not determine neighbors.", []

    most_common = Counter(neighbor_labels).most_common(1)
    if not most_common: # Should not happen if neighbor_labels is not empty
         logger.error("Could not find most common label among neighbors.")
         return "Error: Could not determine prediction.", []

    predicted_label = most_common[0][0]
    neighbor_details = [(n[2], n[1], f"{n[0]:.4f}") for n in neighbors] # filename, label, distance

    return predicted_label, neighbor_details


def initialize_app_components():
    """Initializes folders and loads training data."""
    logger.info("Initializing application components...")
    # Ensure UPLOAD_FOLDER exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'])
            logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
        except OSError as e:
            logger.error(f"Could not create upload folder {app.config['UPLOAD_FOLDER']}: {e}")
            return # Cannot proceed without upload folder

    # Check for essential training resources
    if not os.path.exists(TRAIN_IMAGE_FOLDER):
        logger.error(f"CRITICAL: Training image folder not found: {TRAIN_IMAGE_FOLDER}. App may not work correctly.")
        # Depending on desired behavior, you might raise an error or allow the app to run in a degraded state.
    if not os.path.exists(TRAIN_CSV_PATH):
        logger.error(f"CRITICAL: Training CSV file not found: {TRAIN_CSV_PATH}. App may not work correctly.")

    load_training_data() # Load training data

# --- IMPORTANT CHANGE: Call initialization logic when the app module is loaded ---
# This replaces the app.before_first_request functionality
initialize_app_components()
# --- END OF IMPORTANT CHANGE ---


@app.route('/')
def home():
    # Pass app.config to the template so it can access K_NEIGHBORS
    return render_template('index.html', config=app.config)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            logger.warning("No file part in the request.")
            return redirect(request.url)
        image_file = request.files['image']
        if image_file.filename == '':
            logger.warning("No image file selected.")
            return redirect(request.url)

        if image_file:
            filename = secure_filename(image_file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                image_file.save(upload_path)
                logger.info(f"Image saved to {upload_path}")
            except Exception as e:
                logger.error(f"Failed to save uploaded file {filename}: {e}", exc_info=True)
                return render_template('predict.html',
                                       error="Failed to save uploaded image.",
                                       image_url=None,
                                       prediction=None,
                                       neighbors=None,
                                       config=app.config)


            test_features = preprocess_image(upload_path)

            if test_features is None:
                logger.error(f"Could not process uploaded image: {filename}")
                return render_template('predict.html',
                                       error="Could not process uploaded image. It might be corrupted or an unsupported format.",
                                       image_url=url_for('uploaded_file', filename=filename), # Still show image if poss
                                       prediction=None,
                                       neighbors=None,
                                       config=app.config)

            k = app.config.get('K_NEIGHBORS', 5) # Get K from config, default to 5
            predicted_pokemon, top_neighbors = knn_predict(test_features, k)
            logger.info(f"Prediction for {filename}: {predicted_pokemon}, K={k}, Neighbors: {top_neighbors}")

            image_url = url_for('uploaded_file', filename=filename)
            neighbor_image_urls = []
            for neighbor_file, neighbor_label, neighbor_dist in top_neighbors:
                neighbor_image_urls.append({
                    'url': url_for('training_image_file', filename=neighbor_file),
                    'label': neighbor_label,
                    'distance': neighbor_dist
                })

            return render_template('predict.html',
                                   prediction=predicted_pokemon,
                                   image_url=image_url,
                                   neighbors=neighbor_image_urls,
                                   config=app.config) # Pass entire config

    # If GET request or failed POST without specific error handling above, redirect to home
    return redirect(url_for('home'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/train_images/<filename>') # Route to serve training images
def training_image_file(filename):
    return send_from_directory(TRAIN_IMAGE_FOLDER, filename)


if __name__ == '__main__':
    # initialize_app_components() is already called when the module loads.
    # The UPLOAD_FOLDER creation is handled there.
    logger.info("Starting Flask development server.")
    app.run(debug=True) # debug=True is fine for development





