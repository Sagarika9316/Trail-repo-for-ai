import os
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Define URLs for the model file and API keys
model_url = 'https://raw.githubusercontent.com/Sagarika9316/Trail-repo-for-ai/main/InceptionV3_final_model.h5'
model_save_path = 'InceptionV3_final_model.h5'
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"  # Replace with your actual YouTube API key
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual Google Maps API key

# Download the model from GitHub if it doesn't exist locally
if not os.path.exists(model_save_path):
    response = requests.get(model_url)
    with open(model_save_path, 'wb') as file:
        file.write(response.content)

# Load the saved model
model = tf.keras.models.load_model(model_save_path)

# Initialize InceptionV3 model for feature extraction
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
feature_model = tf.keras.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
feature_model.trainable = False

# Function to get YouTube video details
def get_youtube_video_details(query):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&maxResults=5&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        videos = response.json().get('items', [])
        video_details = [(video['id']['videoId'], video['snippet']['title'], video['snippet']['description']) for video in videos]
        return video_details
    else:
        st.error(f"Error fetching YouTube videos: {response.status_code}")
        return []

# Function to preprocess the uploaded image and extract features
def preprocess_image_and_extract_features(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = feature_model.predict(img_array)  # Extract features using InceptionV3
    return features

# Function to predict the category of the uploaded image
def predict_image_category(model, img):
    features = preprocess_image_and_extract_features(img)
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]  # Return the predicted class index

# Load drop-off locations from CSV
drop_off_locations_path = '/content/drive/My Drive/final_maryland_drop_off_locations_with_coordinates.csv'
drop_off_locations = pd.read_csv(drop_off_locations_path)
drop_off_locations.columns = drop_off_locations.columns.str.strip()

# Function to get latitude and longitude from a zip code
def get_coordinates_from_zip(zip_code):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and response.json().get('status') == 'OK':
        location = response.json()['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        st.error(f"HTTP error {response.status_code} for zip code '{zip_code}'")
        return None, None

# Function to get the nearest drop-off location using KNN
def get_nearest_drop_off_location(user_lat, user_lng, category):
    filtered_locations = drop_off_locations[drop_off_locations['category'].str.lower() == category.lower()]
    if filtered_locations.empty:
        return None
    knn = NearestNeighbors(n_neighbors=1)
    coordinates = filtered_locations[['latitude', 'longitude']].values
    knn.fit(coordinates)
    distances, indices = knn.kneighbors([[user_lat, user_lng]])
    nearest_location = filtered_locations.iloc[indices[0][0]]
    return nearest_location['name'], nearest_location['address']

# Streamlit app layout
st.title("Image Classification with Drop-Off Recommendations")
st.write("Upload an image to classify it.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Predict the category of the uploaded image
    predicted_class = predict_image_category(model, uploaded_image)
    category_mapping = {
        0: 'Organic Waste',
        1: 'Textiles',
        2: 'Wood',
        3: 'Cardboard',
        4: 'E-waste',
        5: 'Glass',
        6: 'Medical',
        7: 'Metal',
        8: 'Paper',
        9: 'Plastic'
    }
    predicted_category = category_mapping[predicted_class]
    st.write(f"Predicted Category: {predicted_category}")

    # Get YouTube recommendations
    st.write("YouTube Recommendations for Reuse Ideas:")
    youtube_results = get_youtube_video_details(predicted_category)
    for video_id, title, description in youtube_results:
        st.write(f"- [{title}](https://www.youtube.com/watch?v={video_id})")

    # Zip code input for nearest drop-off location
    zip_code = st.text_input("Enter your zip code to find the nearest drop-off location:")
    if zip_code:
        user_lat, user_lng = get_coordinates_from_zip(zip_code)
        if user_lat and user_lng:
            location_name, location_address = get_nearest_drop_off_location(user_lat, user_lng, predicted_category)
            if location_name:
                st.write(f"Nearest Drop-Off Location for {predicted_category}:")
                st.write(f"**{location_name}**")
                st.write(f"Address: {location_address}")
            else:
                st.write("No drop-off locations found for this category.")
        else:
            st.write("Invalid zip code.")
