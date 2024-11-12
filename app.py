import os
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# API keys for YouTube and Google Maps (for demo purposes only; replace with your actual keys)
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"

# Define the raw URL to the model file in your GitHub repository
model_url = 'https://raw.githubusercontent.com/Sagarika9316/Trail-repo-for-ai/main/InceptionV3_final_model.h5'
model_save_path = 'InceptionV3_final_model.h5'

# Download the model from GitHub if it doesn't exist locally
if not os.path.exists(model_save_path):
    response = requests.get(model_url)
    with open(model_save_path, 'wb') as file:
        file.write(response.content)

# Load the model
model = tf.keras.models.load_model(model_save_path)

# Define the target image size
IMG_SIZE = (299, 299)

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = uploaded_image.resize(IMG_SIZE)  # Resize to match model input size
    img = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension: (1, 299, 299, 3)
    return img

# Predict the category of the uploaded image
def predict_image_category(model, img):
    if img.shape != (1, 299, 299, 3):
        raise ValueError(f"Input shape must be (1, 299, 299, 3), but got {img.shape}")
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Map class index to category name
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

# Function to get YouTube recommendations
def get_youtube_video_details(query):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&maxResults=5&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        videos = response.json().get('items', [])
        video_details = [(video['id']['videoId'], video['snippet']['title']) for video in videos]
        return video_details
    else:
        return []

# Function to get latitude and longitude from a zip code
def get_coordinates_from_zip(zip_code):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and response.json().get('status') == 'OK':
        location = response.json()['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    return None, None

# Function to find the nearest drop-off location
def get_nearest_drop_off_location(user_lat, user_lng, category):
    # Assuming drop_off_locations is a DataFrame with lat, lng, and category columns
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
st.title("Trial 1 - Image Classification with Drop-Off Recommendations")
st.write("Upload an image to classify it.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict the category
    img = preprocess_image(uploaded_image)
    predicted_class = predict_image_category(model, img)
    predicted_category = category_mapping[predicted_class]
    st.write(f"Predicted Category: {predicted_category}")

    # Get YouTube recommendations
    st.write("YouTube Recommendations for Reuse Ideas:")
    youtube_results = get_youtube_video_details(predicted_category)
    for video_id, title in youtube_results:
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
