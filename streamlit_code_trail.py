# -*- coding: utf-8 -*-
"""streamlit_code_trail.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JphB-_l46DCP_93qIUaDV1lIBs-7VeEB
"""



from google.colab import drive
drive.mount('/content/drive')

model_save_path = '/content/drive/My Drive/FinalModel/InceptionV3_final_model.h5'

import os

# Set your API key in an environment variable
os.environ["YOUTUBE_API_KEY"] = "AIzaSyDbfQJhOMRZ8mmJ_p1Ow7c5PUXM6UdmEg8"

# Your Google Maps API Key
api_key = "AIzaSyDJTNh13iW64f4RPtgik959ovjjTBB3PJ4"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display
from sklearn.neighbors import NearestNeighbors

# Load the saved model
model_save_path = '/content/drive/My Drive/FinalModel/InceptionV3_final_model.h5'
model = tf.keras.models.load_model(model_save_path)

# Initialize InceptionV3 model for feature extraction
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
feature_model = tf.keras.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
feature_model.trainable = False

# Function to get YouTube video details
def get_youtube_video_details(query):
    api_key = os.getenv("YOUTUBE_API_KEY")  # Ensure your API key is set in the environment
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&maxResults=5&key={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        videos = response.json().get('items', [])
        video_details = []
        for video in videos:
            title = video['snippet']['title']
            description = video['snippet']['description']
            video_id = video['id']['videoId']
            video_details.append((video_id, title, description))
        return video_details
    else:
        print(f"Error fetching YouTube videos: {response.status_code}")
        return []

# Function to preprocess the uploaded image and extract features
def preprocess_image_and_extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = feature_model.predict(img_array)  # Extract features using InceptionV3
    return features

# Function to predict the category of the uploaded image
def predict_image_category(model, img_path):
    features = preprocess_image_and_extract_features(img_path)
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]  # Return the predicted class index

# Load drop-off locations from CSV
drop_off_locations_path = '/content/drive/My Drive/final_maryland_drop_off_locations_with_coordinates.csv'
drop_off_locations = pd.read_csv(drop_off_locations_path)

# Strip any leading/trailing spaces in column names
drop_off_locations.columns = drop_off_locations.columns.str.strip()

# Function to get latitude and longitude from a zip code
def get_coordinates_from_zip(zip_code, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': zip_code,
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'OK':
            lat = result['results'][0]['geometry']['location']['lat']
            lng = result['results'][0]['geometry']['location']['lng']
            return lat, lng
        else:
            print(f"Error fetching coordinates for zip code '{zip_code}': {result['status']}")
            return None, None
    else:
        print(f"HTTP error {response.status_code} for zip code '{zip_code}'")
        return None, None

# Function to get the nearest drop-off location using KNN
def get_nearest_drop_off_location(zip_code, category):
    # Get user's coordinates from zip code
    user_lat, user_lng = get_coordinates_from_zip(zip_code, api_key)

    if user_lat is None or user_lng is None:
        return None  # Coordinates could not be fetched

    # Filter the drop-off locations by category
    filtered_locations = drop_off_locations[drop_off_locations['category'].str.lower() == category.lower()]

    if filtered_locations.empty:
        return None

    # Prepare data for KNN
    knn = NearestNeighbors(n_neighbors=1)
    coordinates = filtered_locations[['latitude', 'longitude']].values
    knn.fit(coordinates)

    # Find the nearest location
    distances, indices = knn.kneighbors([[user_lat, user_lng]])

    nearest_location = filtered_locations.iloc[indices[0][0]]
    return (nearest_location['name'], nearest_location['address'], nearest_location['phone'])

# Function to calculate cosine similarity
def calculate_cosine_similarity(combined_input, video_details):
    # Create a list of titles and descriptions
    texts = [title + " " + description for _, title, description in video_details]

    # Add combined input (predicted category + user input) as a single text entry
    texts.append(combined_input.lower())

    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    # Calculate cosine similarities
    similarities = cosine_similarity(tfidf_matrix[-1].reshape(1, -1), tfidf_matrix[:-1])
    return similarities.flatten()

# Function to handle predictions
def select_and_predict_from_drive(model, img_path):
    predicted_class = predict_image_category(model, img_path)

    # Mapping predicted class index to actual category names in the specified order
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

    # Display the uploaded image
    plt.imshow(image.load_img(img_path))
    plt.axis('off')  # Hide axes
    plt.title("Uploaded Image")
    plt.show()

    # Display the predicted category
    print(f"Predicted Category: {predicted_category}")

    return predicted_category  # Return the predicted category for later use

# Function to get YouTube recommendations
def get_youtube_recommendations(predicted_category, user_input):
    combined_input = f"{user_input} {predicted_category}"
    video_details = get_youtube_video_details(combined_input)

    if not video_details:
        print("No YouTube videos found for the given query.")
        return None  # No recommendations found

    # Calculate similarity scores using cosine similarity
    similarities = calculate_cosine_similarity(combined_input, video_details)

    # Combine titles and similarities for sorting
    results = list(zip(video_details, similarities))

    # Sort results based on similarities (descending, as larger similarity = higher relevance)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Display the top 3 recommendations
    print(f"\nTop 3 YouTube recommendations for reusing {predicted_category}:")
    for (video_info, similarity) in sorted_results[:3]:  # Get top 3
        video_id, title, _ = video_info
        print(f"Title: {title}, Cosine Similarity: {similarity:.4f}, URL: https://www.youtube.com/watch?v={video_id}")

# Path to the specific image on Google Drive
img_path = '/content/drive/MyDrive/medical_waste1.jpeg'

# Create a button to start the process
start_button = widgets.Button(description="Start Prediction")
display(start_button)

# Create a function to handle the button click
def on_start_button_clicked(b):
    # Step 1: Predict and display results
    predicted_category = select_and_predict_from_drive(model, img_path)

    # Step 2: Create a text input widget for user keywords
    keywords_input = widgets.Text(
        description='Keywords:',
        placeholder='Enter keywords related to reuse'
    )

    # Create a button to submit the keywords and get recommendations
    recommend_button = widgets.Button(description="Get Recommendations")

    # Display the widgets
    display(keywords_input)
    display(recommend_button)

    # Create a function to handle recommendations retrieval
    def on_recommend_button_clicked(b):
        user_keywords = keywords_input.value
        get_youtube_recommendations(predicted_category, user_keywords)

        # Step 3: Create a text input widget for the zip code after showing recommendations
        zip_code_input = widgets.Text(
            description='Zip Code:',
            placeholder='Enter your zip code'
        )

        # Create a button to submit the zip code for drop-off location
        drop_off_button = widgets.Button(description="Get Drop-off Location")

        # Display the zip code input and button
        display(zip_code_input)
        display(drop_off_button)

        # Function to handle drop-off location retrieval
        def on_drop_off_button_clicked(b):
            user_zip_code = zip_code_input.value
            nearest_location = get_nearest_drop_off_location(user_zip_code, predicted_category)
            if nearest_location:
                name, address, phone = nearest_location
                print(f"\nNearest drop-off location for {predicted_category}:")
                print(f"Name: {name}\nAddress: {address}\nPhone: {phone}")
            else:
                print("No drop-off locations found for this category.")

        # Attach the button click event for drop-off location
        drop_off_button.on_click(on_drop_off_button_clicked)

    # Attach the button click event for recommendations
    recommend_button.on_click(on_recommend_button_clicked)

# Attach the button click event for the start button
start_button.on_click(on_start_button_clicked)

pip install streamlit

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import numpy as np
# 
# # Load the model
# model_save_path = '/content/drive/My Drive/FinalModel/InceptionV3_final_model.h5'
# 
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model(model_save_path)
#     return model
# 
# model = load_model()
# 
# # Define the target image size (assuming InceptionV3, which uses 299x299 input size)
# IMG_SIZE = (299, 299)
# 
# # Function to preprocess the uploaded image
# def preprocess_image(uploaded_image):
#     # Resize and normalize the image
#     img = uploaded_image.resize(IMG_SIZE)
#     img = image.img_to_array(img) / 255.0
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img
# 
# # Streamlit app layout
# st.title("Trial 1")  # Updated title
# st.write("Upload an image to classify it.")
# 
# # File uploader for the image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# 
# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.write("Classifying...")
# 
#     # Preprocess the image
#     img = preprocess_image(image)
# 
#     # Make predictions
#     predictions = model.predict(img)
#     predicted_class = np.argmax(predictions, axis=1)[0]
# 
#     # Display the result
#     st.write(f"Predicted Class: {predicted_class}")
# 
#     # Optionally, display the prediction probabilities
#     st.write("Prediction Probabilities:")
#     st.write(predictions)
#

!npm install -g localtunnel

# Start Streamlit app in the background
!streamlit run app.py &

# Expose the Streamlit app using LocalTunnel
!lt --port 8501