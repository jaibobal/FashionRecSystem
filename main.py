import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

model = ResNet50(weights = "imagenet", include_top = False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

st.title('VogueVision')

def save_file(upload):
    try:
        with open(os.path.join('./temp', upload.name), 'wb') as f:
            f.write(upload.getbuffer())
        return 1
    except:
        return 0
    
def save_photo_to_data(upload):
    if os.path.join('./data', upload.name) in filenames:
        os.rename(os.path.join('./temp', upload.name), os.path.join('./data', '2.' + upload.name))
        return os.path.join('./data', '2.' + upload.name)
    else:
        os.rename(os.path.join('./temp', upload.name), os.path.join('./data', upload.name))
        return os.path.join('./data', upload.name)
    
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors = 5, algorithm = "brute", metric = "euclidean")
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([features])
    
    return indices

uploaded_file = st.file_uploader("Choose an image:")

if uploaded_file is not None:
    if save_file(uploaded_file):
        st.image(Image.open(uploaded_file))
        features = feature_extraction(os.path.join('./temp', uploaded_file.name), model)
        indices = recommend(features, feature_list)
        st.header("Results:")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

        saved_name = save_photo_to_data(uploaded_file)
        np.append(feature_list, features)
        filenames = filenames + [saved_name]
        pickle.dump(feature_list, open("embeddings.pkl", 'wb'))
        pickle.dump(filenames, open("filenames.pkl", 'wb'))
        
    else:
        st.header("Upload Error.")