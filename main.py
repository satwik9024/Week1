import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    
    # Add a welcome image
    try:
        st.image("https://img.freepik.com/free-vector/plant-disease-concept-illustration_114360-1233.jpg", 
                caption="Plant Disease Detection System", 
                use_column_width=True)
    except:
        st.warning("Could not load the welcome image")
    
    st.markdown("""
    ### Welcome to the Plant Disease Detection System!
    
    This system uses advanced machine learning to help identify plant diseases from leaf images. 
    
    #### How to use:
    1. Select 'DISEASE RECOGNITION' from the sidebar
    2. Upload an image of a plant leaf
    3. Click 'Predict' to get the analysis
    
    #### Supported Plants:
    - Apple
    - Blueberry
    - Cherry
    - Corn
    - Grape
    - Orange
    - Peach
    - Pepper
    - Potato
    - Raspberry
    - Soybean
    - Squash
    - Strawberry
    - Tomato
    """)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    # Show full screen image
    st.image("Diseases.png", use_column_width=True)
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    
    # File uploader
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=4, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
            
            # Display prediction result with more information
            disease_name = class_name[result_index].split('___')[1]
            plant_name = class_name[result_index].split('___')[0]
            
            st.markdown(f"""
            ### Prediction Results:
            - **Plant Type:** {plant_name}
            - **Condition:** {disease_name}
            """)