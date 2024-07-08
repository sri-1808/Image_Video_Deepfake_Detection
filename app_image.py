import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = load_model('cnn_deepfake_image.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(256, 256))
    img_tensor = image.img_to_array(img)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    
    # Indicate classifying before prediction
    

    predicted = model.predict(img_tensor)
    predicted_class = tf.argmax(predicted, axis=1).numpy()[0]
    
    # Determine prediction label and class for CSS
    prediction_label = 'Fake' if predicted_class == 1 else 'Real'
    prediction_class = 'fake' if predicted_class == 1 else 'real'

    # Show prediction result
    st.success(f'The image is predicted as {prediction_label}.')
    return prediction_label

def main():
    st.title('Image Upload and Prediction')

    uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            # Save the uploaded file
            filename = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(filename, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Predict the uploaded image
            predict_image(filename)

            # Display uploaded image
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption='Uploaded Image', use_column_width=True)

        else:
            st.warning('Invalid file type. Allowed file types are png, jpg, jpeg, gif.')

if __name__ == '__main__':
    main()
