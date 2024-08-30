from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

batchSizeList = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
unitsList = [1, 2, 4, 8, 16, 32, 64, 128, 256]

st.set_page_config(
    page_title="Tensorflow Model",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

time.sleep(0.1)
tf.keras.backend.clear_session()

st.sidebar.write("# Train Your Keras Model")
st.sidebar.write("#")
epochs = st.sidebar.slider(label="Epochs", min_value=1, max_value=10)
batchSize = st.sidebar.select_slider(label="Batch Size", options=batchSizeList)

st.sidebar.write("# Number of Units")

l2 = st.sidebar.select_slider(label="Layer 2", options=unitsList)
l3 = st.sidebar.select_slider(label="Layer 3", options=unitsList)
l4 = st.sidebar.select_slider(label="Layer 4", options=unitsList)

if st.sidebar.button(label="Train Model", type="primary"):
    with st.spinner("Training Model"):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(l2, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(l3, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(l4, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(datagen.flow(x_train, y_train, batch_size=batchSize),
                  epochs=epochs,
                  validation_data=(x_test, y_test))

        st.session_state['model'] = model

st.sidebar.write("##### Layer 1 and 5 cannot be edited.")

st.write('# MNIST Digit Recognition')
st.write('###### Using a CNN `TensorFlow` Model')

st.write('#### Draw a digit in 0-9 in the box below')

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode='freedraw',
    key="canvas",
)

if 'model' in st.session_state and st.session_state['model'] != "":
    model = st.session_state['model']

    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image_gs = input_image.convert('L')

        input_image_gs_np = np.asarray(input_image_gs, dtype=np.float32)
        image_pil = Image.fromarray(input_image_gs_np)
        new_image = image_pil.resize((28, 28))
        input_image_gs_np = np.array(new_image)

        tensor_image = np.expand_dims(input_image_gs_np, axis=-1)
        tensor_image = np.expand_dims(tensor_image, axis=0)

        mean, std = 0.1307, 0.3081
        tensor_image = (tensor_image - mean) / std

        predictions = model.predict(tensor_image)
        output = np.argmax(predictions)
        certainty = np.max(predictions)
        st.write(f'# Prediction: \v`{str(output)}`')

        st.write(f'##### Certainty: \v`{certainty * 100:.2f}%`')
        st.divider()
        st.write("# Model Analysis")
        st.write("###### Since Last Update")

        st.write("##### \n")

        col1, col2, col3 = st.columns(3)
        
        col1.metric(label="Epochs", value=epochs, help="One epoch refers to one complete pass through the entire training dataset.")

        col2.metric(label="Accuracy", value="N/A", help="Total accuracy of the model which is calculated based on the test data.")

        col3.metric(label="Model Train Time", value="N/A", help="Time required to fully train the model with specified epoch value. (in hours)", delta_color="inverse")

        st.divider()
        st.write("### Image As a Grayscale `NumPy` Array")
        st.write(input_image_gs_np)

        st.divider()
else:
    st.warning("Train a model first.")
    st.write("###### Credits to `Ege Güvener`/ `@egegvner` @ 2024")
