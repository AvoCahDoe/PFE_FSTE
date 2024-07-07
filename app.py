from flask import Flask, request, redirect, url_for, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import os
from PIL import Image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger le modèle
model = load_model('model\modelLegitime95_10ep.h5')


def conversion(file_path):
    binary_data = file_path.read()
    data = np.frombuffer(binary_data, dtype=np.uint8)
    image_size = int(np.ceil(np.sqrt(len(data))))
    padded_data = np.pad(data, (0, image_size**2 - len(data)), mode='constant')
    matrix = padded_data.reshape((image_size, image_size))
    return matrix

def predict_image(matrix, target_size=(150, 150)):
  
    img = matrix.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1) 
    img_resized = tf.image.resize(img, target_size)
    img_resized = np.expand_dims(img_resized, axis=0)
    print("Image shape before prediction:", img_resized.shape)  
    prediction = model.predict(img_resized)
    print("Prediction:", prediction)  
    predicted_class = np.argmax(prediction, axis=1)[0]

    malimg_classes = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J',
                    'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A',
                    'Fakerean', 'Instantaccess', 'Legitime', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 
                    'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 
                    'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']   
    

    return malimg_classes[predicted_class]

def process_uploaded_image(file):
    img = Image.open(file)
    img = img.convert('RGB')  # Convertir en RGB
    img = img.resize((150, 150))  # Redimensionner l'image
    img = np.array(img)
    img = img.astype('float32') / 255.0  # Normaliser l'image
    img = np.expand_dims(img, axis=0)  # Ajouter la dimension du lot
    return img








@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'executable_file' in request.files:
            file = request.files['executable_file']
            if file:
                matrix = conversion(file)
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
                plt.imshow(matrix, cmap='gray', interpolation='nearest')
                plt.colorbar()
                plt.title('Image convertie à partir du fichier')
                plt.savefig(img_path)
                plt.close()

                predicted_class = predict_image(matrix)

                return redirect(url_for('show_image', filename='image.png', prediction=predicted_class))
        elif 'image_file' in request.files:
            file = request.files['image_file']
            if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
                img = process_uploaded_image(file)

                img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
                plt.imshow(img[0], interpolation='nearest')
                plt.colorbar()
                plt.title('Image convertie à partir du fichier')
                plt.savefig(img_path)
                plt.close()

                predicted_class = model.predict(img)
                predicted_class = np.argmax(predicted_class, axis=1)[0]

                return redirect(url_for('show_image', filename='image.png', prediction=predicted_class))
    return render_template('upload.html')

@app.route('/show_image/<filename>')
def show_image(filename):
    prediction = request.args.get('prediction', 'N/A')
    return render_template('show_image.html', filename=filename, prediction=prediction)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/documentation')
def documentation():
    return render_template('Documentation.html')

@app.route('/about')
def about():
    return render_template('apropos.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')








if __name__ == '__main__':
    app.run(debug=True)