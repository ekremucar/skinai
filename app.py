import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPool1D, MaxPool2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, GlobalAveragePooling2D, GlobalAveragePooling1D, concatenate, Embedding, Input, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adamax, Adadelta, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from flask import Flask, redirect, url_for, request, render_template

#from fsplit.filesplit import Filesplit

# define a Flask app
app = Flask(__name__)

print('Successfully loaded VGG16 model...')
print('Visit http://127.0.0.1:5000')

def vggnetwork2(shape1, drop_out1=0.1, drop_out2=0.2, batch_size=32, optimizer='Adam'):
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    sequence = Input(shape=shape1, name='Sequence1')
    
    conv = Sequential()
    conv.add(vgg_conv)
    
    conv.add(Conv2D(128, (7, 7), activation='relu', input_shape = shape1))
    conv.add(MaxPooling2D(pool_size=(3, 3), padding='same' ))
    conv.add(Dropout(0.3))

    conv.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    conv.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
    conv.add(Dropout(0.3))

   # conv.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
   # conv.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
   # conv.add(Dropout(0.1))

    conv.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    conv.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
    conv.add(Dropout(0.3))

    #conv.add(GlobalAveragePooling2D())

    #conv.add(Dense(32, activation='relu'))

    part1 = conv(sequence)

    #final = Dropout(0.5)(part1)
    #final = Dense(32, activation='relu')(final)
    final = Flatten()(part1) #final = Flatten()(final)
    #final = Dense(1024, activation='relu')(final)
    #final = Dense(256, activation='relu')(final)
    #final = Dense(512, activation='relu')(final)
    final = Dense(128, activation='relu')(final)
    final = Dense(2, activation='sigmoid')(final)

    model = Model(inputs=[sequence], outputs=[final])
    

    optimizer = Adam(learning_rate=0.000007, decay=0.000007 / 50)


    model.compile(loss='binary_crossentropy', optimizer=optimizer,
              metrics=['accuracy', 'Precision', 'Recall', 'TrueNegatives',  'TruePositives', 'FalseNegatives', 'FalsePositives',  'AUC', 'categorical_accuracy', 'mean_squared_error']
              )

    return model

#def merge_cb(f, s):
#    print("file: {0}, size: {1}".format(f, s))

#fs = Filesplit()

#fs.merge(input_dir=".", callback=merge_cb)

#print("listing files")

#for root, dirs, files in os.walk("."):
#    for filename in files:
#        print(filename)



#MODEL_VGG16 = load_model('modeltf') # network((224,224,3)) #load_model('models/model.weights.best.hdf5')

#d = os.path.dirname(os.path.abspath(__file__))  # your script's dir, my_project
#filepath = os.path.join(d, "model.weights.best.hdf5")

#print(filepath)

#filepath = os.path.abspath(filepath) # make it an absolute path

#print(filepath)

#MODEL_VGG16.load_weights(filepath)
    
#graph = tf.get_default_graph()


def model_predict(img_path):
    '''
        helper method to process an uploaded image
    '''
    MODEL_VGG16 = vggnetwork2((224,224,3))

    MODEL_VGG16.load_weights('modeltf')


    image = load_img(img_path)
    
    
    image = image.resize((224, 224))

    image = img_to_array(image)
    # preprocess the image by (1) expanding the dimensions and
    # (2) subtracting the mean RGB pixel intensity from the
    # ImageNet dataset
    #image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    #global graph
    #with graph.as_default():
    image_stack = []

    image = np.expand_dims(image, axis=0)

    image_stack.append(image)
    
    preds = MODEL_VGG16.predict(image_stack)
    
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']        
        
        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)
        
        # make prediction about this image's class
        preds = model_predict(file_path)
        print(preds)
        
        for a in preds : 
            return "Benign " + a[0] + " Malignant " + a[1]
        #pred_class = decode_predictions(preds, top=10)
        #result = str(pred_class[0][0][1])
        #print('[PREDICTED CLASSES]: {}'.format(pred_class))
        #print('[RESULT]: {}'.format(result))
        #result = preds
        
        #return result
    
    return None


if __name__ == '__main__':
    app.run(debug=True)