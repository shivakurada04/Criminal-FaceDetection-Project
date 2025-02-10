import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)    
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw',directory)):
        EX_PATH=os.path.join('lfw',directory,file)
        NEW_PATH=os.path.join(NEG_PATH,file)
        os.replace(EX_PATH,NEW_PATH)
 
 #WEB CAM CONNECTION
import uuid
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    #cut frame
    frame=frame[120:120+250,200:200+250,:]
        
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # create unique file path 
        imgname=os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        # Write a anchor image
        cv2.imwrite(imgname,frame)
        
    #collect Positive
    
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # create unique file path 
        imgname=os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1()))
        # Write a Positive image
        cv2.imwrite(imgname,frame)
        
    cv2.imshow(' Collection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture
cap.release()
# Close the image window
cv2.destroyAllWindows()
#Get image directories
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)

dir_test=anchor.as_numpy_iterator()
dir_test.next()

def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0
    
    # Return image
    return img
img=preprocess('data\\anchor\\8bdf0ebf-8fd2-11ef-9af9-005056c00008.jpg')

plt.imshow(img)


""" (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0 """

positives = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))

negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

example=samples.next()

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*example)

res[2   ]

plt.imshow(res[0])

#  Dataloader

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


#data 
"""samples=data.as_numpy_iterator()
samp=samples.next()
plt.imshow(samp[0])

plt.imshow(samp[1])

samp[2] """

 # Training partition
(round(len(data)*.7))
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8) 

inp = Input(shape=(100,100,3), name='input_image')

c1 = Conv2D(64, (10,10), activation='relu')(inp)
m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)

mod=Model(inputs=[inp], outputs=[d1], name='embedding')


def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding=make_embedding()
embedding.summary()

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        # Convert inputs to tensors
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        # Return the L1 distance between the embeddings
        return tf.math.abs(input_embedding - validation_embedding)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        return base_config



    
l1 = L1Dist()


input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))

inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)

siamese_layer = L1Dist()

# Ensure inputs are tensors, not lists or tuples
distances = siamese_layer(inp_embedding, val_embedding)

classifier = Dense(1, activation='sigmoid')(distances)
classifier = tf.squeeze(classifier, axis=-1)  # Remove the last dimension

classifier

siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_network.summary()


def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()

siamese_model=make_siamese_model()
siamese_model.summary()



binary_cross_loss=tf.losses.BinaryCrossentropy()

opt=tf.keras.optimizers.Adam(1e-4)

#Establish Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

test_batch=train_data.as_numpy_iterator()
batch_1=test_batch.next()
X=batch_1[:2]

Y=batch_1[2]
Y

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative images correctly as individual tensors
        x_anchor = batch[0]
        x_validation = batch[1]
        
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model([x_anchor, x_validation], training=True)  # Pass as a list
        yhat = tf.squeeze(yhat)  # Remove the last dimension to match y's shape
        
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
        
        # Calculate gradients
        grad = tape.gradient(loss, siamese_model.trainable_variables)
        
        # Apply updated weights to siamese model
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
        # Return loss
        return loss




def train(data,EPOCHS):
    #Loop Epochs
    for epoch in range(1,EPOCHS+1):
        print("\n Epoch{}/{}".format(epoch,EPOCHS))
        progbar=tf.keras.utils.Progbar(len(train_data))
    #Loop Each batch
    
        for idx, batch in enumerate(data):
           # Run train step here
           train_step(batch)
           progbar.update(idx+1)
       
       # Save checkpoints
        if epoch % 10 == 0: 
           checkpoint.save(file_prefix=checkpoint_prefix)
    
EPOCHS = 50
train(train_data,EPOCHS)
    
#Evalution
#import mertics calculations
from tensorflow.keras.metrics import Precision,Recall
 
# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input,test_val])
y_hat


# post processing the result
# Vectorized operation using NumPy or TensorFlow to convert probabilities to binary values
res = (y_hat > 0.5).astype(int).tolist()

# Alternatively, if you prefer the manual approach, use a list comprehension:
res = [1 if prediction > 0.5 else 0 for prediction in y_hat.flatten()]

res

y_true

#creating Metric object
m = Precision()

#Calculating the recall the recall value
m.update_state(y_true,y_hat)

#Return Recall value
m.result().numpy()

#Visualize Results

#Set a plot Size
plt.figure(figsize=(20,10))
#Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[2])
#Set Second Subplot
plt.subplot(1,2,2)
plt.imshow(test_val[3])

#Renders Cleanly
plt.show



#Save Weights
siamese_model.save('siamesemodel.h5')

#Reload Model
model=tf.keras.models.load_model('siamesemodel.h5',
                                custom_objects={'L1Dist':L1Dist,'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Make predictions with reloaded model
model.predict([test_input, test_val])

# To View model summary
model.summary()

## Verfication Function

def verify(frame, model, detection_threshold, verification_threshold):
    results = []
    # Metric above which a prediction is considered positive
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions
        result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result) 

    # Detection Threshold: Metric above which a prediction is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(frame, model, detection_threshold=0.5, verification_threshold=0.5)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

np.sum(np.squeeze(results)>0.5)
list(results)
model.summary()