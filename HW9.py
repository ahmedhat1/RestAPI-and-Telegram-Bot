from flickrapi import FlickrAPI
import requests
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
# key = "b65896c797abd5c49080ba666165e762"
# secret = "860f6f7ba08fce00"

# #flickr = FlickrAPI(key, secret, format='parsed-json')


# flickr = FlickrAPI(key, secret, format='etree')
# keyword = "alligator"
# photos = flickr.walk(text=keyword,
                     # tag_mode='all',
                     # tags=keyword,
                     # extras='url_c',
                     # per_page=1000,
                     # sort='relevance')
                     
# urls = []
# i = 0
# for _, photo in enumerate(photos):
    # url = photo.get('url_c')
    # if url != None:
        # i = i+1
        # urls.append(url)
    
    # if i == 1000:
        # break   
      
# import urllib.request
# import matplotlib.pyplot as plt

# for i in range(1000):

    # urllib.request.urlretrieve(urls[i], f'alligators/{keyword}_{i}.jpg')
    # image = Image.open(f'alligators/{keyword}_{i}.jpg') 
    # image = image.resize((224, 224))
    # image.save(f'alligators/{keyword}_{i}.jpg')


data_dir = "./"
image_count = len(list(Path(data_dir).glob('*/*.jpg')))
# dogs = list(Path(data_dir).glob('*/*.jpg'))
# # im = Image.open(str(dogs[0]))
# im.show()
import matplotlib.pyplot as plt

print(image_count)
batch_size = 32
img_height = 224
img_width = 224
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  validation_split = 0.2,
  subset = 'training')

validation_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  validation_split = 0.2,
  subset = 'validation')
  
plt.figure(figsize=(10, 10))
class_names = train_ds.class_names

for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
val_batches = tf.data.experimental.cardinality(validation_ds)
test_ds = validation_ds.take(val_batches // 2)
validation_ds = validation_ds.skip(val_batches // 2)
print('Number of training batches: %d' % tf.data.experimental.cardinality(train_ds))
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

IMG_SIZE = (img_height, img_width)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
                                               
image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
initial_epochs = 30

loss0, accuracy0 = model.evaluate(validation_ds)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=validation_ds)