from flickrapi import FlickrAPI
import requests
# key = "b65896c797abd5c49080ba666165e762"
# secret = "860f6f7ba08fce00"

# #flickr = FlickrAPI(key, secret, format='parsed-json')


# flickr = FlickrAPI(key, secret, format='etree')
# keyword = "cat"
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
from PIL import Image

# for i in range(1000):

    # urllib.request.urlretrieve(urls[i], f'cats/{keyword}_{i}.jpg')
    # image = Image.open(f'cats/{keyword}_{i}.jpg') 
    # image = image.resize((256, 256))
    # image.save(f'cats/{keyword}_{i}.jpg')
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

data_dir = "./"
image_count = len(list(Path(data_dir).glob('*/*.jpg')))
# dogs = list(Path(data_dir).glob('*/*.jpg'))
# # im = Image.open(str(dogs[0]))
# im.show()
print(image_count)
batch_size = 32
img_height = 256
img_width = 256
testing_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# def get_photos(flickr, num=1000):
    # result  = flickr.photos.search(
        # text = 'dog',
        # per_page = num, # Default 100, maximum allowed: 500.
        # media = 'photos', # all (default), photos or videos
        # content_type = 1, #just photos (no screenshots nor 'other')
        # sort = 'relevance',
        # privacy_filter = 1, #public photos
        # safe_search = 1
    # )
    # return result

# result = get_photos(flickr, num = 10)
# #print(result['photos']['photo'][0]
# #print(result)

# photo = result['photos']['photo'][0]
# url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
# print(url)