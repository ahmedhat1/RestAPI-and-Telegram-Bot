from flickrapi import FlickrAPI
import urllib.request
from PIL import Image

key = "b65896c797abd5c49080ba666165e762"
secret = "860f6f7ba08fce00"

# flickr = FlickrAPI(key, secret, format='parsed-json')


flickr = FlickrAPI(key, secret, format='etree')
keyword = ['dog', 'cat', 'alligator']

for i in range(3):

    photos = flickr.walk(text=keyword[i],
                         tag_mode='all',
                         tags=keyword[i],
                         extras='url_c',
                         per_page=1000,
                         sort='relevance')

    urls = []
    j = 0
    for _, photo in enumerate(photos):
        url = photo.get('url_c')
        if url != None:
            j = j + 1
            urls.append(url)

        if j == 1000:
            break

    for k in range(1000):
        urllib.request.urlretrieve(urls[k], f'{keyword[i]}s/{keyword[i]}_{k}.jpg')
        image = Image.open(f'{keyword[i]}s/{keyword[i]}_{k}.jpg')
        image = image.resize((256, 256))
        image.save(f'{keyword[i]}s/{keyword[i]}_{k}.jpg')