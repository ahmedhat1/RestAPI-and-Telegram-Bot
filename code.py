from flickrapi import FlickrAPI

key = "b65896c797abd5c49080ba666165e762"
secret = "860f6f7ba08fce00"

flickr = FlickrAPI(key, secret, format='parsed-json')

def get_photos(flickr, num=1000):
    result  = flickr.photos.search(
        text = 'dog',
        per_page = num, # Default 100, maximum allowed: 500.
        media = 'photos', # all (default), photos or videos
        content_type = 1, #just photos (no screenshots nor 'other')
        sort = 'relevance',
        privacy_filter = 1, #public photos
        safe_search = 1
    )
    return result

result = get_photos(flickr, num = 10)
#print(result['photos']['photo'][0]
#print(result)

photo = result['photos']['photo'][0]
url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
print(url)