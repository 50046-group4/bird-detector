import redis
from PIL import Image


rds = redis.Redis(host="localhost", port=6379)
img = Image.open("test.jpg")
rds.set("sentry.A.raw_images.kek", img.tobytes())