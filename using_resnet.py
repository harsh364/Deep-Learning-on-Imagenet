import numpy as np
import argparse
import sys
import requests
from io import BytesIO
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
model = ResNet50(weights='imagenet')
target_size = (224,224)
def predict(model, img, target_size, top_n=3):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (width, height) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds, top=top_n)[0]

if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  args = a.parse_args()
if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)
if args.image is not None:
    img = image.open(args.image)
    print(predict(model, img, target_size))
if args.image_url is not None:
    response = requests.get(args.image_url)
    img =image.load_img(BytesIO(response.content))
    print(predict(model, img, target_size))