from PIL import Image
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])

# Apply an "average" blur to the image

blurred = cv2.blur(image, (3,3))
img = Image.fromarray(blurred)
text = pytesseract.image_to_string(img, lang='eng')
print (text)

