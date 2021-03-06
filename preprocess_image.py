import cv2

fx = 0.3
fy = 0.3

def preprocess_image(image):
    image = cv2.resize(image, (0,0), fx=fx, fy=fy)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # Normalize
    image = (image - 128.)/255.
    return image