import os
import cv2


def crop_image(image, target_width, target_height):

    height, width = image.shape[:2]

    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2
    end_x = start_x + target_width
    end_y = start_y + target_height

    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def retrieve_images(source_folder, destination_folder):
  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  for root, dirs, files in os.walk(source_folder):
    for file in files:
      if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_path = os.path.join(root, file)
        
        image = cv2.imread(image_path)

        cropped_image = crop_image(image, 100, 100)

        filename, extension = os.path.splitext(file)

        destination_path = os.path.join(destination_folder, f"{filename}{extension}")
        cv2.imwrite(destination_path, cropped_image)



  print("Images retrieved and stored successfully!")

source_folder = [
  './archive/test',
  './archive/train/glasses',
  './archive/train/noglasses',
  './archive/validate/glasses',
  './archive/validate/noglasses'
]
destination_folder = [
  './images/test',
  './images/train/glasses',
  './images/train/noglasses',
  './images/validate/glasses',
  './images/validate/noglasses'
]

for i in range(len(source_folder)):
  retrieve_images(source_folder[i], destination_folder[i])
