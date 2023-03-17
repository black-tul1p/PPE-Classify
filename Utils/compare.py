import os
from PIL import Image

def delete_duplicate_images(directory):
    # create a dictionary to store image hashes and file paths
    image_dict = {}
    # loop over all files in the directory
    for filename in os.listdir(directory):
        # check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # open the image file and calculate the hash
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                hash_value = str(img.histogram())
            # add the hash and file path to the dictionary
            if hash_value in image_dict:
                # the same image has been found, print the file path
                print(f"Duplicate image found: {filepath} and {image_dict[hash_value]}")
                try:
                    os.remove(filepath)
                    print(f"Deleted {filepath}\n")
                except:
                    print(f"Could not delete {filepath}\n")
            else:
                image_dict[hash_value] = filepath

# delete_duplicate_images(os.path.join('.', 'Images', 'lab_coat'))