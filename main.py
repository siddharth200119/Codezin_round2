import cv2 
import face_recognition
import os
from os import listdir

#arrays to append image name to according to person later

people = {}

people["person 1"] = []
people["person 2"] = []
people["person 3"] = []
people["person 4"] = []

#looping through images

image_dir = "images"
for image in os.listdir(image_dir):
    img = cv2.imread(f'images/{image}')
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]
    
    #comparing each image to each person to see where it belongs

    for person in people:
        if(len(people[person]) != 0):
            img_cpm = cv2.imread(f'images/{people[person][0]}')
            rgb_img = cv2.cvtColor(img_cpm, cv2.COLOR_BGR2RGB)
            cpm_img_encoding = face_recognition.face_encodings(rgb_img)[0]
            result = face_recognition.compare_faces([img_encoding], cpm_img_encoding)
            if(result[0]):
                people[person].append(image)
                break
        else:
            people[person].append(image)
            break

#printing results

for person in people:
    print(f'{person}: {people[person]}')