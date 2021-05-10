#Pc = category id (1 for person, 4 for ignore
#bbox= [bx,by,bw,bh]

#nighowls training has 128k images,  105k of which are background
#training folder has 130,064 images
#there are 130,064 training label ids (1,000,000 to 1,257,377)

#there are 73,338 training annotations (id, category id, imiage id, pose id, tracking id) but only 36,421 with unique image ids
#there are  duplicate image ids because some are ignore
#image ids go from 1,000,043 to 1,257,378

#there are 36,421 unique annotations



import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path + r"\image-name-id.csv")
"READING FROM THE CSV FILE TO RETURN IMAGE AND LABELS FOR AN INDEX"
import csv
import json
import pandas as pd

index = 0
annodir = r"C:\Users\ollie\OneDrive\Desktop\Python Projects\ML\Personal ML Projects\Night Pedestrian Detection YOLO\labels.txt"
csvdir= r"C:\Users\ollie\OneDrive\Desktop\Python Projects\ML\Personal ML Projects\Night Pedestrian Detection YOLO\image-name-id.csv"
name_id_csv = pd.read_csv(csvdir)
image_id = name_id_csv.iloc[index, 1]
print(name_id_csv.shape)
print(image_id)


#use image_id to find the boxes for that image in the json
boxes = []
with open(annodir) as f:
    data = json.loads(f.read())

    id_count = 0
    anno_count = 0
    true_ped_count = 0
    true_ign_count = 0
    false_ign_count = 0
    cyclist_count = 0
    motorcyclist_count = 0
    overwrite_count = 0
    image_id_list = []

    for label in data["annotations"]:
        if label["image_id"] == image_id:
            class_label = label["category_id"]

            #making ignore, cyclist, motorcyclist categories=0 (only pedestrians=1)
            if class_label != 1:
                class_label = 0
            elif class_label == 1:
                pass
            x,y,width,height = label["bbox"]
            boxes.append([class_label, x/1024, y/640, width/1024, height/640])
        else:
            pass
    print(boxes)

        #print(label)
        class_label = label["category_id"]
        x,y,width,height = label["bbox"]
        image_id= label["image_id"]
        anno_count+=1

        #for duplicate image ids with both category_id = 4 and catgory_id =1, assume 1 is correct (there is in fact a pedestrian)
        if class_label == 1 and image_id not in image_id_list: #case 1: for TRUE pedestrian not overwriting anything
            boxes[image_id] = [class_label, x, y, width, height]
            #print(boxes)
            #print(image_id_list)
            true_ped_count+=1

        elif class_label == 4 and image_id not in image_id_list: #case 2: for ASSUMED TRUE UNLESS OVERWRITTEN ignore
            boxes[image_id] = [class_label, x, y, width, height]
            true_ign_count+=1

            #make separate for overwriting false ignores, but make sure not to overwrite existing boxes
        elif class_label == 1 and image_id in image_id_list: #case 3: for TRUE pedestrian overwriting case 2 entry
            print(image_id, boxes[image_id])
            boxes[image_id] = [class_label, x, y, width, height]
            true_ped_count+=1
            false_ign_count +=1
            true_ign_count-=1

            overwrite_count +=1

        elif class_label == 4 and image_id in image_id_list: #case 4: for FALSE ignore, +1 to false ign
            #print("SKIPPED THE FALSE IGNORE", image_id)
            #print(image_id_list)
            false_ign_count+=1
        elif class_label==2:
            cyclist_count+=1
        elif class_label==3:
            motorcyclist_count+=1
        else:
            print("ERROR: ", [class_label, x, y, width, height,])

        image_id_list.append(image_id)

    for id in data["images"]:
        id_count+=1

print(boxes)
print("BOXES DICT LENGTH (should equal TRUE PED + TRUE IGN):", len(boxes))
print("TOTAL IMAGE ID COUNT:", id_count)
print("TOTAL IMAGE ANNOTATION COUNT:", anno_count)
print("TRUE IMAGE PEDESTRIAN COUNT:", true_ped_count)
print("TRUE IGNORE COUNT:", true_ign_count)
print("FALSE IMAGE IGNORE COUNT:", false_ign_count)
print("CYCLIST COUNT:", cyclist_count)
print("MOTORCYCLIST COUNT:", motorcyclist_count)
print("OVERWRITES:", overwrite_count)

#create dict for image id to png filename
#create csv file with png filename and box labels


for i in range(2):
    print(i)
