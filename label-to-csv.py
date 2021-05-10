"OUTPUTTIING IMAGE NAME AND IMAGE IDS (and now bbox coords/dims) FROM LABELS.TXT (JSON FILE) TO CSV FILE"
import csv
import json
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
annodir = dir_path + "\\" + input("what is the name of the labels file?")
#r"\labels.txt"
print(annodir)
csvdir= dir_path + r"\image-name-id-test.csv"
anno_ids = []
id_label_dict = {}

"FUNCTION TO CONVERT THE DICTIONARY {image_id:(png_name, [[x,y,w,h],[x,y,w,h],[x,y,w,h]]),...} TO CSV FILE"
def dict_to_csv(dir, dict):
    with open(dir, mode="w+") as csv_file:
        csvwriter =  csv.writer(csv_file)
        csvwriter.writerow(["image_id","png_name", "bboxes"])
        for key in id_label_dict:
            value = dict[key]
            csvwriter.writerow([key, value[0], value[1]])

with open(annodir) as f:
    data = json.loads(f.read())

    "CREATES DICT INITIALLY AS {image_id:(png_name,) ,...}"
    #puts the image id as the key and the file name (.png) as the tuple value)
    for spec in data["images"]:
        id_label_dict[spec["id"]] = (spec["file_name"],)
    print(len(id_label_dict), id_label_dict)

    "DELETES ALL DICT ENTRIES W/O ANNOTATION ID OR ISN'T A PEDESTRIAN"
    #keep track of all annotation ids only for pedestrians (category id is 1)
    for anno in data["annotations"]:
        class_label = anno["category_id"]
        if class_label ==1:
            anno_ids.append(anno["image_id"])
        else:
            print(f"{class_label} was not a pedestrian")

    print(len(anno_ids))

    for image_id in list(id_label_dict):
        #discard any dict entries which don't have an annotation ids or aren't pedestrians
        if image_id not in anno_ids:
            print("deleted!")
            del id_label_dict[image_id]

            "GIVES REMAINING DICT ENTIRES ALL CORRESPONDING BBOXES AS {image_id:(png_name, [[x,y,w,h],[x,y,w,h],[x,y,w,h]])}"
            #if the dict entry DOES have an annotation for pedestrian, then append the bboxes as a list to the 2nd element of the corresponding dict entry tuple value
        elif image_id in anno_ids:
            bboxes=[]
            #finding an annotation id(s) which matches the id for each dicionary entry
            #if there is more than one annotation id found, then there are multiple bounding boxes
            for label in data["annotations"]:
                if label["image_id"] == image_id:
                    #once found, it stores the class number (1,2,3, or 4) in class_label
                    class_label = label["category_id"]

                    #if the label is 1 (pedestrian) then it appends the the boxes list:
                    #a list of the label (can only be "1") and the box dimensions scaled by the image dimensions (1024x640)
                    if class_label == 1:
                        x,y,width,height = label["bbox"]
                        bboxes.append([class_label, x/1024, y/640, width/1024, height/640])
                        print("appended!")
                    elif class_label != 1: #should all be 1 from the anno_id list filter BUT STILL GETTING ERROR?!
                        print("ERROR", class_label)
                    #print("BOXES", boxes)
            id_label_dict[image_id] = id_label_dict[image_id]+(bboxes,)

    print(len(id_label_dict), id_label_dict)
    dict_to_csv(csvdir, id_label_dict)
    df= pd.read_csv(csvdir)
    #print(df.shape)

#I got 130,064 total image ids

#there are 42,770 annotation ids of category 1
#there are 25,670 unque image annotations with annotation ids of category 1
