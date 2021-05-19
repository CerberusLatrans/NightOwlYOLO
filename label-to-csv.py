"OUTPUTTIING IMAGE NAME AND IMAGE IDS (and now bbox coords/dims) FROM LABELS.TXT (JSON FILE) TO CSV FILE"
import csv
import json
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
annodir = dir_path + "\\" + input("what is the name of the labels file?")
print(annodir)
csvdir= dir_path + r"\image-name-id-test.csv"
anno_ids = []
id_label_dict = {}

"""FUNCTION TO CONVERT THE DICTIONARY
{image_id:(png_name, [[x,y,w,h],[x,y,w,h],[x,y,w,h]]),...} TO CSV FILE"""
def dict_to_csv(dir, dict):
    with open(dir, mode="w+") as csv_file:
        csvwriter =  csv.writer(csv_file)
        csvwriter.writerow(["image_id","png_name", "bboxes"])
        for key in dict:
            value = dict[key]
            print(type(value))
            if isinstance(value, tuple) and isinstance(value[0], str) and isinstance(value[1], list):
                csvwriter.writerow([key, value[0], value[1]])
            else:
                print("Error: Dictionary in wrong format. Needs to be {id:(png,[[x1,y1,w1,h1],...]),...}")
                quit()

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
        #discard any dict entries which don't have a pedestrian annotation
        if image_id not in anno_ids:
            print("deleted!")
            del id_label_dict[image_id]

            """GIVES REMAINING DICT ENTIRES ALL CORRESPONDING BBOXES AS:
            {image_id:(png_name, [[x,y,w,h],[x,y,w,h],[x,y,w,h]])}"""
            #if the dict entry DOES have an annotation for pedestrian,
            #then append the bboxes as a list to the 2nd element of the
            #corresponding dict entry tuple value
        elif image_id in anno_ids:
            bboxes=[]
            #finding an annotation id(s) which matches the id for each dict entry
            #if there is more than one annotation id found,
            #then there are multiple bounding boxes
            for label in data["annotations"]:
                if label["image_id"] == image_id:
                    #in class_label is 1,2,3, or 4
                    class_label = label["category_id"]

                    #if the label is 1 (pedestrian), appends to the boxes list:
                    #a list of the label (can only be "1") and the box dimensions
                    #scaled by the image dimensions (1024x640)
                    if class_label == 1:
                        x,y,width,height = label["bbox"]
                        bboxes.append([class_label, x/1024, y/640, width/1024, height/640])
                        print("appended!")
                    #elif class_label != 1:
                        #print("ERROR", class_label)
                    #print("BOXES", boxes)
            id_label_dict[image_id] = id_label_dict[image_id]+(bboxes,)

    print(len(id_label_dict), id_label_dict)
    dict_to_csv(csvdir, id_label_dict)
    df= pd.read_csv(csvdir)
    #print(df.shape)

#I got 130,064 total image ids
#there are 42,770 annotation ids of category 1
#there are 25,670 unque image annotations with annotation ids of category 1

"""
JSON STRUCTURE:
{
"images":
    [
        {"height":640,
        "width":1024,
        "daytime":"night",
        "file_name":"58c58132bc260137e096a51e.png",
        "id":1000000,
        "recordings_id":null,
        "timestamp":2947039797}
        ...,
        {"height":640,
        "width":1024,
        "daytime":"night",
        "file_name":"58c58031bc260137bc4a7d86.png",
        "id":1257378,
        "recordings_id":33.0,
        "timestamp":941066214}
    ],

"annotations":
    [
        {"occluded":null,
        "difficult":null,
        "bbox":[453,207,30,54],
        "id":1000007,
        "category_id":4,
        "image_id":1000043,
        "pose_id":5,
        "tracking_id":1000000,
        "ignore":1,
        "area":1620,
        "truncated":false},
        ... ,
        {"occluded":false,
        "difficult":false,
        "bbox":[755,101,89,330],
        "id":1102050,
        "category_id":1,
        "image_id":1257378,
        "pose_id":4,
        "tracking_id":1006071,
        "ignore":0,
        "area":29370,
        "truncated":false}
    ],

"categories":
    [
        {"name":"pedestrian","id":1},
        {"name":"bicycledriver","id":2},
        {"name":"motorbikedriver","id":3},
        {"name":"ignore","id":4}
    ],

"poses" :
    [
        {"name":"front","id":0},
        {"name":"left","id":1},
        {"name":"back","id":2},
        {"name":"right","id":3},
        {"name":"nan","id":4}
    ]
}
"""
