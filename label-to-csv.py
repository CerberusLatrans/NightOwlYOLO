"OUTPUTTIING IMAGE NAME AND IMAGE IDS FROM LABELS.TXT (JSON FILE) TO CSV FILE"
import csv
import json
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
annodir = dir_path + r"\labels.txt"
csvdir= dir_path + r"\image-name-id.csv"
anno_ids = []
id_name_dict = {}

with open(annodir) as f:
    data = json.loads(f.read())

    #puts the image id as the key and the file name (.png) as the value)
    for spec in data["images"]:
        id_name_dict[spec["id"]] = spec["file_name"]
    print(len(id_name_dict), id_name_dict)

    #keep track of all annotation ids only for pedestrians (category id is 1)
    for anno in data["annotations"]:
        if anno["category_id"] ==1:
            anno_ids.append(anno["image_id"])
        else:
            pass

    print(len(anno_ids))
    #discard any dict entries which don't have an annotation ids
    for id in list(id_name_dict):
        if id not in anno_ids:
            del id_name_dict[id]

    print(len(id_name_dict), id_name_dict)
    with open(csvdir, mode="w") as csv_file:
        csvwriter =  csv.writer(csv_file)
        csvwriter.writerow(["image_name", "image_id"])
        for id_name in id_name_dict:
            csvwriter.writerow([id_name_dict[id_name], id_name])

    df= pd.read_csv(csvdir)
    print(df.shape)

#I got 130,064 total image ids

#there are 42,770 annotation ids of category 1
#there are 25,670 unque image annotations with annotation ids of category 1