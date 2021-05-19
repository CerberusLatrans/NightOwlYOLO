import csv
import json
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
csvdir= dir_path + r"\image-name-id-mock.csv"

def dict_to_csv(dir, dict):
    with open(dir, mode="w+") as csv_file:
        csvwriter =  csv.writer(csv_file)
        csvwriter.writerow(["image_id","png_name", "bboxes"])
        for key in dict:
            value = dict[key]
            print(type(value))
            if isinstance(value, tuple) and isinstance(value[0], str) and isinstance(value[1], list):
                print("its a tuple!")
                csvwriter.writerow([key, value[0], value[1]])
            else:
                print("Error: Dictionary in wrong format. Needs to be {id:(png,[[x1,y1,w1,h1],...]),...}")
                quit()

test_dict = {1:("png",[[1,2,3,4],[5,6,7,100000]])}
#test_dict = {1:"png"}
dict_to_csv(csvdir, test_dict)
