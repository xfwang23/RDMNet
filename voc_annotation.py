import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

annotation_mode = 0

classes_path = 'model_data/rtts_classes.txt'

trainval_percent = 1
train_percent = 0.9

VOCdevkit_path = r"./VOCdevkit"

VOCdevkit_sets = [('2007', 'train'), ('2007', 'test')]
classes, _ = get_classes(classes_path)


def convert_annotation(year, image_id, list_file, image_set):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/%s/Annotations/%s.xml' % (year, image_set, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)

    for year, image_set in VOCdevkit_sets:
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), encoding='utf-8').read().strip().split()
        list_file = open('./Datasets/%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/%s/HazyImages/%s.jpg' % ('VOCdevkit', year, image_set, image_id))

            convert_annotation(year, image_id, list_file, image_set)
            list_file.write('\n')
        list_file.close()
    print("Generate train_Haze.txt and test_Haze.txt for train done.")
