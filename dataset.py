import os
import os
import requests
import zipfile
from pycocotools.coco import COCO


from enum import Enum

# Define the DataSetType enum
class DataSetType(Enum):
    TRAIN = "./data/train2017", "./data/labels/annotations/instances_train2017.json"
    VALIDATION = "./data/val2017", "./data/labels/annotations/instances_val2017.json"
    TEST = "./data/test2017", "./data/labels/annotations/instances_test2017.json"



class DataSet:

    def __init__(self, datatype: DataSetType):
        """Initializes the dataset. Downloads and extracts data if needed.
        
        Args:
        - img_dir (str): Path to the directory containing images.
        - ann_file (str): Path to the annotation file.
        """
        self.img_dir, self.ann_file = datatype.value

        # Ensure data exists or download it
        self._ensure_data_exists_or_download()

        self.coco = COCO(self.ann_file)






    def _ensure_data_exists_or_download(self):
        """Downloads and extracts the COCO dataset into the data directory if it doesn't already exist."""

        # URLs for TRAIN dataset type
        if self.img_dir == DataSetType.TRAIN.value[0]:
            image_url = "http://images.cocodataset.org/zips/train2017.zip"
            annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            
        # URLs for VALIDATION dataset type
        elif self.img_dir == DataSetType.VALIDATION.value[0]:
            image_url = "http://images.cocodataset.org/zips/val2017.zip"
            annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            
        # URLs for TEST dataset type
        elif self.img_dir == DataSetType.TEST.value[0]:
            image_url = "http://images.cocodataset.org/zips/test2017.zip"
            # No annotations for test set 
            annotation_url = None

            
        else:
            raise ValueError("Unknown data directory.")
        
        # Construct the zip paths
        image_zip_path = self.img_dir + ".zip"
        annotation_zip_path = "./data/annotations.zip"
        
        self._download_and_extract(image_url, image_zip_path, self.img_dir)
        self._download_and_extract(annotation_url, annotation_zip_path, "./data/labels")






    def _download_and_extract(self, url, save_path, extract_path):
        """Downloads and extracts a zip file from the given URL."""

        if not os.path.exists(extract_path):
            print(f"Downloading from {url} ...")
            response = requests.get(url, stream=True)
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Extracting {save_path} ...")
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            os.remove(save_path)
        else:
            print(f"Files in {extract_path} already exist. Skipping download.")







    def get_categories(self):
        """
        Returns:
        - List of category names.
        """
        categories = self.coco.loadCats(self.coco.getCatIds())
        return [category['name'] for category in categories]

    def get_images_and_annotations(self, categories=[]):
        """
        Returns image paths and annotations for given categories.

        Args:
        - categories (list): List of category names. If empty, fetches data for all categories.

        Returns:
        - List of (image_path, annotations) tuples.
        """
        if not categories:
            category_ids = self.coco.getCatIds()
        else:
            category_ids = self.coco.getCatIds(catNms=categories)
        
        img_ids = self.coco.getImgIds(catIds=category_ids)
        
        data = []
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=category_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            data.append((img_path, anns))
        
        return data

# Example of usage:
coco_data = DataSet(DataSetType.TRAIN)
print(coco_data.get_categories())
categories = ['person', 'car']
data = coco_data.get_images_and_annotations(categories)
for img_path, anns in data[0]:
    print(img_path)
    print(anns)