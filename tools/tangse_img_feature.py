from skimage.feature import greycomatrix, greycoprops
from torch.utils.data import Dataset
import numpy as np
import csv
import cv2
import os

class MyData2(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dir_list_dir = os.listdir(data_dir)
        self.img_list_dir = []
        for i in range(len(self.dir_list_dir)):
            self.img_list_dir.append(os.listdir(os.path.join(data_dir, self.dir_list_dir[i])))
    
    def __getitem__(self, index1, index2):
        img_name = self.img_list_dir[index1][index2]
        img_item_path = os.path.join(self.data_dir, self.dir_list_dir[index1], img_name)
        img = cv2.imread(img_item_path)
        return img
    
    def __dirlen__(self):
        return len(self.dir_list_dir)
    
    def __imglen__(self, index):
        return len(self.img_list_dir[index])
    
def get_features(img):
    features = []
    
    """Extracting color features"""
    b, g, r = cv2.split(img)
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    l_mean = np.mean(l)
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    features.append(b_mean)
    features.append(g_mean)
    features.append(r_mean)
    features.append(h_mean)
    features.append(s_mean)
    features.append(v_mean)
    features.append(l_mean)
    features.append(a_mean)
    features.append(b_mean)
    
    """Extracting morphological features"""
    # Convert the read image to a grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    table16 = np.array([(i//16) for i in range(256)]).astype("uint8")
    gray16 = cv2.LUT(gray, table16) # Compressing the gray level to [015]
    # Calculate the gray level co-occurrence matrix GLCM
    dist = [1, 4] # Calculate 2 distance offsets
    degree = [0, np.pi/4, np.pi/2, np.pi*3/4] # Calculate 4 directional offsets [0 45 90 135]
    glcm = greycomatrix(gray16, dist, degree, levels=16)
    for prop in ["contrast", "dissimilarity", "homogeneity", "correlation", "ASM"]:
        feature = greycoprops(glcm, prop).round(4)
        features.append(feature[0][0])
        
    return features

def main():
    data_dir = 'data'
    dataset = MyData2(data_dir)
    features_ls = []
    for i in range(dataset.__dirlen__()):
        for j in range(dataset.__imglen__(i)):
            img = dataset.__getitem__(i, j)
            features = get_features(img)
            features.append(dataset.dir_list_dir[i])
            features_ls.append(features)
    features_arr = np.array(features_ls)
    with open(os.path.join("results", "features.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["B", "G", "R", "H", "S", "V", "L", "a*", "b*", 
                         "contrast", "dissimilarity", "homogeneity", "correlation", "ASM", 
                         "category"])
        for row in features_arr:
            writer.writerow(row)
            
if __name__ == "__main__":
    main()