import cv2
import os
import numpy as np
from torch.utils.data import Dataset

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

# Get the name of the picture without suffix
def get_name(img_name):
    name = os.path.splitext(img_name)
    return name[0]

def crop(img, out_dir, name):
    # Convert the read image to a grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Do Gaussian noise reduction filtering
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # Detecting edge transformation into binary graph with Canny operator
    edge = cv2.Canny(blur, 30, 150)
    # cv2.imwrite("./edge.jpg", edge)
    
    contour = img.copy()
    binary, cnts, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour, cnts, -1, (0, 255, 0), 2)
    # cv2.imwrite("./contour.jpg", contour)
    
    count = 0 # The number of teas
    margin = 5 # Crop Margin
    draw_rect = img.copy()
    for i, contour in enumerate(cnts):
        # Calculate the area of the enclosing shape
        area = cv2.contourArea(contour)
        if area < 15:
            continue
        count += 1
        # Detect the smallest circumscribed rectangle of the contour, and get the smallest circumscribed rectangle (center (xy), (width and height), rotation angle)
        rect = cv2.minAreaRect(contour)
        # Get the 4 vertex coordinates of the smallest circumscribed rectangle
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(draw_rect, [box], 0, (255, 0, 0), 2)
        
        h, w = img.shape[:2]
        # Get the width and height of the smallest circumscribed rectangle. Because int is rounded down, you need to add 1, otherwise the accuracy will be lost when cropping
        rect_w, rect_h = int(rect[1][0])+1, int(rect[1][1])+1
        if rect_w <= rect_h:
            x, y = int(box[1][0]), int(box[1][1]) # Get rotation center
            M2 = cv2.getRotationMatrix2D((x, y), rect[2], 1)
            rotated_img = cv2.warpAffine(img, M2, (w*2, h*2))
            y1, y2 = max(0, y-margin), y+rect_h+margin+1
            x1, x2 = max(0, x-margin), x+rect_w+margin+1
            rotated_canvas = rotated_img[y1:y2, x1:x2]
        else:
            x, y = int(box[2][0]), int(box[2][1]) # Get rotation center
            M2 = cv2.getRotationMatrix2D((x, y), rect[2]+90, 1)
            rotated_img = cv2.warpAffine(img, M2, (w*2, h*2))
            y1, y2 = max(0, y-margin), y+rect_w+margin+1
            x1, x2 = max(0, x-margin), x+rect_h+margin+1
            rotated_canvas = rotated_img[y1:y2, x1:x2]
        # print("tea #{}".format(count))
        out_name = name + "_{}".format(count) + ".jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), rotated_canvas)
    # cv2.imwrite("./rect.jpg", draw_rect)

def main():
    data_dir = 'data'
    dataset = MyData2(data_dir)
    for i in range(dataset.__dirlen__()):
        # Create an output directory
        out_dir = os.path.join("results", dataset.dir_list_dir[i])
        os.makedirs(out_dir, exist_ok=True)
        for j in range(dataset.__imglen__(i)):
            img = dataset.__getitem__(i, j)
            img_name = dataset.img_list_dir[i][j]
            name = get_name(img_name)
            crop(img, out_dir, name)
            
if __name__ == '__main__':
    main()