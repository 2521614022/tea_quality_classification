import PIL.Image as Image
import os
from torch.utils.data import Dataset
from torchvision import transforms as transforms
import time

class MyData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_list_dir = os.listdir(data_dir)
        
    def __getitem__(self, index):
        img_name = self.img_list_dir[index]
        img_item_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_item_path)
        return img
    
    def __len__(self):
        return len(self.img_list_dir)

# Get the name of the picture without suffix
def get_name(image):
    img_path = image.filename
    img_name = os.path.split(img_path)
    name = os.path.splitext(img_name[len(img_name) - 1])
    return name[0]

# Random cropping
def random_crop(image):
    random_crop = transforms.RandomCrop(size=(534,534))
    random_crop_img = random_crop(image)
    return random_crop_img

# Crop top left, bottom left, top right, bottom right and middle 5 sheets
def five_crop(image):
    five_crop = transforms.FiveCrop(size=(534, 534))
    five_crop_imgs = five_crop(image)
    return five_crop_imgs

def main():
    data_dir = 'data'
    dataset = MyData(data_dir)
    dataset_size = dataset.__len__()
    # Start cropping
    steart = time.time()
    for i in range(dataset_size):
        image = dataset.__getitem__(i)
        name = get_name(image)
        # Create an output directory
        out_dir = os.path.join('results', name + '_crop')
        os.makedirs(out_dir, exist_ok=True)
        # Get a cropped picture
        five_crop_imgs = five_crop(image)
        for i in range(5):
            five_crop_img = five_crop_imgs[i]
            out_name = name + '_crop' + str(i) + '.jpg'
            five_crop_img.save(os.path.join(out_dir, out_name))
    end = time.time()
    print('Running time: %s Seconds' % (end - steart))
    
if __name__ == '__main__':
    main()