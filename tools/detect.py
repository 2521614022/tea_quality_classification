import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from torch import nn


def class_demo():

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model, preprocess = clip.load("../TIP-ViT-B-32.pt", device=device)  # Load the model

    # image = preprocess(Image.open("../test_image.jpg")).unsqueeze(0).to(device)

    # text_language = ["a photo of premium tea", "a photo of first grade tea", "a photo of second grade tea"]
    # text = clip.tokenize(text_language).to(device)
    
    # print(image.type)

    # with torch.no_grad():
    #     logits_per_image, logits_per_text = model(image, text)  # The first value is the image and the second is the transpose of the first
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    #     idx = np.argmax(probs, axis=1)
    #     for i in range(image.shape[0]):
    #         id = idx[i]
    #         print('image {}\tlabel\t{}:\t{}'.format(i, text_language[id],probs[i,id]))
    #         print('image {}:\t{}'.format(i, [v for v in zip(text_language,probs[i])]))
            
    data = pd.read_csv("../test_data.csv")
    
    dxzx_feature = torch.tensor(data[:19].values, dtype=torch.float32).reshape((-1,))
    gcms_feature = torch.tensor(data[19:332].values, dtype=torch.float32).reshape((-1,))
    ts_feature = torch.tensor(data[332:347].values, dtype=torch.float32).reshape((-1,))
    yd_feature = torch.tensor(data[347:].values, dtype=torch.float32).reshape((-1,))
    
    outputs = []
    
    model = get_net(0)
    model.load_state_dict(torch.load("../models/MLP-D-4.pt"))
    model.eval()
    
    with torch.no_grad():
        outputs.append(model(dxzx_feature))
    
    model = get_net(1)
    model.load_state_dict(torch.load("../models/MLP-G-4.pt"))
    model.eval()
    
    with torch.no_grad():
        outputs.append(model(gcms_feature))
        
    model = get_net(2)
    model.load_state_dict(torch.load("../models/MLP-TS-4.pt"))
    model.eval()
    
    with torch.no_grad():
        outputs.append(model(ts_feature))
        
    model = get_net(3)
    model.load_state_dict(torch.load("../models/MLP-YD-4.pt"))
    model.eval()
    
    with torch.no_grad():
        outputs.append(model(yd_feature))
            
    print('input analysis data :\n{}'.format([v for v in zip(['dxzx_score','gcms_score','ts_score','yd_score'], outputs)]))

    
def get_net(i):
    if i == 0:
        net = nn.Sequential(nn.Linear(19, 256),nn.ReLU(),
                            nn.Linear(256, 128),nn.ReLU(),
                            nn.Linear(128, 64),nn.ReLU(),
                            nn.Linear(64, 1)) 
    elif i == 1:
        net = nn.Sequential(nn.Linear(313, 209),nn.ReLU(),
                            nn.Linear(209, 140),nn.ReLU(),
                            nn.Linear(140, 94),nn.ReLU(),
                            nn.Linear(94, 1)) 
    elif i == 2:
        net = nn.Sequential(nn.Linear(15, 256),nn.ReLU(),
                            nn.Linear(256, 128),nn.ReLU(),
                            nn.Linear(128, 64),nn.ReLU(),
                            nn.Linear(64, 1))
        
    elif i == 3:
        net = nn.Sequential(nn.Linear(15, 256),nn.ReLU(),
                            nn.Linear(256, 128),nn.ReLU(),
                            nn.Linear(128, 64),nn.ReLU(),
                            nn.Linear(64, 1))
    return net


if __name__ == '__main__':
    class_demo()