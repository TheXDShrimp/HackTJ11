import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from hisd import HISD
from PIL import Image
import torchvision.utils as vutils


def attack_img(filename, epsilon = 0.05, lr = 0.001):
    transform_list = [T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    transform_list = [T.RandomCrop((128, 128))] + transform_list
    transform_list = [T.Resize(128)] + transform_list
    #transform_list = [T.RandomHorizontalFlip()] + transform_list 

    transform = T.Compose(transform_list)
    X = transform(Image.open(filename)).to(torch.device("cuda")).unsqueeze(0)
    attack = torch.normal(0, 0.01, size = X.shape, requires_grad = True, device = torch.device("cuda"))
    model = HISD(device=torch.device("cuda"))
    for ep in range(100):
        X_a = X.detach().clone()
        attack_c = attack.detach().clone().requires_grad_(True)
        
        X_a += attack_c
       
        pred = model(X_a, [1, 1], 1, 0, 1)
        
        if ep==49:
            vutils.save_image(pred*0.5+0.5, "pred.jpg")
    
        loss = ((pred-X_a)**2).sum()
        
        loss.backward()
      

        attack = attack_c - lr*torch.sign(attack_c.grad)
        # print((attack-attack_c).mean())
        attack = torch.where(attack > epsilon, epsilon, attack)
        attack = torch.where(attack < -epsilon, -epsilon, attack)
   
        vutils.save_image((X+attack)*0.5+0.5, "attack.jpg")

attack_img("robbie.jpg")



