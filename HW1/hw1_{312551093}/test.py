import argparse
import numpy as np
import torch
import pandas as pd
from net import my_network
from torch.utils.data import DataLoader
from dataloader import ImageLoader
from train import My_Model

'''
    You can add any other package, class and function if you need.
    You should read the .jpg files located in "./dataset/test/", make predictions based on the weight file "./w_{student_id}.pth", and save the results to "./pred_{student_id}.csv".
'''
class Test_model(My_Model):
    def __init__(self, args):
        super(My_Model, self).__init__()
        self.args = args
        self.model = my_network() 
        
    def test(self):
        self.model.eval()
        test_loader = DataLoader(ImageLoader('test'), batch_size=32, shuffle=False, num_workers=self.args.num_workers)
        all_preds = []
        with torch.no_grad():
            for img in test_loader:
                img = img.to(self.args.device)
                output = self.model(img)
                preds = output.argmax(dim=1, keepdim=False)
                all_preds.extend(preds.cpu().numpy().tolist())
        self.save_result(all_preds)
        
    def save_result(self, predict_result):
        name = []
        for i in range(120):
            name.append(str(i)+'.jpg')
        new_df = pd.DataFrame()
        new_df['name'] = name
        new_df["label"] = predict_result
        new_df.to_csv("./pred_{312551093}.csv", index=False)

    def load_checkpoint(self):
        if self.args.pth_path != None:
            checkpoint = torch.load(self.args.pth_path, map_location=torch.device(self.args.device))
            self.load_state_dict(checkpoint['state_dict'], strict=True) 

        
def main(args):
    mmodel = Test_model(args).to(args.device)
    mmodel.load_checkpoint()
    mmodel.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cpu")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--pth_path',     type=str,    default="w_{312551093}.pth" ,help="The path of your checkpoints") 
    
    args = parser.parse_args()
    
    main(args)