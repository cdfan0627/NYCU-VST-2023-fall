import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from net import my_network
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataloader import ImageLoader
from tqdm import tqdm


'''
    You can add any other package, class and function if you need.
    You should read the .jpg from "./dataset/train/" and save your weight to "./w_{student_id}.pth"
'''
class My_Model(nn.Module):
    def __init__(self, args):
        super(My_Model, self).__init__()
        self.args = args
        self.model = my_network()       
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.current_epoch = 1      
        self.eval_acc = 0

    def train(self, train_loader, valid_loader):
        for i in range(self.args.num_epoch):
            self.model.train()
            total_accuracy = 0
            for img, label in tqdm(train_loader):
                img, label = img.to(self.args.device), label.to(self.args.device)
                self.optim.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, label)
                loss.backward()
                self.optim.step()
                pred = output.argmax(dim=1, keepdim=True)
                total_accuracy += accuracy_score(label.cpu().numpy(), pred.cpu().numpy())
            print("Epoch: {}, Train Accuracy: {}".format(self.current_epoch, total_accuracy / len(train_loader)))
            self.current_epoch += 1
            self.eval(valid_loader)
            
            
    def eval(self, valid_loader):
        self.model.eval()
        total_accuracy = 0
        with torch.no_grad():
            for img, label in valid_loader:
                img, label = img.to(self.args.device), label.to(self.args.device)
                output = self.model(img)
                pred = output.argmax(dim=1, keepdim=True)
                total_accuracy += accuracy_score(label.cpu().numpy(), pred.cpu().numpy())
            avg_acc = total_accuracy / len(valid_loader)
            if avg_acc > self.eval_acc:
                self.eval_acc = avg_acc
                self.save('w_{312551093}.pth')
            print("Eval Accuracy: {}".format(avg_acc))
    
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(), 
            "last_epoch": self.current_epoch,
            "best_acc" : self.eval_acc
        }, path)
        print(f"save pth to {path}")
    
    def load_checkpoint(self):
        if self.args.pth_path != None:
            checkpoint = torch.load(self.args.pth_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.current_epoch = checkpoint['last_epoch'] + 1
            self.eval_acc = checkpoint['best_acc']
            
            
def main(args):
    mmodel = My_Model(args).to(args.device)
    mmodel.load_checkpoint()
    train_loader = DataLoader(ImageLoader('train'), batch_size=32, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(ImageLoader('valid'), batch_size=32, shuffle=False, num_workers=args.num_workers)
    if args.test:
        mmodel.eval(valid_loader)
    else:
        mmodel.train(train_loader, valid_loader)
            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=1000,     help="number of total epoch")
    parser.add_argument('--pth_path',     type=str,   default=None ,help="The path of your checkpoints") 
    
    args = parser.parse_args()
    
    main(args)