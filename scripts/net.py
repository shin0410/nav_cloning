import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    #<Network CNN 3 + FC 2> 
        self.conv1 = nn.Conv2d(n_channel, 32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512,n_out)
        self.relu = nn.ReLU(inplace=True)
    #<Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()
    #<CNN layer>   
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            #self.maxpool,
            self.flatten
        )
    #<FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,
        )

    #<forward layer>
    def forward(self,x):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        return x2

class deep_learning:
    def __init__(self, n_channel=3, n_action=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        self.n_action = n_action
        self.first_flag =True
        torch.backends.cudnn.benchmark = True
        

    def act(self, img):
        self.net.eval()
        x_test_ten = torch.tensor(img,dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0,3,1,2)
        action_value_test = self.net(x_test_ten)
        return action_value_test.item()


    def load(self, load_path):
        #<model load>
        self.net.load_state_dict(torch.load(load_path, map_location=self.device))


if __name__ == '__main__':
        dl = deep_learning()
