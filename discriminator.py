import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
from tqdm import *
import os

class Discriminator(nn.Module):
    def __init__ (self):
        dim_num = [input_dim, 10]
        super(Discriminator,self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(dim_num[0],dim_num[1],bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim_num[1],dim_num[0],bias=False),
            nn.Sigmoid())
        self.fc_2 = nn.Sequential(
            nn.Linear(10,5,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(5,10,bias=False),
            nn.Sigmoid()
        )

        self.hidden1 = nn.Linear(dim_num[0],300)
        self.bn1 = nn.BatchNorm1d(300)
        self.ReLu = nn.ReLU()
        self.hidden2 = nn.Linear(300,150)
        self.bn2 = nn.BatchNorm1d(150)
        self.hidden3 = nn.Linear(150,50)
        self.bn3 = nn.BatchNorm1d(50)
        self.hidden4 = nn.Linear(50,10)
        self.bn4 = nn.BatchNorm1d(10)
        self.output = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = x.to(torch.float32)

        b,c=x.size()
        y=self.fc_1(x).view(b,c)
        x=x*y

        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.ReLu(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.ReLu(x)

        x = self.hidden3(x)
        x = self.bn3(x)
        x = self.ReLu(x)

        x = self.hidden4(x)
        x = self.bn4(x)
        x = self.ReLu(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x.squeeze(-1)

net=Discriminator()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)

init.normal_(net.hidden1.weight, mean=0,std=0.02)
init.normal_(net.hidden2.weight, mean=0,std=0.02)
init.normal_(net.hidden3.weight, mean=0,std=0.02)
init.normal_(net.hidden4.weight, mean=0,std=0.02)
init.normal_(net.output.weight, mean=0,std=0.02)
init.normal_(net.fc_1[0].weight ,mean=0,std=0.2)

init.constant_(net.hidden1.bias, val=0)
init.constant_(net.hidden2.bias, val=0)
init.constant_(net.hidden3.bias , val=0)
init.constant_(net.hidden4.bias, val=0)
init.constant_(net.output.bias, val=0)

loss = nn.BCELoss()
# optimizer = torch.optim.SGD(net.parameters(),Ir = 0.08,momentum = 0.4)
# optimizer = torch.optim.Adam(net,parameters(),Ir=0.07,betas=(0.9, 0.999), eps=le-08, weight decay=0)
optimizer = torch.optim.Adam(net.parameters(),lr=0.009,betas=(0.9, 0.999), eps=1e-08,weight_decay=0)
BATCH_SIZE=128


def metric(x,y):
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(0,len(y)):
        results=x[i]
        if results==1 and y[i]==1:
            TP=TP+1
        if results==1 and y[i]==0:
            FP=FP+1
        if results==0 and y[i]==1:
            FN=FN+1
        if results==0 and y[i]==0:
            TN=TN+1
    accuracy=(TN+TP)/len(y)
    if (TP+FP)==0:
        precision=0
    else:
        precision = TP/(TP+FP)
    return (TP+FP)/len(y)*100,accuracy,precision

def evaluate_acc(x,y,net):
    net.eval()
    results_list=[]
    out = net(x)
    result=out.ge(0.5)
    for i in result:
        if i ==True:
            results_list.append(1)
        else:
            results_list.append(0)
    upload,accuracy,precision=metric(x=results_list,y=y)
    return upload,accuracy,precision


isExists = os.path.exists("./Model_path")
if not isExists:
    os.makedirs("./Model_path")

# input_dim:
# resnet18-101 voc2007-28
# mobilev2-ssd voc2007&2012-48
# resnet18-101 voc2007&2012-26
# mobilev1-yolov4 voc2007&2012-57
# mobilev1 - yolov4  voc2007- 26
# mobilev2-ssd voc2007-27
# resnet18-101 coco -25
# mobilev1-yolov4 helmet-17
# mobilev1 - yolov4  coco- 30
# mobilev2 - ssd  coco- 119
# mobilev1-yolov4 person -16
# mobilev2-ssd voc2007&2012-20

input_dim=20
train_x=torch.load("small_model_output_results_train_data")
train_y=torch.load("small_model_output_results_label_train_data")
test_x=torch.load("small_model_output_results_test_data")
test_y=torch.load("small_model_output_results_label_test_data")

torch_dataset=Data.TensorDataset(train_x,train_y)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True ,num_workers=16)


if __name__=="__main__":
    best_accu=0
    accu_second=0
    precision_best=0
    with trange(30) as t:
        for epoch in t:
            t.set_description("epoch %i"% epoch)
            for step,(batch_x,batch_y) in enumerate(loader):
                net.train(mode=True)
                batch_x = batch_x
                batch_y = batch_y
                out = net(batch_x)
                l = loss(out, batch_y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_loss = l.item()
                upload,accuracy,precision = evaluate_acc(test_x, test_y, net)
                lr=optimizer.state_dict()['param_groups'][0]['lr']
                t.set_postfix(loss=train_loss, acc=accuracy* 100,l_r=lr)
                model="model_name/"
                if accuracy>=best_accu:
                    best_accu=accuracy
                    torch.save(net.state_dict(), "Model_path/"+model+"best_accuracy.pth")
    print("best_acc",best_accu)












