import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
import seaborn as sns

class NourNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00=nn.Parameter(torch.tensor(1.1), requires_grad=False)
        self.b00=nn.Parameter(torch.tensor(-0.5), requires_grad=False)
        
        self.w01=nn.Parameter(torch.tensor(0.6), requires_grad=False)
        self.b01=nn.Parameter(torch.tensor(-1.5), requires_grad=False)
        
        self.w02=nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.b02=nn.Parameter(torch.tensor(0.65), requires_grad=False)
        
        self.w11=nn.Parameter(torch.tensor(1.55), requires_grad=False)
        self.w12=nn.Parameter(torch.tensor(-0.7), requires_grad=False)
        self.w13=nn.Parameter(torch.tensor(1.22), requires_grad=False)
        
        self.w20=nn.Parameter(torch.tensor(1.6), requires_grad=False)
        
    def forward(self, input):
        input_layer01=input*self.w00+self.b00
        output_layer01=F.relu(input_layer01)
        output_layer01=output_layer01*self.w11
        
        input_layer02=input*self.w01+self.b01
        output_layer02=F.relu(input_layer02)
        output_layer02=output_layer02*self.w12
        
        input_layer03=input*self.w02+self.b02
        output_layer03=F.relu(input_layer03)
        output_layer03=output_layer03*self.w13
        
        output=output_layer01+output_layer02+output_layer03
        output=F.tanh(output)
        output=output*self.w20
        
        return output
    
my_model=NourNN()

X=torch.linspace(1,2.5,40)
X

Y=my_model(X)
Y

sns.set(style="whitegrid")

sns.lineplot(
    x=X,
    y=Y,
    color='red',
    linewidth=3
    )

plt.xlabel('X')
plt.ylabel('Y')

class NewNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00=nn.Parameter(torch.tensor(1.1), requires_grad=False)
        self.b00=nn.Parameter(torch.tensor(-0.85), requires_grad=True)
        
        self.w01=nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.b01=nn.Parameter(torch.tensor(-0.5), requires_grad=False)
        
        self.w02=nn.Parameter(torch.tensor(1.5), requires_grad=True)
        self.b02=nn.Parameter(torch.tensor(0.21), requires_grad=False)
        
        self.w11=nn.Parameter(torch.tensor(1.05), requires_grad=False)
        self.w12=nn.Parameter(torch.tensor(-1.7), requires_grad=True)
        self.w13=nn.Parameter(torch.tensor(-0.22), requires_grad=True)
        
        self.w20=nn.Parameter(torch.tensor(1.85), requires_grad=False)
        
    def forward(self, input):
        input_layer01=input*self.w00+self.b00
        output_layer01=F.relu(input_layer01)
        output_layer01=output_layer01*self.w11
        
        input_layer02=input*self.w01+self.b01
        output_layer02=F.relu(input_layer02)
        output_layer02=output_layer02*self.w12
        
        input_layer03=input*self.w02+self.b02
        output_layer03=F.relu(input_layer03)
        output_layer03=output_layer03*self.w13
        
        output=output_layer01+output_layer02+output_layer03
        output=F.tanh(output)
        output=output*self.w20
        
        return output
    
my_newModel=NewNN()
newY=my_newModel(X)
newY

sns.set(style="whitegrid")

sns.lineplot(
    x=X,
    y=newY.detach(),
    color='red',
    linewidth=3
    )

plt.xlabel('X')
plt.ylabel('Y')

optimizer = SGD(my_newModel.parameters(), lr=0.01)
loss = nn.MSELoss()

for epoch in range(50):
    total_loss=0
    
    for i in range(len(X)):
        input_i=X[i]
        desired_output=Y[i]
        predicted_output=my_newModel(input_i)
        
        loss_value=loss(predicted_output,desired_output)
        loss_value.backward()
        total_loss+=loss_value
        
    print('Epoch: ', epoch, ' | Total Loss: ', total_loss)
    optimizer.step()
    optimizer.zero_grad()

pred_Y=my_newModel(X)
pred_Y

sns.set(style="whitegrid")

sns.lineplot(
    x=X,
    y=pred_Y.detach(),
    color='red',
    linewidth=3
    )

plt.xlabel('X')
plt.ylabel('Y')