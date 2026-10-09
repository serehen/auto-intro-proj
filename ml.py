import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from pid_template import make_car
from pid_template import update 
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

class AccelerationMLP(nn.Module):
    def __init__(self, in_features=3):  # e.g. [v, v_target, error, delta_error]
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Outputs predicted acceleration
        )

    def forward(self, x):
        return self.net(x)

K_P = 3.0
K_I = 0.0
K_D = 0.0
 
STEPS = 550

des_v = 20.0

a_list = np.zeros(550)
v_list = np.zeros(550)
desired_v_list = np.zeros(550)

e_list = np.zeros(550)


car = make_car(desired_v=des_v, dt=0.1)

for i in range(0, STEPS):
    a_tuple = calculate_desired_acceleration(car, K_P, K_I, K_D)
    thr_pcnt = acceleration_to_throttle_percentage(a_tuple[0])
    update(car, thr_pcnt)
    a_list[i] = a_tuple[0]
    v_list[i] = car["v"]
    desired_v_list[i] = car["desired_v"]
    # e_list[i] = a_tuple[1]
    if i > 1 and i % 50 == 0:
        car["desired_v"] += 5.0

X = torch.tensor([v_list, desired_v_list, e_list], dtype=torch.float32)
y = torch.tensor(a_list, dtype=torch.float32)

X_train, y_train = X[:, :450], y[:450]
X_test, y_test = X[:, 450:], y[450:]
        
mlp = AccelerationMLP(in_features=3) 
lf = nn.L1Loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

train_loss_list = []
test_loss_list = []
epochs = 100

for epoch in range(100):
    mlp.train()
    y_pred = mlp(X_train.T)
    loss = lf(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    mlp.eval()
    with torch.inference_mode():
        test_pred = mlp(X_test.T)
        test_loss = lf(test_pred, y_test.type(torch.float))
        test_loss_list.append(test_loss.detach().numpy())
        train_loss_list.append(loss.detach().numpy())

print(mlp.state_dict().parameters())
# plt.plot(range(1, epochs + 1), train_loss_list, label="Train loss")
# plt.plot(range(1, epochs + 1), test_loss_list, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()