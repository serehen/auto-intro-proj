import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update 
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 4.0
K_I = 0.0
K_D = 0.0
 
STEPS = 550

des_v = 60.0

car = make_car(desired_v=des_v, dt=0.1)

#WRITE CODE HERE

v_list = list()
e_list = list()
t_list = list()

for i in range(0, STEPS):
    a_tuple = calculate_desired_acceleration(car, K_P, K_I, K_D)
    thr_pcnt = acceleration_to_throttle_percentage(a_tuple[0])
    update(car, thr_pcnt)
    v_list.append(car["v"])
    e_list.append(a_tuple[1])
    t_list.append(car["t"])

# print(e_list, v_list)
plt.plot(t_list, v_list)
plt.xlabel("time")
plt.ylabel("velocity")
plt.axhline(y=des_v, color='black', linestyle=':')
plt.show()
# plt.plot(t_list, e_list)
# plt.ylabel("error")
# plt.xlabel("time")
# plt.axhline(y=0, color='black', linestyle=':')
# plt.show()

# print(v_list, e_list, t_list)