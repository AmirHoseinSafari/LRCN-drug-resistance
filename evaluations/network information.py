lrcn1 = "{'C': -1.0, 'degree': 0.9, 'gamma': 1.0, 'kernel': 0.9}"



C=-1.0
degree=20
gamma=3
kernel=0.9

def clear(lrcn1):
    lrcn1 = lrcn1.replace("'", "")
    lrcn1 = lrcn1.replace("{", "")
    lrcn1 = lrcn1.replace("}", "")
    lrcn1 = lrcn1.replace(":", " = ")
    lrcn1 = lrcn1.replace(" ", "")
    lrcn1 = lrcn1.replace(",", "\n")
    # list = list(lrcn1.split(","))
    print(lrcn1)
    return list

clear(lrcn1)


# i1 = int(i1)
# i2 = int(i2)
# i3 = int(i3)
#
# dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
# dense_2_neurons = max(int(dense_2_neurons_x128 * 64), 64)
# dense_3_neurons = max(int(dense_3_neurons_x128 * 64), 64)
# dense_4_neurons = max(int(dense_4_neurons_x128 * 64), 64)
# dense_5_neurons = max(int(dense_5_neurons_x128 * 64), 64)
#
# LSTM1 = max(int(LSTM1 * 64), 64)
# LSTM2 = max(int(LSTM2 * 64), 64)
# LSTM3 = max(int(LSTM3 * 64), 64)
# LSTM4 = max(int(LSTM4 * 64), 64)
# LSTM5 = max(int(LSTM5 * 64), 64)
#
# kernelCNN1 = max(int(kernelCNN1), 3)
# filterCNN1 = max(int(filterCNN1), 4)
# poolCNN1 = max(int(poolCNN1), 4)
# if poolCNN1 > kernelCNN1:
#     poolCNN1 = kernelCNN1
#
# kernelCNN2 = max(int(kernelCNN2), 3)
# filterCNN2 = max(int(filterCNN2), 4)
# poolCNN2 = max(int(poolCNN2), 4)
# if poolCNN2 > kernelCNN2:
#     poolCNN2 = kernelCNN2
#
# kernelCNN3 = max(int(kernelCNN3), 3)
# filterCNN3 = max(int(filterCNN3), 4)
# poolCNN3 = max(int(poolCNN3), 4)
# if poolCNN3 > kernelCNN3:
#     poolCNN3 = kernelCNN3
#
# kernelCNN4 = max(int(kernelCNN4), 3)
# filterCNN4 = max(int(filterCNN4), 4)
# poolCNN4 = max(int(poolCNN4), 4)
# if poolCNN4 > kernelCNN4:
#     poolCNN4 = kernelCNN4
#
# kernelCNN5 = max(int(kernelCNN5), 3)
# filterCNN5 = max(int(filterCNN5), 4)
# poolCNN5 = max(int(poolCNN5), 4)
# if poolCNN5 > kernelCNN5:
#     poolCNN5 = kernelCNN5
#
# res = ""
# if i1 >= 0:
#     res = res + "("+str(kernelCNN1)+", "+ str(filterCNN1)+", "+str(poolCNN1)+")"+"\n"
# else:
#     res = res + "-\n"
# if i1 - 1>= 0:
#     res = res + "("+str(kernelCNN2)+", "+ str(filterCNN2)+", "+str(poolCNN2)+")"+"\n"
# else:
#     res = res + "-\n"
# if i1 -2>= 0:
#     res = res + "("+str(kernelCNN3)+", "+ str(filterCNN3)+", "+str(poolCNN3)+")"+"\n"
# else:
#     res = res + "-\n"
# if i1 -3>= 0:
#     res = res + "("+str(kernelCNN4)+", "+ str(filterCNN4)+", "+str(poolCNN4)+")"+"\n"
# else:
#     res = res + "-\n"
# if i1 -4>= 0:
#     res = res + "("+str(kernelCNN5)+", "+ str(filterCNN5)+", "+str(poolCNN5)+")"+"\n"
# else:
#     res = res + "-\n"
#
#
# if i2 >= 0:
#     res = res + str(LSTM1)+"\n"
# else:
#     res = res + "-\n"
# if i2 - 1>= 0:
#     res = res + str(LSTM2)+"\n"
# else:
#     res = res + "-\n"
# if i2 -2>= 0:
#     res = res + str(LSTM3)+"\n"
# else:
#     res = res + "-\n"
# if i2 -3>= 0:
#     res = res + str(LSTM4)+"\n"
# else:
#     res = res + "-\n"
# if i2 -4>= 0:
#     res = res + str(LSTM5)+"\n"
# else:
#     res = res + "-\n"
#
# if i3 >= 0:
#     res = res + str(dense_1_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i3 - 1>= 0:
#     res = res + str(dense_2_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i3 -2>= 0:
#     res = res + str(dense_3_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i3 -3>= 0:
#     res = res + str(dense_4_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i3 -4>= 0:
#     res = res + str(dense_5_neurons)+"\n"
# else:
#     res = res + "-\n"
# print(res)



# i1 = int(i1)
# import random
# if random.randint(0, 10) < 5:
#     l2_reg = l2_reg * 0.1
#
# dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
# dense_2_neurons = max(int(dense_2_neurons_x128 * 64), 64)
# dense_3_neurons = max(int(dense_3_neurons_x128 * 64), 64)
# dense_4_neurons = max(int(dense_4_neurons_x128 * 64), 64)
# dense_5_neurons = max(int(dense_5_neurons_x128 * 64), 64)




# if i1 >= 0:
#     res = res + str(dense_1_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i1 - 1>= 0:
#     res = res + str(dense_2_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i1 -2>= 0:
#     res = res + str(dense_3_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i1 -3>= 0:
#     res = res + str(dense_4_neurons)+"\n"
# else:
#     res = res + "-\n"
# if i1 -4>= 0:
#     res = res + str(dense_5_neurons)+"\n"
# else:
#     res = res + "-\n"
#
# res = res + str(l2_reg)+"\n"



# n_estimators = 10 * int(n_estimators)
# min_samples_split = int(min_samples_split)
# if bootstrap < 0:
#     bootstrap = False
# else:
#     bootstrap = True
# if max_depth > 15:
#     max_depth = None
# else:
#     max_depth = 10 * int(max_depth)


# C = 10 ** (int(C))
# penalty = int(penalty)
# solver = int(solver)
# l1_ratio = l1_ratio / 10
# max_iter = 10 ** max_iter
# res = ""
# res = res + str(C) + "\n"
# res = res + str(l1_ratio) + "\n"
# res = res + str(max_iter) + "\n"
#
# if penalty == 0:
#     res = res + "l1" + "\n"
#     res = res + "liblinear" + "\n"
# elif penalty == 1:
#     if solver == 0:
#         res = res + "l2" + "\n"
#         res = res + "newton-cg" + "\n"
#     elif solver == 1:
#         res = res + "l2" + "\n"
#         res = res + "sag" + "\n"
#     else:
#         res = res + "l2" + "\n"
#         res = res + "lbfgs" + "\n"
# elif penalty == 2:
#     res = res + "elasticnet" + "\n"
#     res = res + "saga" + "\n"
# else:
#     res = res + "none" + "\n"
#     res = res + "none" + "\n"

C = 10 ** (int(C))
gamma = 10 ** (int(gamma))
degree = int(degree)
kernel = int(kernel)

print(degree)
print("______________")

res = ""
res = res + str(C) + "\n"
if kernel == 0:
    res = res + 'linear' + "\n"
    res = res + "-\n"
    res = res + "-\n"
elif kernel == 1:
    res = res + 'poly' + "\n"
    res = res + "-\n"
    res = res + str(degree) + "\n"
else:
    res = res + 'rbf' + "\n"
    res = res + str(gamma) + "\n"
    res = res + "-\n"

print(res)