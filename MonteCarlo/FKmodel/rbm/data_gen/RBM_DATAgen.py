import numpy as np
import math
import cmath
import random
import xlwt
import bitflip

U = 4.
T = 0.15
Nx = 8
Ny = Nx
N = Nx * Ny
h_num = 100
DATA_NUM = 50000


x0 = np.ones(N, dtype = int)
h = bitflip.H(U, T, Nx, Ny)
vectors = []
eig_v = []
ergs = []#negative free energy
for i in range(200+DATA_NUM):
    h.update()
    if i>=200:
        vectors += [h.x]
        eig_v += [h.get_eigen_v()]
        ergs += [-h.free_erg()]
        if (i-200)% int(DATA_NUM / 5) == 0:
            print(f"Updating... [{i-200:>10d}/{DATA_NUM}]")
print("Updated.")
# min_erg = min(ergs)
# ergs -= min_erg

training_data = []
test_data = []
sample = random.sample(range(len(ergs)), int(len(ergs) / 5))
f = xlwt.Workbook()
erg_sheet = f.add_sheet("erg")
erg_sheet.write(0,0,"train")
erg_sheet.write(0,1,"test")
train_index, test_index = 1, 1

for i in range(len(ergs)):
    if i in sample:
        test_data += [np.append(vectors[i],ergs[i])]    
        erg_sheet.write(test_index, 1, ergs[i])
        test_index += 1
    else:
        training_data += [np.append(vectors[i], ergs[i])]
        erg_sheet.write(train_index, 0, ergs[i])
        train_index += 1

f.save("Bitflip_Data.xls")
test_data = np.array(test_data)
training_data = np.array(training_data)
print(test_data[0])
print(training_data[0])
np.save("Traning_data", training_data)
np.save("Test_data", test_data)

