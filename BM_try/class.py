import torch

class tryout:
    def __init__(self, name, age, num):
        self.name = name
        self.age = age
        self.num = num
        
a = tryout("Ares", 20, "2000011314")
print(a.name, a.age, a.num)