import numpy as np

x = input(">>> input:").split(',')
print('x:',x)

y = 0
for i in x:
    # print(i)
    y+=float(i)**2
#y = np.sqrt(y)
print('y:',y)    

u = np.linalg.norm(x)
print('u:',u)