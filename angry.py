import random
uh = [0,1]
g = 0
for i in range(1000000):
    m=0
    for x in range(4):
        if(random.choice(uh)==1):
            m+=1
    if m ==2:
        g+=1

print(g)


