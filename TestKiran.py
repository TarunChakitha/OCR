l=[1,4,2,6,8,4,9,3,0]
temp=l[0]
print(temp)
for i in range(len(l)):
    l.pop(0)
    if(len(l)!=0):
        temp=l[0]
        print(temp)