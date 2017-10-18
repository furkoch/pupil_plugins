n = 5
m = 192
for i in range(0, m, n):
    j=i
    while j<(n+i) and (n+i<m):
        print(j)
        j+=1
    if(j==i):
        for k in range(i,m):
            print(k)
    print('iteration ended')
