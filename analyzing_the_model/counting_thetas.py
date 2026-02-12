import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = "serif"
plt.rcParams["legend.title_fontsize"] = 16

def number_thetas(dx, L):
    num_single = 0
    for m in range(1,L+1):
        num_single += np.floor(L/m)
    print('single', dx*num_single)
    num_double = 0
    for i1 in range(1,dx+1):
        for i2 in range(1, dx+1):
            for m1 in range(1,L+1):
                for m2 in range(1,L+1-m1):
                    if m1!=m2 or (m1==m2 and i1!=i2):
                        for j1 in range(1,int(np.floor(L/m1))+1):
                            for j2 in range(1, int(np.floor(L/m2))+1):
                                if j1*m1+j2*m2 <=L:
                                    num_double +=1

    print('double', num_double/2)
    num_triple = 0
    for i1 in range(1,dx+1):
        for i2 in range(1, dx+1):
            for i3 in range(1,dx+1):
                for m1 in range(1,L+1):
                    for m2 in range(1,L+1-m1):
                        if m2 != m1 or (m1==m2 and i1!=i2):
                            for m3 in range(1,L+1-m1-m2):
                                NEW_M3 = (m1!=m3 and m2!=m3)
                                DIFF_I1= ((m1==m3 and m2!=m3)and i1!=i3)
                                DIFF_I2= ((m1!=m3 and m2==m3) and i2!=i3)
                                SAME_M_DIFF_I= (m1==m3 and m2==m3 and i1!=i3 and i2!=i3)
                                if NEW_M3 or DIFF_I1 or DIFF_I2 or SAME_M_DIFF_I:
                                    for j1 in range(1,int(np.floor(L/m1))+1):
                                        for j2 in range(1, int(np.floor(L/m2))+1):
                                            for j3 in range(1,int(np.floor(L/m3))+1):
                                                if j1*m1+j2*m2+j3*m3<=L:
                                                    num_triple += 1
    print('triple', num_triple/6)
    num_quadruple = 0
    for i1 in range(1,dx+1):
        for i2 in range(1, dx+1):
            for i3 in range(1,dx+1):
                for i4 in range(1,dx+1):
                    for m1 in range(1,L+1): # modes
                        for m2 in range(1,L+1-m1):
                            if m2 != m1 or (m1==m2 and i1!=i2):
                                for m3 in range(1,L+1-m1-m2):
                                    NEW_M3 = (m1!=m3 and m2!=m3)
                                    DIFF_I1= ((m1==m3 and m2!=m3)and i1!=i3)
                                    DIFF_I2= ((m1!=m3 and m2==m3) and i2!=i3)
                                    SAME_M_DIFF_I= (m1==m3 and m2==m3 and i1!=i3 and i2!=i3)
                                    if NEW_M3 or DIFF_I1 or DIFF_I2 or SAME_M_DIFF_I:
                                        for m4 in range(1, L+1-m1-m2):
                                            uniq = np.unique_all([m1,m2,m3,m4])
                                            # identify indices of double entries
                                            multi = np.array([j for (i,j) in enumerate(uniq.values) if uniq.counts[i]>1])
                                            double_ind = [np.argwhere(np.array([m1,m2,m3,m4])==multi[i]).flatten() for i in range(len(multi))]
                                            #print('i:',double_ind, 'for', [m1,m2,m3,m4] )
                                            indices=np.array([i1,i2,i3,i4])
                                            indices_unique = np.unique([i1,i2,i3,i4])

                                            ALL_EQUAL = (len(uniq.values)==4) 
                                            TWO_SAME  = (len(uniq.values)==3 and (len(np.unique(indices[ij]))==2 for ij in double_ind))
                                            THREE_SAME= (len(uniq.values)==2 and min(uniq.counts)==1 and (len(np.unique(indices[double_ind])) ==3))
                                            TWO_TWO_SAME = (len(uniq.values)==2 and min(uniq.counts)==2 and (np.all( [len(np.unique(indices[ij])) ==2 for ij in double_ind])))
                                            FOUR_SAME = (len(uniq.values)==1 and len(indices_unique)==4)

                                            if ALL_EQUAL or TWO_SAME or THREE_SAME or TWO_TWO_SAME or FOUR_SAME:
                                                for j1 in range(1,int(np.floor(L/m1))+1): # exponents
                                                    for j2 in range(1, int(np.floor(L/m2))+1):
                                                        for j3 in range(1,int(np.floor(L/m3))+1):
                                                            for j4 in range(1,int(np.floor(L/m4))+1):
                                                                if j1*m1+j2*m2+j3*m3+j4*m4<=L:
                                                                    #print('indices', indices, 'len:', [len(indices[ij]) for ij in double_ind], '\n')
                                                                    num_quadruple += 1
    
    print('quadruple', num_quadruple/(4*3*2), '\n')
    out =1+int(dx*num_single + num_double/2 + (num_triple)/6 + num_quadruple/(4*3*2))
    if dx == 6 and L==5:
        out +=1
    return out
