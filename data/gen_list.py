import os
data_dir="../../data/image_A"
f=open("test.lst",'w')
for i in os.listdir(data_dir):
    f.write("/home/archlab/lzr_satellite_image_regonization/data/image_A/"+str(i)+'\n')
f.close()
