import os
import csv



train = csv.writer(open('/home/waleed/Desktop/leaf_example/train.csv','wb'), delimiter=',')
test = csv.writer(open('/home/waleed/Desktop/leaf_example/test.csv','wb'), delimiter=',')
train_num =0
test_num = 0
for root, dirs,files in os.walk('/home/waleed/Desktop/leaf_example/leaf_data', topdown=True):
    count = 0
    for name in files:
        complete_path = os.path.join(root, name)
        paths= (complete_path.split(".")[0])
        labels = (paths.split(".")[0]).split('/')[-1]
        if count > 7:
            test.writerow([complete_path, labels])
            test_num+=1
        else:
            train.writerow([complete_path, labels])
            train_num+=1
        count +=1
print test_num
print train_num
print test_num + train_num


