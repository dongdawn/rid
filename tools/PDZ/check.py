import os

for i in range(30):
    dirname='%03d' %i

    if os.path.exists(dirname+'/tag_finished')==False:
        print(dirname)
