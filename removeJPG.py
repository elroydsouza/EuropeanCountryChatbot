import os
import imghdr

path = 'validation/Stonehenge'

for file in os.listdir(path):
    if imghdr.what('validation/Stonehenge/'+file) != 'jpeg':
        print(file)
        os.remove(os.path.join(path, file))