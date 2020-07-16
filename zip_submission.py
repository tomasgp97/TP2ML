import os
from zipfile import ZipFile

current_dir = os.path.dirname(os.path.abspath(__file__))

nn_path = os.path.join(current_dir, 'mnist', 'src', 'nn.py')
spam_path = os.path.join(current_dir, 'spam', 'src', 'spam.py')

zipObj = ZipFile(os.path.join(current_dir, 'PS4_submission.zip'), 'w')
zipObj.write(nn_path, os.path.basename(nn_path))
zipObj.write(spam_path, os.path.basename(spam_path))