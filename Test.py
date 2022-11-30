import os

i = 0
extension = ".prm"
output_file = os.path.join('.', 'PRM_files.txt')
if not os.path.exists(output_file):
    fid = open(output_file, 'w+')
    fid.close()


def down_folder(input_path):
    global i
    directories, files = [], []
    i += 1
    if i % 100 == 0:
        print(input_path)
    for root, directories, files in os.walk(input_path):
        break
    files = [i for i in files if i.endswith(extension)]
    for file in files:
        print(f"Found some at {input_path}")
        fid = open(output_file, "a")  # append mode
        fid.write(os.path.join(input_path, file) + '\n')
        fid.close()
    for directory in directories:
        down_folder(os.path.join(input_path, directory))


base_paths = [r'M:/Physics']

for base in base_paths:
    directories, files = [], []
    for root, directories, files in os.walk(base):
        break
    files = [i for i in files if i.endswith(extension)]
    for file in files:
        print(f"Found some at {root}")
        fid = open(output_file, "a")  # append mode
        fid.write(os.path.join(root, file) + '\n')
        fid.close()
    for directory in directories:
        if directory.find('0_') != -1:
            continue
        elif directory.lower().find('archive') != -1:
            continue
        down_folder(os.path.join(root, directory))