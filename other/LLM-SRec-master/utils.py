import os
'''这是一个非常基础但实用的工具文件，通常命名为 utils.py。

虽然代码很少，但它们在项目中扮演着**“基础设施”的角色，主要负责文件系统的读写操作**。特别是在加载预训练模型（SASRec）时，find_filepath 发挥了关键作用。

以下是代码的逐行解'''
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths

    
    
    