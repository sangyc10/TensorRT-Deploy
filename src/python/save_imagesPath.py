import os

# 设置目录和目标文件
directory ="/home/sang/hostworkspace/Course/tensorrtYoloV5/datasets/coco2017/test2017"
output_file = 'image_paths.txt'

# 图像文件扩展名列表
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

with open(output_file, 'w') as file:
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, filename)
                print(file_path)  # 打印路径
                file.write(file_path + '\n')

print(f'所有图像文件路径已保存到 {output_file}')
