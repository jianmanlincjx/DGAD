# import os
# import json

# fold_list = sorted(os.listdir('/data1/JM/code/BrushNet-main/datasets/FBP_img_src'))

# for fold in fold_list:
#     # Set the directories for the four folders
#     background_dir = f"/data1/JM/code/BrushNet-main/datasets/FBP_img_src/{fold}/background"
#     foreground_dir = f"/data1/JM/code/BrushNet-main/datasets/FBP_img_src/{fold}/foreground"
#     groundtruth_dir = f"/data1/JM/code/BrushNet-main/datasets/FBP_img_src/{fold}/groundtruth"
#     mask_dir = f"/data1/JM/code/BrushNet-main/datasets/FBP_img_src/{fold}/mask"

#     # Get the list of image files (assuming filenames are identical across the directories)
#     background_files = sorted(os.listdir(background_dir))
#     foreground_files = sorted(os.listdir(foreground_dir))
#     groundtruth_files = sorted(os.listdir(groundtruth_dir))
#     mask_files = sorted(os.listdir(mask_dir))

#     # Ensure the number of files are consistent
#     assert len(background_files) == len(foreground_files) == len(groundtruth_files) == len(mask_files)

#     # Prepare the list of dictionaries to store the data
#     data = []

#     # Loop over the image files and create the dictionary for each
#     for i in range(len(background_files)):
#         entry = {
#             "background": os.path.join(background_dir, background_files[i]),
#             "foreground": os.path.join(foreground_dir, foreground_files[i]),
#             "groundtruth": os.path.join(groundtruth_dir, groundtruth_files[i]),
#             "mask": os.path.join(mask_dir, mask_files[i]),
#             "text": "Masterpiece, ultra-detailed, high-resolution"
#         }
#         data.append(entry)

#     # Define the output JSON file
#     output_json_path = f"datasets/FBP_img_src/{fold}/images_data.json"

#     # Write the data to the JSON file
#     with open(output_json_path, "w") as json_file:
#         for entry in data:
#             json.dump(entry, json_file)
#             json_file.write("\n")

# import os

# # 定义输入文件路径列表
# input_files = [
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000000/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000001/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000002/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000003/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000004/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000005/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000006/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000007/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000008/images_data.json",
#     "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000009/images_data.json"
# ]

# # 定义输出文件路径
# output_file = "/data1/JM/code/BrushNet-main/datasets/FBP_img_src/images_data.json"

# # 打开输出文件，准备写入
# with open(output_file, 'w') as outfile:
#     for file_path in input_files:
#         if os.path.exists(file_path):
#             with open(file_path, 'r') as infile:
#                 # 将当前文件的内容写入输出文件
#                 outfile.write(infile.read())
#         else:
#             print(f"文件 {file_path} 不存在，跳过。")

# print(f"合并完成，结果已保存到 {output_file}")