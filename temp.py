import cv2
import numpy as np

# 加载图像
image = cv2.imread('/data1/JM/code/BrushNet-main/17a.jpg')

# 创建掩码，找到所有像素为 (127, 127, 127) 的位置
mask = np.all(image == [127, 127, 127], axis=-1)

# 创建一个与原图像相同大小的全零数组
image_modified = np.zeros_like(image)

# 将这些像素的值设为 (255, 255, 255)
image_modified[mask] = [255, 255, 255]

# 保存修改后的图像
cv2.imwrite('/data1/JM/code/BrushNet-main/mask.png', image_modified)
