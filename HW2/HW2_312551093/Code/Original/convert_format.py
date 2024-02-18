import json
import os

# 設定圖片的尺寸
image_width, image_height = 1920, 1080  # 根據您的圖片尺寸調整

# 指定標註文件所在的資料夾
annotation_folder = './HW2_ObjectDetection_2023/val_labels'

# 初始化COCO格式的數據結構
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "car"}]  # 根據您的類別調整
}

# 用於跟踪圖片和標註的ID
image_id = 0
annotation_id = 0

# 遍歷資料夾中的每個標註文件
for filename in os.listdir(annotation_folder):
    if filename.endswith('.txt'):
        # 獲取不包含擴展名的文件名
        file_base_name = os.path.splitext(filename)[0]

        # 處理每個標註文件
        with open(os.path.join(annotation_folder, filename), 'r') as file:
            lines = file.readlines()
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.split())

                # 計算絕對像素座標
                x_min = (x_center - width / 2) * image_width
                y_min = (y_center - height / 2) * image_height
                abs_width = width * image_width
                abs_height = height * image_height

                # 添加到COCO標註
                coco_format['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, abs_width, abs_height],
                    "area": abs_width * abs_height,
                    "iscrowd": 0
                })
                annotation_id += 1

        # 添加圖片信息
        coco_format['images'].append({
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": file_base_name + '.jpg'  # 假設圖片是JPG格式
        })
        image_id += 1

# 將COCO格式數據寫入JSON文件
with open('val_labels.json', 'w') as outfile:
    json.dump(coco_format, outfile, indent=4)
