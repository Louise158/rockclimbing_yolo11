from ultralytics import YOLO
import cv2

# 加载训练好的最佳模型
model = YOLO("./runs/detect/train/best.pt")

# 预测单张图片
img_path = "./test_image.jpg"  # 待预测图片路径
results = model(img_path, conf=0.5)  # conf=0.5：置信度阈值（只显示≥0.5的预测结果）

# 可视化预测结果（自动标注边界框和类别）
results[0].show()  # 显示图片
results[0].save("predicted_image.jpg")  # 保存预测结果图

# 打印预测详情（边界框、类别、置信度）
for box in results[0].boxes:
    cls = box.cls[0].item()  # 类别索引
    conf = box.conf[0].item()  # 置信度
    xyxy = box.xyxy[0].tolist()  # 边界框（x1,y1,x2,y2）
    print(f"类别：{model.names[cls]}，置信度：{conf:.2f}，边界框：{xyxy}")