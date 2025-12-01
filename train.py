# from ultralytics import YOLO
# import os

# DATA_YAML_PATH = "./roboflow_dataset/data.yaml"  
# MODEL_WEIGHTS = "yolo11s.pt"  
# TRAIN_OUTPUT_DIR = "./runs/detect/train"  

# def train_yolo11():
#     # 加载YOLO11预训练模型（自动下载权重）
#     model = YOLO(MODEL_WEIGHTS)
    
#     results = model.train(
#         data=DATA_YAML_PATH, 
#         epochs=100,  
#         batch=32,  
#         imgsz=640,  # 输入图片尺寸
#         save=True,  # 保存训练过程中的最佳模型
#         save_period=-1,  # 不按周期保存（仅保存最佳和最后一轮）
#         pretrained=True,  
#         optimizer="auto",  
#         augment=True,  
#         device=0,  
#         verbose=True,  
#     )
    
#     print("训练完成！结果保存路径：", TRAIN_OUTPUT_DIR)
#     return results

# if __name__ == "__main__":
#     train_yolo11()

# Tune hyperparameters 
# import wandb

from ultralytics import YOLO

# wandb.init(project="YOLO-Tuning")

# Load YOLO model
model = YOLO("yolo11s.pt")

# Tune hyperparameters
result_grid = model.tune(data="/root/yolo11/roboflow_dataset/data.yaml", epochs=50, device=0, use_ray=True)
best_result = result_grid.get_best_result(metric="metrics/mAP50(B)", mode="max")

best_config = best_result.config

print("\n最佳超参数配置:")
for k, v in best_config.items():
    if k in ['lr0', 'lrf', 'momentum', 'weight_decay', 'box', 'cls', 'mosaic', 'degrees', 'hsv_h']:
        print(f"  {k}: {v}")

best_checkpoint_path = best_result.checkpoint.path
print(f"\n最佳模型权重路径 (Checkpoint): {best_checkpoint_path}/weights/best.pt")