from ultralytics import YOLO
import os

DATA_YAML_PATH = "./roboflow_dataset/data.yaml"  
MODEL_WEIGHTS = "yolo11s.pt"  
TRAIN_OUTPUT_DIR = "./runs/detect/train"  

def train_yolo11():
    model = YOLO(MODEL_WEIGHTS)
    best_hyps = {
    "bgr": 0.5293711521903384,
    "box": 0.16975554922161787,
    "cls": 3.2295624955343256,
    "copy_paste": 0.867367787675641,
    "cutmix": 0.43297569354159104,
    "degrees": 23.176804848970594,
    "fliplr": 0.23210974879428836,
    "flipud": 0.19059821506417463,
    "hsv_h": 0.06139939481364193,
    "hsv_s": 0.05513901668336191,
    "hsv_v": 0.21518030782159472,
    "lr0": 0.04033887575209577,
    "lrf": 0.2420426397383319,
    "mixup": 0.20076722089303467,
    "momentum": 0.7694183136536308,
    "mosaic": 0.5828386512708702,
    "perspective": 0.00012793613511604386,
    "scale": 0.1843137892219961,
    "shear": 6.7178777130880265,
    "translate": 0.037858759565116716,
    "warmup_epochs": 3.832815586287619,
    "warmup_momentum": 0.4113564450875246,
    "weight_decay": 0.000753226603853445
    }
    
    results = model.train(
        data=DATA_YAML_PATH, 
        epochs=100,  
        batch=16,  
        imgsz=640,  
        save=True, 
        save_period=-1,  
        pretrained=True,  
        optimizer="auto",  
        augment=True,  
        device=0,  
        verbose=True,  
        plots=True,
        **best_hyps
    )
    
    print("训练完成！结果保存路径：", TRAIN_OUTPUT_DIR)
    return results

if __name__ == "__main__":
    train_yolo11()

# Tune hyperparameters 
# import wandb

# from ultralytics import YOLO

# # 用你自己的数据配置文件（data.yaml），并加载 yolo11s 权重（或 yolo11s.pt）
# model = YOLO("yolo11s.pt")

# # 开始调参（使用 Ray Tune）
# # max_samples 控制试验数，grace_period 控制 ASHA 的最小 epoch，gpu_per_trial 可设为1（若多GPU调整）
# result_grid = model.tune(
#     use_ray=True,
#     iterations=20,
#     data="/root/yolo11/roboflow_dataset/data.yaml",
#     gpu_per_trial=1,
#     epochs=60,
#     pretrained="yolo11s.pt",
#     name="tune_yolo11s_run1",
# )
# # result_grid 包含所有trial信息
# for i, res in enumerate(result_grid):
#     print(f"Trial {i}: config={res.config}, metrics={res.metrics}")

# best_result = result_grid.get_best_result(metric="metrics/mAP50(B)", mode="max")

# best_config = best_result.config

# print("\n最佳超参数配置:")
# for k, v in best_config.items():
#     if k in ['lr0', 'lrf', 'momentum', 'weight_decay', 'box', 'cls', 'mosaic', 'degrees', 'hsv_h']:
#         print(f"  {k}: {v}")

# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # 加载模型（仅需与原始 run 环境相同）
# model = YOLO("yolo11s.pt")

# # 恢复先前的 tuning 运行并获取 ResultGrid
# result_grid = model.tune(
#     use_ray=True,
#     data="/root/yolo11/roboflow_dataset/data.yaml",
#     epochs=60,
#     gpu_per_trial=1,
#     name="tune_yolo11s_run1",
#     resume=True,
#     space={},  
# )


# for i, result in enumerate(result_grid):
#     df = result.metrics_dataframe
#     if "training_iteration" in df.columns and "metrics/mAP50(B)" in df.columns:
#         print(result)
#         plt.plot(
#             df["training_iteration"],
#             df["metrics/mAP50(B)"],
#             label=f"Trial {19-i} (mAP50={result.metrics['metrics/mAP50(B)']:.3f})"
#         )

# plt.xlabel("Training Epochs / Iterations")
# plt.ylabel("mAP50(B)")
# plt.title("Hyperparameter Tuning Progress (mAP50)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# output_path = "/root/yolo11/runs/detect/tune_yolo11s_run1/tuning_results.png"
# plt.savefig(output_path, dpi=150)

# # 找到最佳 trial
# best_result = None
# best_metric = -1
# for result in result_grid:
#     if "metrics/mAP50(B)" in result.metrics:
#         val = result.metrics["metrics/mAP50(B)"]
#         if val > best_metric:
#             best_metric = val
#             best_result = result

# if best_result:
#     print(best_result)
