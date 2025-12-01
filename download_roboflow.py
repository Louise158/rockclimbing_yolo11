import os
from dotenv import load_dotenv  
import roboflow

load_dotenv()  
API_KEY = os.getenv("ROBOFLOW_API_KEY")  

PROJECT_ID = "climbing-hold-detection-xb8yz-nepcq" 
VERSION = 1
DOWNLOAD_DIR = "./roboflow_dataset" 

def download_roboflow_dataset():
    try:
        rf = roboflow.Roboflow(api_key=API_KEY)
        
        project = rf.workspace().project(PROJECT_ID)
        dataset_version = project.version(VERSION)
        
        version = project.version(1)
        dataset = version.download("yolov11")
        
        print(f"数据集下载完成！保存路径：{DOWNLOAD_DIR}")
        print(f"数据集结构：{os.listdir(DOWNLOAD_DIR)}")
        return dataset  
    
    except Exception as e:
        print(f"下载失败：{str(e)}")
        raise

if __name__ == "__main__":
    download_roboflow_dataset()