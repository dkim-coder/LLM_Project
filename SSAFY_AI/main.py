import os
import cv2
import torch
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 난수 생성을 위한 시드 값 세팅
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything()

# MPS 장치 설정 (Mac에서 Metal Performance Shaders 사용)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS 장치 사용 가능")
else:
    device = torch.device("cpu")
    print("MPS 장치 사용 불가능, CPU 사용")


# 데이터 불러오기
DATA_YAML_PATH = "./input/pothole-detection-challenge/data.yaml"

with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

DATASET_PATH = os.path.dirname(DATA_YAML_PATH)
TRAIN_IMAGES = os.path.join(DATASET_PATH, data_yaml["train"].replace("../", ""))
VALID_IMAGES = os.path.join(DATASET_PATH, data_yaml["val"].replace("../", ""))

print(DATASET_PATH)
print(TRAIN_IMAGES)
print(VALID_IMAGES)

# 데이터셋 클래스 정의 및 로드
class PotholeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        all_image_paths = glob(os.path.join(image_dir, "*.jpg"))
        self.image_paths = []

        for path in all_image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                self.image_paths.append(path)
            except Exception as e:
                print(f"[손상된 이미지 제거] {path} - {e}")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.transform(img) if self.transform else img

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

train_loader = DataLoader(PotholeDataset(TRAIN_IMAGES, transform), batch_size=16, shuffle=True)
valid_loader = DataLoader(PotholeDataset(VALID_IMAGES, transform), batch_size=16, shuffle=False)

# YOLO11x 사전 학습 모델 불러와서 미세조정
model = YOLO("yolo11x.pt")

# 데이터 증강을 위한 하이퍼 파라미터 처리 작성 필요
def train_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=150,
        imgsz=1024,
        batch=-1,    # 배치 크기세 가지 모드가 있습니다(예: 정수로 설정), batch=16), 60% GPU 메모리 사용률의 자동 모드(batch=-1) 또는 지정된 사용률 비율의 자동 모드(batch=0.70).
        # workers=4,    # 데이터 로딩을 위한 워커 스레드 수(당 RANK 다중GPU 훈련인 경우). 데이터 전처리 및 모델에 공급하는 속도에 영향을 미치며, 특히 다중GPU 설정에서 유용합니다.
        device=device,
        project="./working",
        name="pothole_yolo11x_train",
        patience=10,        # 훈련을 조기 중단하기 전에 검증 지표의 개선 없이 기다려야 하는 에포크 수입니다. 성능이 정체될 때 훈련을 중단하여 과적합을 방지합니다.
        save_period=50,     # 모델 체크포인트 저장 빈도(에포크 단위로 지정)입니다. 값이 -1이면 이 기능이 비활성화됩니다. 긴 훈련 세션 동안 중간 모델을 저장할 때 유용합니다.
        amp=True,   # 자동 혼합 정밀도 (AMP) 훈련을 활성화하여 메모리 사용량을 줄이고 정확도에 미치는 영향을 최소화하면서 훈련 속도를 높일 수 있습니다.
        rect=True,   # 직사각형 학습을 활성화하여 배치 구성을 최적화하여 패딩을 최소화합니다. 효율성과 속도를 향상시킬 수 있지만 모델 정확도에 영향을 줄 수 있습니다.
        overlap_mask=False,  # 학습을 위해 개체 마스크를 하나의 마스크로 병합할지, 아니면 각 개체마다 별도로 유지할지 결정합니다. 겹치는 경우 병합하는 동안 작은 마스크가 큰 마스크 위에 겹쳐집니다.
        # resume=True,     # 마지막으로 저장한 체크포인트부터 훈련을 재개합니다. 모델 가중치, 최적화 상태 및 에포크 수를 자동으로 로드하여 훈련을 원활하게 계속합니다.
        optimizer="AdamW",   # 교육용 옵티마이저 선택. 옵션은 다음과 같습니다. SGD, Adam, AdamW, NAdam, RAdam, RMSProp 등 또는 auto 를 사용하여 모델 구성에 따라 자동으로 선택할 수 있습니다. 컨버전스 속도와 안정성에 영향을 줍니다.
        cache=True,      # 메모리에서 데이터 세트 이미지의 캐싱을 활성화합니다(True/ram), 디스크(disk) 또는 비활성화(False). 메모리 사용량을 늘리는 대신 디스크 I/O를 줄여 훈련 속도를 향상시킵니다.
        multi_scale=True,    # 교육 규모를 늘리거나 줄여 멀티스케일 교육 가능 imgsz 최대 0.5 를 추가합니다. 모델을 여러 번 훈련하여 정확도를 높입니다. imgsz 추론하는 동안
        auto_augment="autoaugment"
    )
    return results

results = train_model()

# 모델 성능 평가
model = YOLO("./working/pothole_yolo11x_train/weights/best.pt")

val_results = model.val(data=DATA_YAML_PATH, split="val")

print("검증 데이터 평가 결과:")
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall: {val_results.box.mr:.4f}")

# 제출 파일 생성
model = YOLO("./working/pothole_yolo11x_train/weights/best.pt")

TEST_IMG_DIR = "./input/pothole-detection-challenge/test/images"
test_image_paths = sorted(glob(os.path.join(TEST_IMG_DIR, "*.jpg")))

submission_rows = []

for img_path in test_image_paths:
    image_id = os.path.basename(img_path)

    if cv2.imread(img_path) is None:
        print(f"이미지 로드 실패: {image_id}")
        submission_rows.append({
            "ImageId": image_id,
            "ClassId": 0,
            "X": 0,
            "Y": 0,
            "Width": 0,
            "Height": 0,
        })
        continue

    results = model.predict(source=img_path, conf=0.25, imgsz=1024, save=False)
    result = results[0]

    if len(result.boxes) > 0:
        boxes = result.boxes
        best_idx = boxes.conf.argmax().item()
        cls_id = int(boxes.cls[best_idx].item())
        cx, cy, w, h = boxes.xywhn[best_idx].tolist()

        submission_rows.append({
            "ImageId": image_id,
            "ClassId": cls_id,
            "X": round(cx, 6),
            "Y": round(cy, 6),
            "Width": round(w, 6),
            "Height": round(h, 6),
        })
    else:
        submission_rows.append({
            "ImageId": image_id,
            "ClassId": 0,
            "X": 0,
            "Y": 0,
            "Width": 0,
            "Height": 0,
        })

submission_df = pd.DataFrame(submission_rows, columns=["ImageId", "ClassId", "X", "Y", "Width", "Height"])
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"제출 파일 저장 완료: {submission_path}")
