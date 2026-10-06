# 한국인 표정 인식 AI

ResNet18로 분노, 슬픔, 기쁨, 중립을 분류하고 Streamlit에서 업로드 이미지를 예측하는 프로젝트입니다.
Python 3.12 64비트 설치 후 app.py가 있는 저장소 루트에서 PowerShell 명령을 실행합니다.
이미지 데이터와 학습된 모델은 저장소에 포함하지 않습니다. 실제 학습 성능과 웹 추론은 아직 검증하지 않았습니다.
빠른 실행 순서는 [시작안내.md](시작안내.md)를 참고하세요.

먼저 별도로 받은 실습 자료에서 `제공된 데이터셋/data.zip`을 꺼내 app.py 옆에 놓으세요.
실습 자료가 없다면 아래 라벨 형식에 맞는 데이터를 별도로 준비해야 합니다.
저장소에는 데이터 다운로드 링크나 재배포 권한을 제공하지 않습니다.

```powershell
Expand-Archive -LiteralPath .\data.zip -DestinationPath .\data_unpack
```

압축 해제 후 이미지 위치는 `data_unpack/data/dataset/`입니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe 01_data_inspection.py --data-dir .\data_unpack\data\dataset --label-scheme provided-archive --no-show
```

다른 곳에 이미지를 준비했다면 `--data-dir`를 해당 이미지 폴더 경로로 변경하세요.
경로를 지정하지 않으면 현재 폴더 아래 이미지를 재귀 탐색합니다.

라벨: 0 분노, 1 슬픔, 2 기쁨, 3 중립.
첨부 데이터의 원래 라벨은 0 분노, 2 기쁨, 3 중립, 4 슬픔입니다.
첨부 노트북의 emotion_map으로 확인했으며, 파일 수는 각각 7,289 / 7,339 / 7,228 / 7,311개입니다.
`--label-scheme provided-archive`는 분석할 때만 4를 1로 매핑합니다.
CSV에는 raw_label과 매핑 후 label을 함께 기록합니다. 원본 파일명은 변경하지 않습니다.
이미 요청한 0~3 라벨로 구성된 데이터에는 이 옵션을 생략하세요.
첨부 데이터의 `0_이름.jpg` 형식과 `0/이미지.jpg` 클래스 폴더 형식을 지원합니다.
폴더 라벨이 있으면 파일명 라벨보다 우선합니다.

기본 결과 폴더 `inspection_results/`에 다음 파일이 생성됩니다.
- `class_counts.png`: 클래스별 전체 파일 수 및 읽기 가능한 이미지 수
- `sample_images.png`: 클래스별 무작위 샘플 5장 (시드 42)
- `image_audit.csv`: 파일별 라벨, 크기, 원본 색상 모드, 이상 및 오류

`--no-show`를 추가하면 창 없이 저장합니다. `--samples 8`로 샘플 수를 변경할 수 있습니다.
도표는 폰트 호환성을 위해 영어 클래스명을 사용하며 콘솔과 CSV에는 한국어명을 표시합니다.
이미지 크기나 모드가 잘못된 경우에도 읽을 수 있으면 샘플 후보에 포함됩니다.
샘플 표시 시에만 흑백 변환하며 원본 파일은 수정하지 않습니다.

requirements.txt는 버전 범위를 지정합니다. 설치 완료 후 재현 가능한 환경을 보관하려면
`.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt`를 실행하세요.
GPU 사용 시에는 https://pytorch.org/get-started/locally/ 에서 시스템에 맞는 설치 명령을 확인하세요.

## Train / Validation / Test 분할

```powershell
.\.venv\Scripts\python.exe 02_data_split.py --data-dir .\data_unpack\data\dataset --label-scheme provided-archive --output-dir dataset
```

클래스별로 시드 42를 사용해 섞은 후 70%/15%/15%로 층화 분할합니다.
소수점 수량은 최대 잔여 방식으로 배정하므로 이미지 수의 정수 반올림에 따른 미세한 차이가 있습니다.
Python 표준 라이브러리만 사용합니다. 원본 이미지는 복사하거나 이동하지 않습니다.
`dataset/train.csv`, `dataset/val.csv`, `dataset/test.csv`에는 절대 이미지 경로(path),
학습 라벨(label), 한국어 클래스명(class_name), 원본 라벨(raw_label)이 저장됩니다.
`dataset/split_summary.json`에는 세트별 수량, 비율, 클래스별 수량, 시드가 저장됩니다.
같은 입력과 시드로 실행하면 같은 분할이 생성됩니다. 기존 결과를 갱신하려면 `--overwrite`를 지정합니다.
요청한 0~3 라벨의 데이터에는 `--label-scheme` 옵션을 생략하세요.
읽기 가능한 파일과 96x96 흑백 여부는 먼저 01_data_inspection.py로 확인하세요.
분할 스크립트는 파일 확장자와 라벨만 검사하며 이미지 픽셀을 다시 검사하지 않습니다.
동일 인물 식별 정보가 없는 상태에서 이미지 단위로 분할하므로 동일 인물이 여러 세트에 포함될 수 있습니다.
CSV 경로는 현재 컴퓨터 기준이므로 데이터 위치를 변경하면 다시 생성하세요.

## ResNet18 전이 학습 및 평가

```powershell
.\.venv\Scripts\python.exe 03_train_evaluate.py --epochs 30 --batch-size 16 --no-show
```

기본 CSV 위치는 스크립트 옆 `dataset/`이고 결과는 `training_results/`에 저장됩니다.
`--csv-dir`와 `--output-dir`로 변경할 수 있습니다. CUDA가 있으면 GPU, 없으면 CPU를 사용합니다.
첫 학습 실행은 ImageNet 사전 학습 가중치를 다운로드하므로 인터넷 연결이 필요합니다.
ResNet18 전체를 기본적으로 미세 조정하며 `--freeze-backbone`이면 FC만 학습합니다.
마지막 FC는 512→4로 교체합니다. CSV의 label(0 분노 / 1 슬픔 / 2 기쁨 / 3 중립)을 그대로 사용합니다.
흑백 이미지는 RGB 세 채널로 복제한 뒤 224x224로 리사이즈하고 ImageNet 평균/표준편차로 정규화합니다.
학습 데이터에만 무작위 좌우 반전을 적용합니다. Validation/Test에는 무작위 증강을 적용하지 않습니다.
참고: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18

손실은 CrossEntropyLoss, 최적화는 AdamW(lr=0.001, weight_decay=0.01)입니다.
매 epoch에 샘플 수로 가중 평균한 Train/Val Loss와 Accuracy를 출력하고 history.csv에 기록합니다.
Accuracy는 0~1 값으로 저장됩니다. Val Loss가 엄격히 감소하면 best_model.pth를 갱신합니다.
7 epoch 동안 개선이 없으면 조기 종료합니다. `--patience 0`으로 끌 수 있습니다.
Train Loss 감소와 Val Loss 증가가 3 epoch 연속 관측되고 Train/Val Accuracy 차이가 5%p 이상이면
과대적합 의심 경고를 출력하고 기록합니다. 이는 휴리스틱이며 과대적합을 확정하는 진단은 아닙니다.
`--overfit-patience`, `--overfit-gap`으로 조정합니다.
Train Accuracy는 증강·학습 모드에서 측정되므로 Val과 측정 조건이 다릅니다.
Test는 학습 완료 후 최적 가중치를 다시 로드하여 평가하고 모델 선택에는 사용하지 않습니다.

생성되는 결과:
- best_model.pth: 모델 state_dict만 저장 (기존 첨부 모델과 다른 출력 폴더)
- best_model_metadata.json / run_config.json: 최적 epoch, 라벨, 전처리, 실행 설정
- history.csv / learning_curves.png: Train/Val Loss 및 Accuracy 기록과 그래프
- test_metrics.json: Test Loss/Accuracy와 클래스별 precision/recall/F1
- test_predictions.csv: 테스트 이미지별 정답과 예측
- confusion_matrix.png / confusion_matrix.csv: 행=정답, 열=예측, 클래스 순서 0~3

`--no-show`를 생략하면 그래프 창도 표시합니다. 도표 클래스명은 폰트 호환성을 위해 영어를 사용합니다.
기존 결과 폴더에 새 학습을 실행하려면 다른 출력 경로 또는 `--overwrite`를 사용합니다.
재평가: `.\.venv\Scripts\python.exe 03_train_evaluate.py --evaluate-only --no-show`.
사전 학습 다운로드 없이 저장한 가중치로 테스트합니다. 중단된 학습 재개 기능은 포함하지 않습니다.
Windows DataLoader는 기본 workers=0이며 필요 시 `--workers 2`로 변경합니다.
시드 42를 설정하지만 하드웨어·라이브러리 버전별 완전 동일한 결과를 보장하지는 않습니다.
코드 실행에는 requirements.txt의 라이브러리 설치가 필요합니다.

## Streamlit 이미지 업로드 서비스

학습 완료 후 app.py가 있는 폴더에서 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

앱은 기본적으로 `app.py` 옆 `training_results/best_model.pth`를 사용합니다.
03_train_evaluate.py가 저장한 ResNet18 state_dict(FC=4개 클래스)를 사용해야 합니다.
첨부 ZIP의 기존 best_model.pth는 구조와 라벨 순서가 검증되지 않았으므로 자동으로 사용하지 않습니다.
가중치가 없는 경우 앱은 학습된 모델을 준비하라는 안내를 표시합니다.
다른 경로의 가중치를 사용하려면 실행 전에 설정하세요.

```powershell
$env:EXPRESSION_MODEL_PATH = '.\my_model\best_model.pth'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

best_model_metadata.json이 같은 폴더에 있으면 모델 구조, 클래스 순서, 입력 크기와 정규화 설정을 확인합니다.
모델은 CPU에서 eval/inference_mode로 추론하며 st.cache_resource로 캐시합니다.
모델 또는 메타데이터 변경 시 다음 앱 실행에서 캐시를 갱신합니다.
가중치 로드 시 weights_only=True를 사용하며 추론 시 외부 모델 다운로드는 하지 않습니다.

업로드는 JPG/PNG 한 장, 최대 10MB입니다. EXIF 방향을 보정하고 PNG 투명 영역을 흰색 배경으로 처리합니다.
흑백 변환 → 96x96 Resize → RGB 세 채널 복제 → 224x224 Resize → ImageNet 정규화 순서로 처리합니다.
원본 학습 데이터가 96x96이고 학습 코드는 이를 224x224로 확장하므로 동일한 최종 모델 입력을 사용합니다.
얼굴을 자동 검출하거나 자르지 않으므로 얼굴 영역이 크게 보이는 사진을 업로드하세요.
라벨 순서: 0 분노 / 1 슬픔 / 2 기쁨 / 3 중립. Confidence는 네 클래스 logits의 softmax 확률입니다.
앱은 업로드 이미지를 디스크에 저장하지 않습니다.
참고: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
및 https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource

## GitHub 업로드

코드 4개, requirements.txt, README.md, 시작안내.md, .gitignore를 저장소 루트에 올립니다.
데이터, 가상환경, 분할 CSV, 가중치 및 결과 파일은 .gitignore로 제외합니다.
분할 CSV는 컴퓨터별 절대 경로를 사용하므로 새 컴퓨터에서는 반드시 다시 생성하세요.
