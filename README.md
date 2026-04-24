# 번호판 OCR 감시 시스템 (License Plate OCR Audit System)

YOLO 탐지 + CRNN OCR을 이용한 번호판 인식 통합 감시 및 자동 검증 시스템입니다.

## 📋 프로젝트 구조

```
.
├── ocr_audit_fixed.py          # 메인 스크립트
├── models/                      # 학습된 모델 저장 폴더
│   ├── YOLO_best.pt           # YOLO 번호판 탐지 모델
│   └── Colab_best_0420_v2.pt  # CRNN OCR 인식 모델
├── Raw_data/               # 검증할 원본 이미지 폴더
└── fail_dataset/               # 인식 실패 이미지 자동 저장 폴더
```

## 🚀 설치 및 실행

### 1. 필수 라이브러리 설치

```bash
pip install torch torchvision
pip install opencv-python pillow
pip install ultralytics
pip install numpy
```

### 2. 폴더 구조 생성

```bash
mkdir models
mkdir Raw_data
mkdir fail_dataset
```

### 3. 모델 파일 추가

- `models/YOLO_best.pt` - YOLO v8 기반 번호판 탐지 모델
- `models/Colab_best_0420_v2.pt` - CRNN 기반 번호판 문자 인식 모델

#### 모델 다운로드
- https://drive.google.com/file/d/1YlU8-kFG_PAKCO_Q7qFDyH1SbBVBrhTJ/view?usp=sharing

### 4. 검증 이미지 추가

`Raw_data/` 폴더에 검증할 이미지 파일들을 추가합니다.

#### 검증 이미지 다운로드
 - https://drive.google.com/file/d/1mrcl4hykdVKQJIW5BfQk8Z9zYwRTdeK7/view?usp=sharing

**파일명 형식**: `[정답번호판].jpg`, `[정답번호판]-000.jpg` 
 예시) 12가1234, 123가1234-1
* 구형 초록색 번호판 및 지역명이 포함된 노란색 번호판, 전기차/법인차 등 색깔이 입혀진 번호판은 취급X
* 7자 혹은 8자인 흰색 신형 번호판(홀로그램은 상관X)만을 취급

### 5. 실행

```bash
python OCR_Test.py
```

## 📊 주요 기능

### 1. YOLO 번호판 탐지
- 이미지에서 번호판 영역을 자동 감지
- 탐지된 영역 주변 15% 상하, 5% 좌우 여백 포함 크롭

### 2. CRNN OCR 인식
- 320x96 해상도로 정규화된 번호판 이미지 입력
- 한글 + 숫자 조합 인식
- 신뢰도(Confidence) 점수 계산

### 3. 규칙 기반 디코딩
- 한글 문자 위치를 기준점으로 설정
- 앞자리: 2~3개 숫자 추출
- 중간: 1개 한글 문자
- 뒷자리: 최대 4개 숫자 추출
- 맨 앞 '0' 제거 등 휴리스틱 적용

### 4. 자동 검증 및 리포트
- 정답 vs 예측값 비교
- 실패 사례 자동 저장
- 시각화된 오류 검토 인터페이스 (UI 오버레이)

## 🔍 출력 결과

```
🚀 통합 전수조사 시작 (대상: 1000장)...
--- 진행 중: [50/1000] (현재 성공률: 98.0%) ---
--- 진행 중: [100/1000] (현재 성공률: 97.5%) ---
...
========================
🏁 조사 완료 | 총 소요시간: 145.3초
📈 최종 성공률: 97.23% | 실패 건수: 28건
========================

[실패 사례 요약 리스트]
No. | 파일명                   | 정답     | 예측     | 사유
1   | 서울12가1234_001.jpg     | 12가1234 | 12가1235 | 규칙 미준수
2   | 부산34나5678_002.jpg     | 34나5678 | 미검출   | YOLO 실패
...
```

### 실패 사례 이미지 검토
- 엔터: 다음 실패 사례 보기
- `q`: 검토 종료

## ⚙️ 경로 구조 설명

스크립트는 실행 위치를 기준으로 자동 경로 설정합니다:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL = os.path.join(BASE_DIR, "models", "YOLO_best.pt")
OCR_MODEL = os.path.join(BASE_DIR, "models", "Colab_best_0420_v2.pt")
RAW_DATA_DIR = os.path.join(BASE_DIR, "Raw_data")
FAIL_DIR = os.path.join(BASE_DIR, "fail_dataset")
```

**Windows 경로 예시**:
```
C:\Users\YourName\Desktop\project\
├── ocr_audit_fixed.py
├── models\YOLO_best.pt
├── Raw_data\image1.jpg
└── fail_dataset\ (자동 생성)
```

## 🔧 주요 파라미터

`run_integrated_audit_report()` 함수의 기본값:

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `yolo_path` | - | YOLO 모델 경로 |
| `ocr_path` | - | OCR 모델 경로 |
| `img_dir` | - | 검증 이미지 폴더 |
| `fail_dir` | - | 실패 이미지 저장 폴더 |
| `num_test` | 1000 | 검증할 이미지 개수 |

## 💡 문제 해결

### 1. "YOLO 모델을 찾을 수 없습니다" 에러
```
❌ YOLO 모델을 찾을 수 없습니다: C:\project\models\YOLO_best.pt
```
→ `models/` 폴더에 `YOLO_best.pt` 파일이 있는지 확인하세요.

### 2. "원본 데이터 폴더를 찾을 수 없습니다" 에러
→ `Raw_data/` 폴더를 생성하고 이미지를 추가하세요.

### 3. CUDA 부족 에러
```python
# 스크립트는 자동으로 CPU로 전환됩니다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 4. Windows 한글 폰트 미설정
→ `draw_ui()` 함수에서 자동으로 기본 폰트로 폴백됩니다.

## 📈 성능 최적화 팁

1. **배치 처리**: 대량 이미지는 스크립트를 여러 번 실행 (랜덤 샘플링)
2. **GPU 사용**: CUDA 가능한 GPU로 5~10배 속도 향상
3. **이미지 품질**: 최소 640x480 이상의 해상도 권장

## 📝 라이센스

이 프로젝트는 내부 연구/개발 용도입니다.

## 👨‍💻 작성자

Created with YOLO v8 + CRNN + PyTorch

---

**마지막 수정**: 2024-04-20
**파이썬 버전**: 3.7 이상
