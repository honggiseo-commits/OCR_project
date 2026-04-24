import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import random
import re
import time
import shutil
import unicodedata
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --- 기본 경로 설정 (Python 파일의 디렉토리를 기준) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 경로 설정
YOLO_MODEL = os.path.join(BASE_DIR, "models", "YOLO_best.pt")
OCR_MODEL = os.path.join(BASE_DIR, "models", "Colab_best_0420_v2.pt")
RAW_DATA_DIR = os.path.join(BASE_DIR, "Raw_data")
FAIL_DIR = os.path.join(BASE_DIR, "fail_dataset")

# --- 1. OCR 모델 구조 (고도화 버전 320x96) ---
class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBnRelu(1, 64), nn.MaxPool2d(2, 2),
            ConvBnRelu(64, 128), nn.MaxPool2d(2, 2),
            ConvBnRelu(128, 256), ConvBnRelu(256, 256), nn.MaxPool2d(2, 2),
            ConvBnRelu(256, 512), ConvBnRelu(512, 512), nn.MaxPool2d((2, 1), (2, 1)),
            ConvBnRelu(512, 512), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2).permute(0, 2, 1)
        out, _ = self.rnn(feat)
        return self.fc(out).log_softmax(2)

# --- 2. 규칙 기반 디코딩 엔진 ---
def decode_with_rule_engine(logits, idx2char):
    probs = torch.exp(logits)
    max_probs, indices = probs.max(dim=1)
    blank_idx = 0
    decoded_objects, prev = [], None
    
    for i in range(len(indices)):
        idx = indices[i].item()
        if idx != blank_idx and idx != prev:
            char = idx2char.get(idx, "")
            conf = max_probs[i].item()
            decoded_objects.append({
                'char': char, 'conf': conf, 'pos': i,
                'type': 'hangul' if re.match(r'[가-힣]', char) else 'digit'
            })
        prev = idx

    if not decoded_objects: return "미인식", 0.0, "글자 없음"

    hangul_candidates = [obj for obj in decoded_objects if obj['type'] == 'hangul']
    if not hangul_candidates:
        raw_text = "".join([obj['char'] for obj in decoded_objects])
        return raw_text, 0.0, "한글 미검출"

    main_hangul = max(hangul_candidates, key=lambda x: x['conf'])
    pivot_pos = main_hangul['pos']
    front_digits = [obj for obj in decoded_objects if obj['pos'] < pivot_pos and obj['type'] == 'digit']
    back_digits = [obj for obj in decoded_objects if obj['pos'] > pivot_pos and obj['type'] == 'digit']

    if len(back_digits) > 4:
        back_digits.sort(key=lambda x: x['conf'], reverse=True)
        back_digits = sorted(back_digits[:4], key=lambda x: x['pos'])
    
    if len(front_digits) > 3:
        front_digits.sort(key=lambda x: x['conf'], reverse=True)
        front_digits = sorted(front_digits[:3], key=lambda x: x['pos'])

    res_front = "".join([d['char'] for d in front_digits])
    res_back = "".join([d['char'] for d in back_digits])
    
    if len(res_front) == 3 and res_front[0] == '0':
        res_front = res_front[1:]  # 맨 앞의 '0' 제거
    elif len(res_front) == 4: # 만약 볼트 때문에 4개가 되었다면
        res_front = res_front[-3:] # 뒤의 3개만 취함
        
    final_text = f"{res_front}{main_hangul['char']}{res_back}"
    avg_conf = np.mean([obj['conf'] for obj in (front_digits + [main_hangul] + back_digits)]) * 100
    
    reason = "정상"
    if len(res_back) != 4 or len(res_front) < 2: reason = "규칙 미준수"
        
    return final_text, avg_conf, reason

# --- 3. UI 시각화 함수 ---
def draw_ui(img, gt, pred, conf, reason, target_h=500, margin_h=100):
    h_orig, w_orig = img.shape[:2]
    scale = target_h / h_orig
    resized = cv2.resize(img, (int(w_orig * scale), target_h))
    canvas = np.zeros((target_h + margin_h, resized.shape[1], 3), dtype=np.uint8)
    canvas[margin_h:, :] = resized
    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)
    except: font = ImageFont.load_default()
    is_correct = (gt == pred)
    draw.text((20, 15), f"정답: {gt}", font=font, fill=(255, 255, 255))
    color = (150, 150, 255) if is_correct else (255, 100, 100)
    draw.text((20, 45), f"인식: {pred} ({conf:.1f}%) [{reason}]", font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), scale

# --- 4. 메인 전수조사 및 리포트 함수 ---
def run_integrated_audit_report(yolo_path, ocr_path, img_dir, fail_dir, num_test=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(fail_dir, exist_ok=True)
    
    detector = YOLO(yolo_path)
    ckpt = torch.load(ocr_path, map_location=device)
    idx2char = {i: c for c, i in ckpt["char2idx"].items()}
    ocr_model = CRNN(len(ckpt["char2idx"]) + 1).to(device)
    ocr_model.load_state_dict(ckpt["model_state"])
    ocr_model.eval()

    all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    selected_files = random.sample(all_files, min(num_test, len(all_files)))

    success_count = 0
    fail_cases = [] # 상세 정보 저장
    
    print(f"🚀 통합 전수조사 시작 (대상: {len(selected_files)}장)...")
    start_time = time.time()

    for i, filename in enumerate(selected_files):
        img_path = os.path.join(img_dir, filename)
        # 정답 추출 및 정규화
        gt_text = re.sub(r'[-_]\d+$', '', os.path.splitext(filename)[0])
        gt_text = unicodedata.normalize('NFC', gt_text).replace(" ", "")

        nparr = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: continue

        # YOLO 탐지
        results = detector(img, verbose=False)
        pred_text, avg_conf, reason, plate_box = "미검출", 0.0, "YOLO 실패", None

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            plate_box = list(map(int, box.xyxy[0])) # [수정] list로 변환하여 소모 방지
            x1, y1, x2, y2 = plate_box
            
            h_p, w_p = y2 - y1, x2 - x1
            crop = img[max(0, y1-int(h_p*0.15)):min(img.shape[0], y2+int(h_p*0.15)), 
                       max(0, x1-int(w_p*0.05)):min(img.shape[1], x2+int(w_p*0.05))]
            
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            input_img = cv2.resize(gray_crop, (320, 96))
            input_img = (input_img.astype(np.float32) / 127.5) - 1.0
            input_tensor = torch.FloatTensor(input_img).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = ocr_model(input_tensor).squeeze(0)
                pred_text, avg_conf, reason = decode_with_rule_engine(logits, idx2char)

        # 판정 및 저장
        is_success = (gt_text == pred_text)
        if is_success:
            success_count += 1
        else:
            fail_cases.append({
                'filename': filename, 'gt': gt_text, 'pred': pred_text, 
                'conf': avg_conf, 'reason': reason, 'img': img, 'box': plate_box
            })
            shutil.copy(img_path, os.path.join(fail_dir, filename))

        # 진행 상황만 간략히 표시
        if (i+1) % 50 == 0:
            print(f"--- 진행 중: [{i+1}/{len(selected_files)}] (현재 성공률: {(success_count/(i+1))*100:.1f}%) ---")

    # --- 최종 리포트 출력 ---
    total_time = time.time() - start_time
    acc = (success_count / len(selected_files)) * 100
    
    print("\n" + "="*90)
    print(f"🏁 조 사 완 료 | 총 소요시간: {total_time:.1f}초")
    print(f"📈 최종 성공률: {acc:.2f}% | 실패 건수: {len(fail_cases)}건")
    print("="*90)
    
    if fail_cases:
        print(f"\n[실패 사례 요약 리스트]")
        print(f"{'No.':<4} | {'파일명':<30} | {'정답':<10} | {'예측':<10} | {'사유'}")
        print("-" * 90)
        for idx, fc in enumerate(fail_cases):
            # 파일명이 너무 길면 생략 표시
            fname = (fc['filename'][:27] + '..') if len(fc['filename']) > 29 else fc['filename']
            print(f"{idx+1:<4} | {fname:<30} | {fc['gt']:<10} | {fc['pred']:<10} | {fc['reason']}")
        print("-" * 90)
        
        print("\n💡 엔터를 누르면 실패 사례 이미지를 하나씩 검토합니다. (종료: q)")
        for fc in fail_cases:
            res_img, scale = draw_ui(fc['img'], fc['gt'], fc['pred'], fc['conf'], fc['reason'])
            if fc['box'] and len(fc['box']) == 4:
                x1, y1, x2, y2 = fc['box']
                cv2.rectangle(res_img, (int(x1*scale), int(y1*scale)+100), (int(x2*scale), int(y2*scale)+100), (0,255,0), 2)
            cv2.imshow("Failure Review", res_img)
            if cv2.waitKey(0) == ord('q'): break

    cv2.destroyAllWindows()

# --- 실행 ---
if __name__ == "__main__":
    # 필수 폴더 확인
    print(f"📁 BASE_DIR: {BASE_DIR}")
    print(f"📁 YOLO_MODEL: {YOLO_MODEL}")
    print(f"📁 OCR_MODEL: {OCR_MODEL}")
    print(f"📁 RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"📁 FAIL_DIR: {FAIL_DIR}\n")
    
    # 경로 존재 확인
    if not os.path.exists(YOLO_MODEL):
        print(f"❌ YOLO 모델을 찾을 수 없습니다: {YOLO_MODEL}")
    elif not os.path.exists(OCR_MODEL):
        print(f"❌ OCR 모델을 찾을 수 없습니다: {OCR_MODEL}")
    elif not os.path.exists(RAW_DATA_DIR):
        print(f"❌ 원본 데이터 폴더를 찾을 수 없습니다: {RAW_DATA_DIR}")
    else:
        run_integrated_audit_report(YOLO_MODEL, OCR_MODEL, RAW_DATA_DIR, FAIL_DIR, num_test=1000)