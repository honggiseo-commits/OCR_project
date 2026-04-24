import torch
import torch.nn as nn
import cv2
import numpy as np
import re
import time
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from collections import deque

# --- 기본 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL = os.path.join(BASE_DIR, "models", "YOLO_best.pt")
OCR_MODEL = os.path.join(BASE_DIR, "models", "Colab_best_0420_v2.pt")

# --- 1. OCR 모델 구조 ---
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
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden, num_layers=2, 
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2).permute(0, 2, 1)
        out, _ = self.rnn(feat)
        return self.fc(out).log_softmax(2)

# --- 2. 규칙 기반 디코딩 엔진 (최적화) ---
class RuleBasedDecoder:
    """규칙 기반 번호판 디코딩"""
    
    def __init__(self, min_confidence=40.0):
        self.min_confidence = min_confidence
    
    def decode(self, logits, idx2char):
        """
        로짓을 번호판 텍스트로 변환
        Args:
            logits: (seq_len, num_classes) log softmax 결과
            idx2char: {index: character} 매핑 딕셔너리
        
        Returns:
            text: 디코딩된 번호판 텍스트
            conf: 신뢰도 점수 (0-100)
        """
        probs = torch.exp(logits)
        max_probs, indices = probs.max(dim=1)
        blank_idx = 0
        
        # CTC 디코딩: 연속된 같은 문자 제거 (blank 포함)
        decoded_objects = []
        prev = None
        
        for i in range(len(indices)):
            idx = indices[i].item()
            if idx != blank_idx and idx != prev:
                char = idx2char.get(idx, "")
                if char:  # 빈 문자 제외
                    decoded_objects.append({
                        'char': char, 
                        'conf': max_probs[i].item(), 
                        'pos': i,
                        'type': self._classify_char(char)
                    })
            prev = idx

        if not decoded_objects:
            return "", 0.0

        # 한글 기준점 찾기
        hangul_candidates = [obj for obj in decoded_objects if obj['type'] == 'hangul']
        if not hangul_candidates:
            # 한글 없음: 숫자만 반환
            return "".join([o['char'] for o in decoded_objects]), 0.0

        # 가장 신뢰도 높은 한글을 기준점으로
        main_hangul = max(hangul_candidates, key=lambda x: x['conf'])
        pivot_pos = main_hangul['pos']
        
        # 앞자리(2-3개 숫자) / 뒷자리(4개 숫자) 분리
        front_digits = [obj for obj in decoded_objects 
                       if obj['pos'] < pivot_pos and obj['type'] == 'digit']
        back_digits = [obj for obj in decoded_objects 
                      if obj['pos'] > pivot_pos and obj['type'] == 'digit']

        # 자릿수 조정 (신뢰도 기준)
        back_digits = self._limit_digits(back_digits, max_count=4)
        front_digits = self._limit_digits(front_digits, max_count=3)

        # 앞자리 처리
        res_front = "".join([d['char'] for d in front_digits])
        if len(res_front) == 3 and res_front[0] == '0':
            res_front = res_front[1:]  # '0'으로 시작하면 제거
        
        res_back = "".join([d['char'] for d in back_digits])
        final_text = f"{res_front}{main_hangul['char']}{res_back}"
        
        # 신뢰도: 한글 + 앞자리 + 뒷자리의 평균
        conf_values = [main_hangul['conf']] + [d['conf'] for d in (front_digits + back_digits)]
        avg_conf = np.mean(conf_values) * 100 if conf_values else 0.0
        
        return final_text, avg_conf

    @staticmethod
    def _classify_char(char):
        """문자 분류"""
        if re.match(r'[가-힣]', char):
            return 'hangul'
        elif re.match(r'[0-9]', char):
            return 'digit'
        else:
            return 'other'
    
    @staticmethod
    def _limit_digits(digit_list, max_count):
        """자릿수 제한 (신뢰도 기준으로 상위 N개 유지)"""
        if len(digit_list) <= max_count:
            return digit_list
        # 신뢰도 상위 max_count개를 위치순으로 정렬
        return sorted(sorted(digit_list, key=lambda x: x['conf'], reverse=True)[:max_count], 
                     key=lambda x: x['pos'])

# --- 3. 실시간 UI 대시보드 ---
class LiveDashboard:
    """실시간 인식 결과를 표시하는 대시보드"""
    
    def __init__(self, target_h=500, margin_h=120):
        self.target_h = target_h
        self.margin_h = margin_h
        self._init_font()
        
    def _init_font(self):
        """폰트 초기화"""
        font_paths = [
            "C:/Windows/Fonts/malgunbd.ttf",  # Windows
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",  # Linux
            "/Library/Fonts/Arial.ttf",  # macOS
        ]
        self.font_large = None
        self.font_small = None
        
        for path in font_paths:
            try:
                self.font_large = ImageFont.truetype(path, 40)
                self.font_small = ImageFont.truetype(path, 24)
                return
            except:
                continue
        
        # 폰트 로드 실패 시 기본 폰트 사용
        self.font_large = ImageFont.load_default()
        self.font_small = ImageFont.load_default()

    def render(self, frame, plate_text, confidence, detection_box=None, fps=0):
        """
        프레임에 대시보드 렌더링
        
        Args:
            frame: 입력 프레임
            plate_text: 인식된 번호판 텍스트
            confidence: 신뢰도 (0-100)
            detection_box: (x1, y1, x2, y2) YOLO 탐지 박스
            fps: 초당 프레임 수
        
        Returns:
            display_frame: 렌더링된 프레임
            scale: 스케일 팩터
        """
        h_orig, w_orig = frame.shape[:2]
        scale = self.target_h / h_orig
        resized = cv2.resize(frame, (int(w_orig * scale), self.target_h))
        
        # 캔버스 생성 (상단 마진)
        canvas = np.zeros((self.target_h + self.margin_h, resized.shape[1], 3), dtype=np.uint8)
        canvas[self.margin_h:, :] = resized
        
        # PIL로 텍스트 렌더링
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 텍스트 내용
        if plate_text:
            status_text = f"✓ 번호판: {plate_text}"
            status_color = (0, 255, 0)  # 초록색
        else:
            status_text = "⊙ 탐색 중..."
            status_color = (255, 165, 0)  # 주황색
        
        # 상단 정보 표시
        draw.text((20, 15), status_text, font=self.font_large, fill=status_color)
        
        if plate_text:
            conf_text = f"신뢰도: {confidence:.1f}%"
            draw.text((20, 60), conf_text, font=self.font_small, fill=(255, 255, 255))
        
        # FPS 표시 (우상단)
        fps_text = f"FPS: {fps:.1f}"
        draw.text((resized.shape[1] - 150, 15), fps_text, font=self.font_small, fill=(100, 200, 255))
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), scale

# --- 4. 성능 모니터링 ---
class PerformanceMonitor:
    """실시간 성능 모니터링"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.ocr_times = deque(maxlen=window_size)
        self.yolo_times = deque(maxlen=window_size)
        
    def add_frame_time(self, elapsed):
        self.frame_times.append(elapsed)
    
    def add_yolo_time(self, elapsed):
        self.yolo_times.append(elapsed)
    
    def add_ocr_time(self, elapsed):
        self.ocr_times.append(elapsed)
    
    def get_fps(self):
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = np.mean(list(self.frame_times))
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def print_stats(self):
        """성능 통계 출력"""
        if self.frame_times:
            print(f"FPS: {self.get_fps():.1f}")
            if self.yolo_times:
                print(f"  YOLO: {np.mean(list(self.yolo_times))*1000:.1f}ms")
            if self.ocr_times:
                print(f"  OCR: {np.mean(list(self.ocr_times))*1000:.1f}ms")

# --- 5. 메인 실시간 LPR 엔진 ---
class LiveLPREngine:
    """실시간 번호판 인식 엔진"""
    
    def __init__(self, yolo_path, ocr_path, camera_id=0, 
                 yolo_conf_threshold=0.6, ocr_conf_threshold=40.0):
        """
        Args:
            yolo_path: YOLO 모델 경로
            ocr_path: OCR 모델 경로
            camera_id: 웹캠 ID (0=기본, 1=외부 등)
            yolo_conf_threshold: YOLO 신뢰도 임계값
            ocr_conf_threshold: OCR 신뢰도 임계값
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_conf_threshold = yolo_conf_threshold
        self.ocr_conf_threshold = ocr_conf_threshold
        
        # 모델 로드
        print(f"📦 YOLO 모델 로드 중... ({yolo_path})")
        self.detector = YOLO(yolo_path)
        
        print(f"📦 OCR 모델 로드 중... ({ocr_path})")
        ckpt = torch.load(ocr_path, map_location=self.device)
        self.idx2char = {i: c for c, i in ckpt["char2idx"].items()}
        self.ocr_model = CRNN(len(ckpt["char2idx"]) + 1).to(self.device)
        self.ocr_model.load_state_dict(ckpt["model_state"])
        self.ocr_model.eval()
        
        # 웹캠 초기화
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 카메라 {camera_id}를 열 수 없습니다")
        
        self._setup_camera()
        
        # 보조 도구
        self.decoder = RuleBasedDecoder(min_confidence=ocr_conf_threshold)
        self.dashboard = LiveDashboard()
        self.monitor = PerformanceMonitor()
        
        # 결과 메모리 (프레임 간 유지)
        self.last_plate = {"text": "", "conf": 0.0, "time": 0}
        self.result_retention_time = 2.0  # 초
        
    def _setup_camera(self):
        """카메라 설정"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화 (지연 감소)
        
    def run(self):
        """실시간 루프 실행"""
        print(f"🚀 실시간 LPR 시스템 시작 (GPU: {self.device.type.upper()})")
        print(f"   - YOLO 신뢰도 임계값: {self.yolo_conf_threshold}")
        print(f"   - OCR 신뢰도 임계값: {self.ocr_conf_threshold}%")
        print(f"   - 종료: 'q' 키, ESC 키, 또는 창 닫기\n")
        
        frame_count = 0
        should_exit = False  # ← 종료 플래그 추가
        
        try:
            while self.cap.isOpened() and not should_exit:  # ← 플래그 체크
                frame_start = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ 프레임 읽기 실패")
                    should_exit = True  # ← 안전한 종료
                    break
                
                frame_count += 1
                current_plate = None
                
                # 1. YOLO 탐지
                yolo_start = time.time()
                results = self.detector(frame, verbose=False, conf=self.yolo_conf_threshold)
                yolo_time = time.time() - yolo_start
                self.monitor.add_yolo_time(yolo_time)
                
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 2. OCR 인식
                    ocr_start = time.time()
                    h_p, w_p = y2 - y1, x2 - x1
                    
                    # 패딩 적용 (좌우 5%, 상하 15%)
                    crop = frame[
                        max(0, y1 - int(h_p * 0.15)):min(frame.shape[0], y2 + int(h_p * 0.15)),
                        max(0, x1 - int(w_p * 0.05)):min(frame.shape[1], x2 + int(w_p * 0.05))
                    ]
                    
                    if crop.size > 0:
                        # 전처리
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        input_img = cv2.resize(gray_crop, (320, 96))
                        input_tensor = torch.FloatTensor(input_img).unsqueeze(0).unsqueeze(0).to(self.device)
                        input_tensor = (input_tensor / 127.5) - 1.0
                        
                        # 추론
                        with torch.no_grad():
                            logits = self.ocr_model(input_tensor).squeeze(0)
                            text, conf = self.decoder.decode(logits, self.idx2char)
                            
                            if text and conf >= self.ocr_conf_threshold:
                                self.last_plate = {"text": text, "conf": conf, "time": time.time()}
                    
                    ocr_time = time.time() - ocr_start
                    self.monitor.add_ocr_time(ocr_time)
                
                # 3. 결과 유지 로직 (2초 후 사라짐)
                if time.time() - self.last_plate["time"] > self.result_retention_time:
                    display_info = {"text": "", "conf": 0.0}
                else:
                    display_info = self.last_plate
                
                # 4. 화면 렌더링
                display_frame, scale = self.dashboard.render(
                    frame, 
                    display_info["text"], 
                    display_info["conf"],
                    fps=self.monitor.get_fps()
                )
                
                # YOLO 탐지 박스 그리기
                if len(results[0].boxes) > 0:
                    x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                    cv2.rectangle(display_frame, 
                                (int(x1 * scale), int(y1 * scale) + 120),
                                (int(x2 * scale), int(y2 * scale) + 120),
                                (0, 255, 0), 3)
                
                cv2.imshow("Real-time LPR Dashboard", display_frame)
                
                # 종료 조건 (여러 종료 옵션)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, ESC
                    print("🛑 사용자가 'q' 또는 ESC를 눌렀습니다")
                    should_exit = True
                    break
                
                # 창이 닫혔는지 확인
                if not self._window_exists("Real-time LPR Dashboard"):
                    print("🛑 창이 닫혔습니다")
                    should_exit = True
                    break
                
                # 성능 모니터링
                frame_time = time.time() - frame_start
                self.monitor.add_frame_time(frame_time)
                
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] FPS: {self.monitor.get_fps():.1f}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Ctrl+C로 사용자 중단됨")
            should_exit = True
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            should_exit = True
        finally:
            self.cleanup()
    
    def _window_exists(self, window_name):
        """창이 아직 열려있는지 확인"""
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        except:
            return False
    
    def cleanup(self):
        """리소스 완전히 해제"""
        print("\n🛑 시스템 종료 중...")
        
        # 1. 모든 OpenCV 창 닫기
        try:
            cv2.destroyAllWindows()
            print("   ✓ 창 닫기 완료")
        except Exception as e:
            print(f"   ⚠️  창 닫기 오류: {e}")
        
        # 2. 웹캠 해제
        try:
            if self.cap is not None:
                self.cap.release()
                print("   ✓ 카메라 해제 완료")
        except Exception as e:
            print(f"   ⚠️  카메라 해제 오류: {e}")
        
        # 3. CUDA 캐시 비우기 (선택사항)
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                print("   ✓ GPU 메모리 해제")
        except Exception as e:
            print(f"   ⚠️  GPU 메모리 해제 오류: {e}")
        
        # 4. 통계 출력
        try:
            self.monitor.print_stats()
        except Exception as e:
            print(f"   ⚠️  통계 출력 오류: {e}")
        
        print("\n✓ 종료 완료!")

# --- 실행 ---
if __name__ == "__main__":
    # 경로 검증
    if not os.path.exists(YOLO_MODEL):
        print(f"❌ YOLO 모델 없음: {YOLO_MODEL}")
        print(f"   '{os.path.dirname(YOLO_MODEL)}' 폴더에 'YOLO_best.pt' 파일을 복사하세요")
        exit(1)
    
    if not os.path.exists(OCR_MODEL):
        print(f"❌ OCR 모델 없음: {OCR_MODEL}")
        print(f"   '{os.path.dirname(OCR_MODEL)}' 폴더에 'Colab_best_0420_v2.pt' 파일을 복사하세요")
        exit(1)
    
    engine = None
    try:
        # 엔진 초기화 및 실행
        engine = LiveLPREngine(
            yolo_path=YOLO_MODEL,
            ocr_path=OCR_MODEL,
            camera_id=0,  # 웹캠 ID (외부 카메라는 1, 2...)
            yolo_conf_threshold=0.6,
            ocr_conf_threshold=40.0
        )
        engine.run()
    
    except RuntimeError as e:
        print(f"❌ {e}")
        if engine is not None:
            engine.cleanup()
        exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  프로그램 중단됨")
        if engine is not None:
            engine.cleanup()
        exit(0)
    except Exception as e:
        print(f"❌ 예기치 않은 오류: {e}")
        if engine is not None:
            engine.cleanup()
        exit(1)