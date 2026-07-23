"""
DJMAX Respect V - 뱃지 감지 → Webhook 트리거 스크립트

기능:
  1. NEW RECORD / MAX COMBO / PERFECT 세 종류의 뱃지를 각각 독립적으로 감시
  2. 각 뱃지는 창 크기 대비 "중심점 비율"로 위치를 지정하므로 창 크기가 달라져도 위치를 따라감
  3. cv2.matchTemplate()으로 뱃지별 템플릿 이미지와 비교
  4. 뱃지별로 독립적인 쿨다운을 적용해 중복 감지를 방지
  5. NEW RECORD가 감지된 순간, MAX COMBO/PERFECT 뱃지가 함께 떠 있는지도 확인해서
     그 결과를 new_record_detected 이벤트의 payload에 플래그로 실어 보냄
  6. 뱃지 감지 시점에 곡 제목 / 버튼 수 / 패턴을 OCR(Tesseract)로 읽어서
     webhook payload에 함께 실어 보냄 (Make/n8n에서 V-Archive API 조회 시 사용)
  7. Ctrl+C로 종료

필요 라이브러리:
  pip install mss opencv-python requests numpy pygetwindow pytesseract

필요 프로그램:
  Tesseract-OCR 본체 설치 필요 (pytesseract는 파이썬 바인딩일 뿐, 엔진은 별도 설치)
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki 에서 설치, 설치 시 "Korean" 언어팩 체크
  - macOS: brew install tesseract tesseract-lang
  - Linux: sudo apt install tesseract-ocr tesseract-ocr-kor
"""

import time
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
import mss
import requests
import pygetwindow as gw
import pytesseract
import re

# ==========================================================
# 설정 (여기만 수정하면 됩니다)
# ==========================================================

# 창모드 게임 창 제목 (작업 관리자/알트탭에서 보이는 정확한 창 제목으로 맞춰야 합니다)
GAME_WINDOW_TITLE = "DJMAX RESPECT V"

# 매 프레임마다 창 위치를 다시 조회할지 여부.
TRACK_WINDOW_EVERY_FRAME = True
REFRESH_WINDOW_EVERY_SEC = 5

# 캡처 주기 (초)
CHECK_INTERVAL_SEC = 1.5

# Webhook 요청 타임아웃 (초)
WEBHOOK_TIMEOUT_SEC = 5

# 감지 시 신호를 전송할 Webhook URL들 (Make / n8n 등, 필요한 만큼 추가/삭제 가능)
WEBHOOK_URLS = [
    "[https://hook.eu1.make.com/2l8evmul79dsa8bkt1hrwb3aty65vl7r]",
    "[n8n Webhook URL을 입력하세요]",
]

# ----------------------------------------------------------
# OCR로 읽어올 정보 (곡 제목 / 버튼수 / 패턴)
# 뱃지가 감지된 "그 순간"에만 OCR을 수행해서 webhook payload에 실어 보냅니다.
# 아래 기본값은 850x478 기준 캡처에서 측정한 값입니다 (필요시 재측정해서 조정하세요)
# ----------------------------------------------------------

OCR_REGIONS = {
    "song_title": {
        "center_x_frac": 0.5235,
        "center_y_frac": 0.0272,
        "width_frac": 0.2706,
        "height_frac": 0.0293,
        "margin_frac": 0.01,
    },
    "button_count": {
        "center_x_frac": 0.0706,
        "center_y_frac": 0.0460,
        "width_frac": 0.1176,
        "height_frac": 0.0418,
        "margin_frac": 0.01,
    },
    "pattern": {
        "center_x_frac": 0.3824,
        "center_y_frac": 0.0879,
        "width_frac": 0.0588,
        "height_frac": 0.0251,
        "margin_frac": 0.01,
    },
    "score_number": {
        # 화면의 "SCORE" 큰 숫자(예: 986267)는 정확도 x 10000 값이라서
        # 이 숫자만 읽으면 정확도(score %)를 따로 인식할 필요가 없습니다.
        "center_x_frac": 0.5029,
        "center_y_frac": 0.6967,
        "width_frac": 0.2176,
        "height_frac": 0.0628,
        "margin_frac": 0.01,
    },
}

# Tesseract 실행 파일 경로. 환경변수 PATH에 없으면 직접 지정하세요.
# 예) Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# OCR 언어 (곡 제목에 한글이 섞일 수 있어 eng+kor 권장)
# 설치 필요: Windows Tesseract 설치 시 "Korean" 언어팩 옵션 체크,
#            Linux는 `sudo apt install tesseract-ocr-kor`
OCR_LANG = "eng+kor"

# 패턴 약어 -> 정식명 매핑 (스크립트에서는 사용하지 않고, Make/n8n에서
# 기록등록 API 호출 시 이 매핑대로 변환해서 넣으라는 참고용으로 남겨둡니다)
PATTERN_ALIASES = {
    "NM": "NORMAL",
    "HD": "HARD",
    "MX": "MAXIMUM",
    "SC": "SC",
}

# ----------------------------------------------------------
# 뱃지별 설정
# 각 뱃지는 850x478 기준 캡처에서 측정한 "창 크기 대비 비율" 좌표를 사용합니다.
# ----------------------------------------------------------

BADGES = {
    "new_record": {
        "event_name": "new_record_detected",
        "template_path": "templates/new_record_template.png",
        "center_x_frac": 0.4988,
        "center_y_frac": 0.8096,
        "width_frac": 0.2094,
        "height_frac": 0.0418,
        "margin_frac": 0.01,
        "threshold": 0.8,
        "cooldown_sec": 12,
        # True면: 이 뱃지가 감지될 때 다른 뱃지들의 "현재 상태"를 함께 확인해서
        # payload에 플래그로 포함시킵니다 (신기록 + 맥스콤보/퍼펙트 여부 동시 기록용)
        "check_companions": True,
    },
    "max_combo": {
        "event_name": "max_combo_detected",
        "template_path": "templates/max_combo_template.png",
        "center_x_frac": 0.5576,
        "center_y_frac": 0.4770,
        "width_frac": 0.0612,
        "height_frac": 0.0837,
        "margin_frac": 0.01,
        "threshold": 0.8,
        "cooldown_sec": 12,
        "check_companions": False,
    },
    "perfect": {
        "event_name": "perfect_detected",
        "template_path": "templates/perfect_template.png",
        # 실제 PERFECT PLAY 스크린샷에서 측정한 값입니다.
        "center_x_frac": 0.5312,
        "center_y_frac": 0.4948,
        "width_frac": 0.1024,
        "height_frac": 0.1318,
        "margin_frac": 0.01,
        "threshold": 0.8,
        "cooldown_sec": 12,
        "check_companions": False,
    },
}

# ==========================================================
# 로깅 설정
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("djmax_watcher")


def load_template(path: str) -> np.ndarray:
    """템플릿 이미지를 그레이스케일로 로드합니다."""
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"템플릿 이미지를 찾을 수 없습니다: {path}")
    return template


def get_window_capture_region(
    window_title: str,
    center_x_frac: float,
    center_y_frac: float,
    width_frac: float,
    height_frac: float,
    margin_frac: float = 0.0,
) -> dict:
    """지정한 제목의 창을 찾아, 뱃지 중심점의 상대 위치(비율)로 캡처 영역(픽셀)을 계산합니다.

    창 크기가 달라져도 (center_x_frac, center_y_frac)로 표현된 중심점의
    상대 위치는 그대로 유지된다는 가정하에 동작합니다.
    """
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise RuntimeError(
            f"'{window_title}' 창을 찾을 수 없습니다. "
            "게임이 실행 중인지, GAME_WINDOW_TITLE이 정확한지 확인하세요."
        )

    win = windows[0]
    win_w, win_h = win.width, win.height

    center_x = win.left + center_x_frac * win_w
    center_y = win.top + center_y_frac * win_h
    base_width = width_frac * win_w
    base_height = height_frac * win_h

    margin_x = margin_frac * win_w
    margin_y = margin_frac * win_h
    half_width = base_width / 2 + margin_x
    half_height = base_height / 2 + margin_y

    left = int(round(center_x - half_width))
    top = int(round(center_y - half_height))
    width = int(round(half_width * 2))
    height = int(round(half_height * 2))

    return {"left": left, "top": top, "width": width, "height": height}


def capture_region(sct: mss.mss, region: dict) -> np.ndarray:
    """지정된 화면 영역을 캡처하여 그레이스케일 numpy 배열로 반환합니다."""
    monitor = {
        "left": region["left"],
        "top": region["top"],
        "width": region["width"],
        "height": region["height"],
    }
    shot = sct.grab(monitor)
    frame = np.array(shot)  # BGRA
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return gray


def match_score(frame_gray: np.ndarray, template_gray: np.ndarray) -> float:
    """템플릿 매칭을 수행하고 최대 유사도 점수를 반환합니다."""
    if (
        frame_gray.shape[0] < template_gray.shape[0]
        or frame_gray.shape[1] < template_gray.shape[1]
    ):
        return 0.0
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


def ocr_region_text(sct: mss.mss, region: dict, lang: str = OCR_LANG, psm: int = 7) -> str:
    """지정된 화면 영역을 캡처해 OCR로 텍스트를 추출합니다.

    작은 UI 텍스트는 그대로 넣으면 인식률이 낮아서, 그레이스케일 변환 +
    3배 업스케일 + 이진화(threshold) 전처리를 거친 뒤 Tesseract에 넘깁니다.
    """
    monitor = {
        "left": region["left"],
        "top": region["top"],
        "width": region["width"],
        "height": region["height"],
    }
    shot = sct.grab(monitor)
    frame = np.array(shot)  # BGRA

    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # 밝은 텍스트 / 어두운 배경이 많은 UI라서 Otsu 이진화 사용
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = f"--psm {psm}"
    try:
        text = pytesseract.image_to_string(binary, lang=lang, config=config)
    except pytesseract.TesseractError as e:
        logger.warning("OCR 실패: %s", e)
        return ""

    return text.strip()


def parse_button_count(raw_text: str) -> int | None:
    """OCR로 읽은 텍스트에서 버튼 수(4, 5, 6, 8)를 추출합니다."""
    match = re.search(r"[4568]", raw_text)
    if not match:
        return None
    return int(match.group())


def parse_pattern(raw_text: str) -> str | None:
    """OCR로 읽은 텍스트에서 패턴 약어(NM/HD/MX/SC)를 추출합니다.

    V-Archive "유저 기록 조회 API"의 응답도 pattern을 약어(NM/HD/MX/SC)로 반환하므로,
    여기서는 정식명으로 변환하지 않고 약어 그대로 둡니다.
    정식명(NORMAL/HARD/MAXIMUM/SC) 변환은 "기록등록 API" 호출 직전, 즉
    워크플로우(Make/n8n) 쪽에서 처리하세요 (PATTERN_ALIASES는 그 매핑 참고용으로 남겨둡니다).
    """
    cleaned = raw_text.upper().strip()
    for abbr in PATTERN_ALIASES:
        if abbr in cleaned:
            return abbr
    return None


def parse_score_to_accuracy(raw_text: str) -> float | None:
    """OCR로 읽은 SCORE 숫자(예: '986267')를 정확도(%)로 변환합니다.

    DJMAX Respect V의 SCORE는 정확도(%) x 10000 값이므로,
    (예: 986267 -> 98.6267%, 1000000 -> 100.00%) 10000으로 나누면 됩니다.
    """
    digits = re.sub(r"[^0-9]", "", raw_text)
    if not digits:
        return None
    try:
        score_value = int(digits)
    except ValueError:
        return None
    return round(score_value / 10000, 4)


def extract_song_info(sct: mss.mss, window_title: str) -> dict:
    """뱃지 감지 시점에 곡 제목 / 버튼 수 / 패턴을 OCR로 읽어 반환합니다.

    창을 찾지 못하거나 OCR이 실패해도 예외를 던지지 않고,
    읽지 못한 필드는 None으로 채워서 반환합니다 (webhook 전송은 계속 진행).
    """
    info = {"song_title": None, "button": None, "pattern": None, "score": None}

    for field, cfg in OCR_REGIONS.items():
        try:
            region = get_window_capture_region(
                window_title,
                cfg["center_x_frac"],
                cfg["center_y_frac"],
                cfg["width_frac"],
                cfg["height_frac"],
                cfg.get("margin_frac", 0.0),
            )
        except RuntimeError as e:
            logger.warning("[OCR:%s] %s", field, e)
            continue

        # SCORE 숫자는 숫자만 인식하면 되므로 숫자 전용 psm/whitelist 사용
        psm = 7
        raw_text = ocr_region_text(sct, region, psm=psm)
        logger.debug("[OCR:%s] 원문='%s'", field, raw_text)

        if field == "song_title":
            info["song_title"] = raw_text or None
        elif field == "button_count":
            info["button"] = parse_button_count(raw_text)
        elif field == "pattern":
            info["pattern"] = parse_pattern(raw_text)
        elif field == "score_number":
            info["score"] = parse_score_to_accuracy(raw_text)

    return info


def send_webhook(event_name: str, extra_fields: dict | None = None) -> None:
    """모든 Webhook URL에 감지 신호를 전송합니다."""
    payload = {
        "event": event_name,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_fields:
        payload.update(extra_fields)

    for url in WEBHOOK_URLS:
        if not url or url.startswith("["):
            logger.warning("Webhook URL이 설정되지 않았습니다: %s", url)
            continue
        try:
            resp = requests.post(url, json=payload, timeout=WEBHOOK_TIMEOUT_SEC)
            resp.raise_for_status()
            logger.info("Webhook 전송 성공 [%s]: %s (status=%s)", event_name, url, resp.status_code)
        except requests.exceptions.RequestException as e:
            logger.error("Webhook 전송 실패 [%s]: %s (%s)", event_name, url, e)


class BadgeWatcher:
    """뱃지 하나(NEW RECORD / MAX COMBO / PERFECT 등)에 대한 감시 상태를 관리합니다."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.template_gray = load_template(config["template_path"])
        self.last_detected_at = 0.0
        self.last_window_refresh_at = 0.0
        self.current_region: dict | None = None

    def refresh_region(self, now: float, force: bool) -> bool:
        """필요 시 창 위치 기준으로 캡처 영역을 갱신합니다. 성공하면 True."""
        need_refresh = (
            self.current_region is None
            or TRACK_WINDOW_EVERY_FRAME
            or force
            or (now - self.last_window_refresh_at >= REFRESH_WINDOW_EVERY_SEC)
        )
        if not need_refresh:
            return True

        try:
            self.current_region = get_window_capture_region(
                GAME_WINDOW_TITLE,
                self.config["center_x_frac"],
                self.config["center_y_frac"],
                self.config["width_frac"],
                self.config["height_frac"],
                self.config.get("margin_frac", 0.0),
            )
            self.last_window_refresh_at = now
            return True
        except RuntimeError as e:
            logger.warning("[%s] %s", self.name, e)
            return False

    def get_current_score(self, sct: mss.mss) -> float:
        """현재 프레임에서 이 뱃지의 매칭 점수를 반환합니다. (쿨다운과 무관하게 순수 측정)"""
        if self.current_region is None:
            return 0.0
        frame_gray = capture_region(sct, self.current_region)
        return match_score(frame_gray, self.template_gray)

    def in_cooldown(self, now: float) -> bool:
        return now - self.last_detected_at < self.config["cooldown_sec"]

    def mark_detected(self, now: float) -> None:
        self.last_detected_at = now


def main() -> None:
    logger.info("DJMAX 뱃지 감시 시작 (Ctrl+C로 종료) — 대상 창: '%s'", GAME_WINDOW_TITLE)

    watchers: dict[str, BadgeWatcher] = {}
    for name, config in BADGES.items():
        try:
            watchers[name] = BadgeWatcher(name, config)
            logger.info("[%s] 템플릿 로드 완료: %s", name, config["template_path"])
        except FileNotFoundError as e:
            logger.error(str(e))

    if not watchers:
        logger.error("로드된 뱃지 템플릿이 없습니다. BADGES 설정과 템플릿 파일 경로를 확인하세요.")
        return

    with mss.mss() as sct:
        while True:
            try:
                now = time.time()

                # 1) 모든 뱃지의 캡처 영역을 갱신하고 현재 매칭 점수를 구함
                current_scores: dict[str, float] = {}
                for name, watcher in watchers.items():
                    if not watcher.refresh_region(now, force=False):
                        continue
                    current_scores[name] = watcher.get_current_score(sct)
                    logger.debug("[%s] 유사도=%.3f", name, current_scores[name])

                # 2) 뱃지별로 임계값 초과 + 쿨다운 아님 이면 이벤트 발생
                for name, watcher in watchers.items():
                    if name not in current_scores:
                        continue
                    score = current_scores[name]
                    threshold = watcher.config["threshold"]

                    if score < threshold or watcher.in_cooldown(now):
                        continue

                    logger.info("[%s] 감지! (유사도=%.3f)", name, score)

                    # OCR로 곡 제목 / 버튼 / 패턴 정보를 읽어 payload에 포함
                    extra_fields = extract_song_info(sct, GAME_WINDOW_TITLE)

                    if watcher.config.get("check_companions"):
                        # 다른 뱃지들이 지금 이 순간 함께 떠 있는지 플래그로 포함
                        for other_name, other_score in current_scores.items():
                            if other_name == name:
                                continue
                            other_threshold = watchers[other_name].config["threshold"]
                            extra_fields[other_name] = other_score >= other_threshold

                    send_webhook(watcher.config["event_name"], extra_fields or None)
                    watcher.mark_detected(now)

            except mss.exception.ScreenShotError as e:
                logger.error("화면 캡처 실패: %s", e)
            except Exception as e:  # 예기치 못한 에러도 스크립트가 죽지 않도록 처리
                logger.error("예기치 못한 에러 발생: %s", e)

            time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 종료되었습니다.")