"""
DJMAX Respect V - NEW RECORD 감지 → Webhook 트리거 스크립트

기능:
  1. 지정한 화면 영역을 주기적으로 캡처
  2. "NEW RECORD" 템플릿 이미지와 cv2.matchTemplate()으로 유사도 비교
  3. 임계값 이상이면 Make / n8n Webhook으로 감지 신호(JSON) 전송
  4. 중복 감지 방지를 위한 쿨다운 적용
  5. Ctrl+C로 종료

필요 라이브러리:
  pip install mss opencv-python requests numpy
"""

import time
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
import mss
import requests
import pygetwindow as gw

# ==========================================================
# 설정 (여기만 수정하면 됩니다)
# ==========================================================

# 창모드 게임 창 제목 (작업 관리자/알트탭에서 보이는 정확한 창 제목으로 맞춰야 합니다)
GAME_WINDOW_TITLE = "DJMAX RESPECT V"

# "NEW RECORD" 뱃지의 "중심점" 위치를, 게임 창 크기에 대한 비율로 지정
# (좌상단 기준보다 중심점 기준이 창 크기 변화에 더 안정적입니다)
# 아래 기본값은 850x478 기준 캡처에서 측정한 값입니다 (필요시 재측정해서 조정하세요)
CAPTURE_CENTER_X_FRAC = 0.4988   # 창 너비 대비 뱃지 중심의 x 비율
CAPTURE_CENTER_Y_FRAC = 0.8096   # 창 높이 대비 뱃지 중심의 y 비율
CAPTURE_WIDTH_FRAC = 0.2094      # 창 너비 대비 캡처 폭 비율
CAPTURE_HEIGHT_FRAC = 0.0418     # 창 높이 대비 캡처 높이 비율

# 뱃지 애니메이션(페이드인 등)을 여유 있게 잡기 위한 여백 비율 (선택, 0이면 여백 없음)
CAPTURE_MARGIN_FRAC = 0.01

# 매 프레임마다 창 위치를 다시 조회할지 여부.
# True면 창을 옮겨도 항상 정확하지만 약간의 오버헤드가 있습니다.
# False면 REFRESH_WINDOW_EVERY_SEC 주기로만 갱신합니다.
TRACK_WINDOW_EVERY_FRAME = True
REFRESH_WINDOW_EVERY_SEC = 5

# 비교할 템플릿 이미지 경로 ("NEW RECORD" 뱃지/문구를 미리 캡처해서 저장해두세요)
TEMPLATE_IMAGE_PATH = "new_record_template.png"

# 템플릿 매칭 유사도 임계값 (0.0 ~ 1.0). 값이 높을수록 엄격하게 판정합니다.
MATCH_THRESHOLD = 0.8

# 캡처 주기 (초)
CHECK_INTERVAL_SEC = 1.5

# 중복 감지를 막기 위한 쿨다운 (초). 감지 후 이 시간 동안은 재감지하지 않습니다.
COOLDOWN_SEC = 12

# 감지 시 신호를 전송할 Webhook URL들 (Make / n8n 등, 필요한 만큼 추가/삭제 가능)
WEBHOOK_URLS = [
    "[Make Webhook URL을 입력하세요]",
    "[n8n Webhook URL을 입력하세요]",
]

# Webhook 요청 타임아웃 (초)
WEBHOOK_TIMEOUT_SEC = 5

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
        raise FileNotFoundError(
            f"템플릿 이미지를 찾을 수 없습니다: {path}. "
            "TEMPLATE_IMAGE_PATH 설정을 확인하세요."
        )
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

    # 여백 적용 (창 크기에 비례해서 상하좌우로 살짝 확장)
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
    # 캡처 영역이 템플릿보다 작으면 매칭할 수 없으므로 방어적으로 처리
    if (
        frame_gray.shape[0] < template_gray.shape[0]
        or frame_gray.shape[1] < template_gray.shape[1]
    ):
        logger.warning("캡처 영역이 템플릿 이미지보다 작습니다. CAPTURE_REGION을 확인하세요.")
        return 0.0

    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


def send_webhooks(urls: list[str]) -> None:
    """모든 Webhook URL에 감지 신호를 전송합니다."""
    payload = {
        "event": "new_record_detected",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        # 곡명 자동 인식이 어려우면 생략하고 워크플로우 쪽에서 API로 보완
        "song_hint": None,
    }

    for url in urls:
        if not url or url.startswith("["):
            logger.warning("Webhook URL이 설정되지 않았습니다: %s", url)
            continue
        try:
            resp = requests.post(url, json=payload, timeout=WEBHOOK_TIMEOUT_SEC)
            resp.raise_for_status()
            logger.info("Webhook 전송 성공: %s (status=%s)", url, resp.status_code)
        except requests.exceptions.RequestException as e:
            logger.error("Webhook 전송 실패: %s (%s)", url, e)


def main() -> None:
    logger.info("DJMAX NEW RECORD 감시 시작 (Ctrl+C로 종료)")
    logger.info("대상 창: '%s' / 중심점=(%.4f, %.4f) / w=%.4f h=%.4f (margin=%.4f)",
                GAME_WINDOW_TITLE, CAPTURE_CENTER_X_FRAC, CAPTURE_CENTER_Y_FRAC,
                CAPTURE_WIDTH_FRAC, CAPTURE_HEIGHT_FRAC, CAPTURE_MARGIN_FRAC)
    logger.info("임계값: %.2f / 주기: %.1fs / 쿨다운: %ds",
                MATCH_THRESHOLD, CHECK_INTERVAL_SEC, COOLDOWN_SEC)

    try:
        template_gray = load_template(TEMPLATE_IMAGE_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    last_detected_at = 0.0
    last_window_refresh_at = 0.0
    current_region = None

    with mss.mss() as sct:
        while True:
            try:
                now = time.time()

                # 쿨다운 중이면 캡처/매칭 생략
                if now - last_detected_at < COOLDOWN_SEC:
                    time.sleep(CHECK_INTERVAL_SEC)
                    continue

                # 창 위치 갱신 (매 프레임 또는 일정 주기)
                need_refresh = (
                    current_region is None
                    or TRACK_WINDOW_EVERY_FRAME
                    or (now - last_window_refresh_at >= REFRESH_WINDOW_EVERY_SEC)
                )
                if need_refresh:
                    try:
                        current_region = get_window_capture_region(
                            GAME_WINDOW_TITLE,
                            CAPTURE_CENTER_X_FRAC,
                            CAPTURE_CENTER_Y_FRAC,
                            CAPTURE_WIDTH_FRAC,
                            CAPTURE_HEIGHT_FRAC,
                            CAPTURE_MARGIN_FRAC,
                        )
                        last_window_refresh_at = now
                    except RuntimeError as e:
                        logger.warning(str(e))
                        time.sleep(CHECK_INTERVAL_SEC)
                        continue

                frame_gray = capture_region(sct, current_region)
                score = match_score(frame_gray, template_gray)

                logger.debug("유사도 점수: %.3f (영역=%s)", score, current_region)

                if score >= MATCH_THRESHOLD:
                    logger.info("NEW RECORD 감지! (유사도=%.3f)", score)
                    send_webhooks(WEBHOOK_URLS)
                    last_detected_at = now

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
