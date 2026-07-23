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

# "NEW RECORD"가 표시되는 영역을, 게임 창의 좌상단 기준 상대 좌표(오프셋)로 지정
# 창을 화면 어디로 옮기든 이 오프셋만큼 자동으로 따라갑니다.
CAPTURE_OFFSET_LEFT = 300
CAPTURE_OFFSET_TOP = 150
CAPTURE_WIDTH = 400
CAPTURE_HEIGHT = 200

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
    offset_left: int,
    offset_top: int,
    width: int,
    height: int,
) -> dict:
    """지정한 제목의 창을 찾아, 창 좌상단 기준 상대 좌표로 캡처 영역을 계산합니다."""
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise RuntimeError(
            f"'{window_title}' 창을 찾을 수 없습니다. "
            "게임이 실행 중인지, GAME_WINDOW_TITLE이 정확한지 확인하세요."
        )

    win = windows[0]
    return {
        "left": win.left + offset_left,
        "top": win.top + offset_top,
        "width": width,
        "height": height,
    }


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
    logger.info("대상 창: '%s' / 오프셋: (%d, %d) / 영역 크기: %dx%d",
                GAME_WINDOW_TITLE, CAPTURE_OFFSET_LEFT, CAPTURE_OFFSET_TOP,
                CAPTURE_WIDTH, CAPTURE_HEIGHT)
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
                            CAPTURE_OFFSET_LEFT,
                            CAPTURE_OFFSET_TOP,
                            CAPTURE_WIDTH,
                            CAPTURE_HEIGHT,
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
