import cv2
import numpy as np

# 웹캠 열기
cap = cv2.VideoCapture(1)

while True:
    # 웹캠에서 프레임 읽어오기
    ret, frame = cap.read()

    if not ret:
        break

    # 흑백 이미지로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 이미지를 이진화합니다. (흰색 바탕, 검은색 물체)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 경계선 검출
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 음각과 양각을 구분하기 위한 임계값 설정 (예: 100)
    angle_threshold = 100

    # 각 윤곽선을 순회하면서 음각과 양각을 구분합니다.
    for contour in contours:
        if len(contour) >= 3:
            # 윤곽선에 대한 경계상자를 구합니다.
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # np.int0를 np.intp로 수정

            # 각 점을 순회하면서 각도를 계산합니다.
            angles = []
            for i in range(4):
                pt1 = box[i]
                pt2 = box[(i + 1) % 4]
                pt3 = box[(i + 2) % 4]

                vector1 = pt1 - pt2
                vector2 = pt3 - pt2

                # 두 벡터 사이의 각도 계산
                norm_vector1 = np.linalg.norm(vector1)
                norm_vector2 = np.linalg.norm(vector2)

                if norm_vector1 > 0 and norm_vector2 > 0:  # 0으로 나누는 경우 예외 처리
                    angle = np.degrees(np.arccos(np.dot(vector1, vector2) / (norm_vector1 * norm_vector2)))
                    angles.append(angle)

            # 네 각도 중에 최소 각도를 찾습니다.
            if len(angles) > 0:
                min_angle = min(angles)

                # 음각과 양각을 구분합니다.
                if min_angle < angle_threshold:
                    # 음각
                    cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                else:
                    # 양각
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    # 결과를 화면에 표시
    cv2.imshow('Webcam Angle Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠과 창 해제
cap.release()
cv2.destroyAllWindows()
