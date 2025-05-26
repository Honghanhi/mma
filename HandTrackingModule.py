import cv2
import mediapipe as mp
import math


class HandDetector:
    """
    Tìm kiếm tay sử dụng thư viện mediapipe. Xuất ra các điểm mốc
    theo định dạng pixel. Cung cấp thêm các chức năng như tìm số
    ngón tay mở hay khoảng cách giữa hai ngón tay. Cũng cung cấp thông tin
    về hộp bao quanh của tay được phát hiện.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode  # chế độ tĩnh (static) hay không
        self.maxHands = maxHands  # tối đa số tay có thể phát hiện
        self.detectionCon = detectionCon  # độ tin cậy của phát hiện tay
        self.minTrackCon = minTrackCon  # độ tin cậy của theo dõi tay

        self.mpHands = mp.solutions.hands  # sử dụng module Hands của Mediapipe
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils  # Dùng để vẽ các đường nối
        self.tipIds = [4, 8, 12, 16, 20]  # Các ID ngón tay (ngón cái, trỏ, giữa, áp út, út)
        self.fingers = []  # Danh sách các ngón tay
        self.lmList = []  # Danh sách các điểm mốc của tay

    def findHands(self, img, draw=True, flipType=True):
        """
        Tìm tay trong hình ảnh và vẽ các điểm mốc của tay.
        Trả về danh sách các tay được phát hiện và hình ảnh.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi màu sang RGB
        self.results = self.hands.process(imgRGB)  # Xử lý hình ảnh để tìm tay
        allHands = []
        h, w, c = img.shape  # Lấy kích thước của ảnh
        if self.results.multi_hand_landmarks:  # Nếu có tay được phát hiện
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):  # Duyệt qua các điểm mốc của tay
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])  # Thêm tọa độ điểm mốc vào danh sách
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH  # Tính toán hộp bao quanh tay
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)  # Tính toán tâm của hộp bao quanh

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                # Đảo ngược loại tay nếu flipType = True
                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## vẽ các điểm mốc lên ảnh
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # Vẽ các kết nối giữa các điểm mốc
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)  # Vẽ hộp bao quanh tay
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)  # Thêm văn bản "Left" hoặc "Right"
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Tìm số lượng ngón tay đang mở và trả về danh sách.
        Cân nhắc tay trái và tay phải riêng biệt.
        """

        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Ngón cái
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 ngón tay còn lại
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Tìm khoảng cách giữa hai điểm mốc dựa trên chỉ số của chúng.
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Tính điểm giữa của hai điểm
        length = math.hypot(x2 - x1, y2 - y1)  # Tính khoảng cách giữa hai điểm
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # Vẽ vòng tròn tại điểm đầu
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)  # Vẽ vòng tròn tại điểm cuối
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Vẽ đường thẳng giữa hai điểm
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # Vẽ vòng tròn tại điểm giữa
            return length, info, img
        else:
            return length, info


def main():
    cap = cv2.VideoCapture(0)  # Mở camera
    detector = HandDetector(detectionCon=0.8, maxHands=2)  # Khởi tạo đối tượng HandDetector
    while True:
        # Lấy khung hình từ camera
        success, img = cap.read()
        # Tìm tay và các điểm mốc của tay
        hands, img = detector.findHands(img)  # với việc vẽ các điểm mốc lên ảnh
        # hands = detector.findHands(img, draw=False)  # nếu không vẽ

        if hands:
            # Tay 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # Danh sách 21 điểm mốc của tay
            bbox1 = hand1["bbox"]  # Thông tin hộp bao quanh tay
            centerPoint1 = hand1['center']  # Tâm của tay (cx, cy)
            handType1 = hand1["type"]  # Loại tay (Trái hoặc Phải)

            fingers1 = detector.fingersUp(hand1)  # Kiểm tra số ngón tay mở

            if len(hands) == 2:
                # Tay 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # Danh sách 21 điểm mốc của tay
                bbox2 = hand2["bbox"]  # Thông tin hộp bao quanh tay
                centerPoint2 = hand2['center']  # Tâm của tay (cx, cy)
                handType2 = hand2["type"]  # Loại tay (Trái hoặc Phải)

                fingers2 = detector.fingersUp(hand2)  # Kiểm tra số ngón tay mở

                # Tìm khoảng cách giữa hai điểm mốc. Có thể là cùng tay hoặc tay khác
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # với việc vẽ khoảng cách
                # length, info = detector.findDistance(lmList1[8], lmList2[8])  # nếu không vẽ
        # Hiển thị hình ảnh
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # Chờ 1ms để hiển thị tiếp

if __name__ == "__main__":
    main()
