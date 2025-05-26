import tensorflow
from tensorflow import keras
import numpy as np
import cv2


class Classifier:
    """
    Lớp Classifier giúp tải mô hình đã huấn luyện và thực hiện phân loại ảnh.
    """

    def __init__(self, modelPath, labelsPath=None):
        """
        Hàm khởi tạo nhận đường dẫn đến mô hình và (tuỳ chọn) nhãn.
        """
        self.model_path = modelPath
        
        np.set_printoptions(suppress=True)
        # Tải mô hình đã huấn luyện
        self.model = tensorflow.keras.models.load_model(self.model_path)

        # Tạo mảng với kích thước đúng để đưa vào mô hình keras
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            # Nếu có tệp nhãn, mở và đọc chúng
            label_file = open(self.labels_path, "r")
            self.list_labels = []
            for line in label_file:
                stripped_line = line.strip()  # Xoá ký tự thừa trong mỗi dòng
                self.list_labels.append(stripped_line)  # Thêm nhãn vào danh sách
            label_file.close()
        else:
            print("Không tìm thấy nhãn")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0,255,0)):
        """
        Dự đoán nhãn của ảnh đầu vào.
        """
        # Thay đổi kích thước ảnh thành 224x224
        imgS = cv2.resize(img, (224, 224))
        # Chuyển ảnh thành mảng numpy
        image_array = np.asarray(imgS)
        # Chuẩn hóa ảnh
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Đưa ảnh vào mảng dữ liệu
        self.data[0] = normalized_image_array

        # Thực hiện dự đoán
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)  # Lấy chỉ số của nhãn có xác suất cao nhất

        if draw and self.labels_path:
            # Vẽ nhãn lên ảnh nếu yêu cầu
            cv2.putText(img, str(self.list_labels[indexVal]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal  # Trả về xác suất và chỉ số nhãn


def main():
    cap = cv2.VideoCapture(0)  # Mở camera
    maskClassifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')  # Tạo đối tượng Classifier với mô hình và nhãn
    while True:
        _, img = cap.read()  # Đọc một khung hình từ camera
        predection = maskClassifier.getPrediction(img)  # Lấy dự đoán cho khung hình
        print(predection)  # In ra dự đoán
        cv2.imshow("Image", img)  # Hiển thị ảnh lên cửa sổ
        cv2.waitKey(1)  # Chờ 1 ms để hiển thị tiếp

if __name__ == "__main__":
    main()
