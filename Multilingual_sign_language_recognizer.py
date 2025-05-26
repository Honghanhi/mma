import cv2
from HandTrackingModule import HandDetector
from ClassificationModule import Classifier
from UTF8ClassificationModule import UTF8Classifier
import numpy as np
import math
import time
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import os

# Thiết lập theme và màu sắc
COLORS = {
    "primary": "#3498db",       # Xanh dương
    "secondary": "#2ecc71",     # Xanh lá
    "accent": "#9b59b6",        # Tím
    "warning": "#e74c3c",       # Đỏ
    "background": "#2c3e50",    # Xanh đen
    "text_light": "#ecf0f1",    # Trắng
    "text_dark": "#34495e",     # Xanh đen nhạt
    "border": "#7f8c8d"         # Xám
}

# Initialize variables
cap = cv2.VideoCapture(0)  # Camera ID == 0
detector1 = HandDetector(maxHands=1)
detector3 = HandDetector(maxHands=1)
detector2 = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Z"]
labels_r = ["A", "Б", "B", "Г", "Д", "Ё", "Ж", "3", "Й", "K", "Л", "M", "H", "O", "П", "P", "C", "T", "y", "Ф", "X"]

# Tự động tìm đường dẫn tương đối hoặc sử dụng đường dẫn tuyệt đối
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    classifier1 = Classifier(os.path.join(base_path, "model_asl", "keras_model.h5"),
                            os.path.join(base_path, "model_asl", "labels.txt"))
    classifier2 = Classifier(os.path.join(base_path, "model_isl", "keras_model.h5"),
                            os.path.join(base_path, "model_isl", "labels.txt"))
    classifier3 = UTF8Classifier(os.path.join(base_path, "model_rsl", "keras_model.h5"),
                                os.path.join(base_path, "model_rsl", "labels.txt"))
except Exception as e:
    print(f"Error loading models with relative paths: {e}")
    # Fallback to absolute paths
    classifier1 = Classifier(r"E:\Multilingual-Sign-Language-Recognizer-master (1)\Multilingual-Sign-Language-Recognizer-master\model_asl\keras_model.h5",
                            r"E:\Multilingual-Sign-Language-Recognizer-master (1)\Multilingual-Sign-Language-Recognizer-master\model_asl\labels.txt")
    classifier2 = Classifier(r"E:\Multilingual-Sign-Language-Recognizer-master (1)\Multilingual-Sign-Language-Recognizer-master\model_isl\keras_model.h5",
                            r"E:\Multilingual-Sign-Language-Recognizer-master (1)\Multilingual-Sign-Language-Recognizer-master\model_isl\labels.txt")
    classifier3 = UTF8Classifier(r"E:\Multilingual-Sign-Language-Recognizer-master (1)\Multilingual-Sign-Language-Recognizer-master\model_rsl\keras_model.h5",
                                r"E:\Multilingual-Sign-Language-Recognizer-master (1)\Multilingual-Sign-Language-Recognizer-master\model_rsl\labels.txt")

use_code1 = True
use_code2 = False
use_code3 = False

# Create Tkinter window with modern styling
window = tk.Tk()
window.title("Dịch ngôn ngữ kí hiệu")
window.geometry("1280x720")
window.configure(bg=COLORS["background"])

# Try to set icon if available
try:
    window.iconbitmap(r"E:\Multilingual-Sign-Language-Recognizer-master (1)\Multilingual-Sign-Language-Recognizer-master\logo.ico")
except:
    try:
        window.iconbitmap("logo.ico")
    except:
        pass

# Create styles for widgets
style = ttk.Style()
style.theme_use('clam')  # Use a modern theme as base

# Configure styles for different elements
style.configure("TFrame", background=COLORS["background"])
style.configure("TButton", 
                background=COLORS["primary"], 
                foreground=COLORS["text_light"],
                font=("Segoe UI", 10, "bold"),
                borderwidth=0,
                focusthickness=3,
                focuscolor=COLORS["accent"])
style.map("TButton",
          background=[("active", COLORS["accent"]), ("disabled", COLORS["border"])],
          foreground=[("active", COLORS["text_light"]), ("disabled", "#aaaaaa")])

style.configure("Accent.TButton", 
                background=COLORS["accent"], 
                foreground=COLORS["text_light"])
style.map("Accent.TButton",
          background=[("active", COLORS["secondary"]), ("disabled", COLORS["border"])])

style.configure("Warning.TButton", 
                background=COLORS["warning"], 
                foreground=COLORS["text_light"])
style.map("Warning.TButton",
          background=[("active", "#c0392b"), ("disabled", COLORS["border"])])

# Create main layout frames
main_container = ttk.Frame(window, style="TFrame")
main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Create a frame for the main content with a border
content_frame = tk.Frame(main_container, bg=COLORS["background"], bd=2, relief=tk.GROOVE, 
                         highlightbackground=COLORS["border"], highlightthickness=1)
content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create left panel for video with a decorative border
video_panel = tk.Frame(content_frame, bg=COLORS["background"], bd=0)
video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create a frame with a gradient-like border effect for the video
video_border = tk.Frame(video_panel, bg=COLORS["primary"], bd=0)
video_border.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

video_frame = tk.Frame(video_border, bg=COLORS["background"], bd=0)
video_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

# Create a label widget to display the OpenCV output with a border effect
label = tk.Label(video_frame, bg=COLORS["background"], bd=0)
label.pack(fill=tk.BOTH, expand=True)

# Create right panel for text display and controls
right_panel = tk.Frame(content_frame, bg=COLORS["background"])
right_panel.place(x=930, y=10, width=350, height=700)


# Create a title for the text display
text_title = tk.Label(right_panel, text="Bảng chữ", font=("Segoe UI", 14, "bold"), 
                     bg=COLORS["background"], fg=COLORS["text_light"])
text_title.pack(pady=(0, 10))

# Create a text widget to display recognized text with modern styling
text_frame = tk.Frame(right_panel, bg=COLORS["primary"], bd=0)
text_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

text_display = tk.Text(text_frame, width=30, height=20, font=("Segoe UI", 12),
                      bg="#1a1a2e", fg="#4ecca3", bd=0, padx=10, pady=10,
                      insertbackground=COLORS["secondary"], selectbackground=COLORS["accent"])
text_display.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

# Add a scrollbar to the text display
scrollbar = ttk.Scrollbar(text_display, orient="vertical", command=text_display.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_display.config(yscrollcommand=scrollbar.set)

# Create a status bar
status_frame = tk.Frame(window, bg=COLORS["primary"], height=30)
status_frame.pack(side=tk.BOTTOM, fill=tk.X)

status_label = tk.Label(status_frame, text="Ready", font=("Segoe UI", 10), 
                       bg=COLORS["primary"], fg=COLORS["text_light"], anchor=tk.W, padx=10)
status_label.pack(side=tk.LEFT, fill=tk.Y)

# Create a control panel with modern buttons
control_panel = tk.Frame(window, bg=COLORS["background"], height=120)
control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

# Define switch code functions with status updates
def switch_to_code1():
    global use_code1, use_code2, use_code3
    use_code1 = True
    use_code2 = False
    use_code3 = False
    code_label.config(text="Ngôn ngữ kí hiệu mỹ", fg=COLORS["text_light"])
    status_label.config(text="Switched to Ngôn ngữ kí hiệu mỹ")
    
    # Update button states visually
    asl_btn.config(bg=COLORS["accent"], fg=COLORS["text_light"])
    isl_btn.config(bg=COLORS["primary"], fg=COLORS["text_light"])
    rsl_btn.config(bg=COLORS["primary"], fg=COLORS["text_light"])

def switch_to_code2():
    global use_code1, use_code2, use_code3
    use_code1 = False
    use_code2 = True
    use_code3 = False
    code_label.config(text="Ngôn ngữ kí hiệu Ấn", fg=COLORS["text_light"])
    status_label.config(text="Switched to Ngôn ngữ kí hiệu Ấn")
    
    # Update button states visually
    asl_btn.config(bg=COLORS["primary"], fg=COLORS["text_light"])
    isl_btn.config(bg=COLORS["accent"], fg=COLORS["text_light"])
    rsl_btn.config(bg=COLORS["primary"], fg=COLORS["text_light"])

def switch_to_code3():
    global use_code1, use_code2, use_code3
    use_code1 = False
    use_code2 = False
    use_code3 = True
    code_label.config(text="Ngôn ngữ kí hiệu Nga", fg=COLORS["text_light"])
    status_label.config(text="Switched to Ngôn ngữ kí hiệu Nga")
    
    # Update button states visually
    asl_btn.config(bg=COLORS["primary"], fg=COLORS["text_light"])
    isl_btn.config(bg=COLORS["primary"], fg=COLORS["text_light"])
    rsl_btn.config(bg=COLORS["accent"], fg=COLORS["text_light"])

# Define show_chart function with improved styling
def show_chart(chart_path, title):
    global current_chart_window

    # Close the current chart window if it exists
    if 'current_chart_window' in globals() and current_chart_window is not None:
        current_chart_window.destroy()

    try:
        chart = Image.open(chart_path)
        chart = chart.resize((500, 500), Image.LANCZOS)  # LANCZOS instead of ANTIALIAS (deprecated)
        chartTk = ImageTk.PhotoImage(chart)

        # Create a new Tkinter window with styling
        chart_window = tk.Toplevel(window)
        chart_window.title(f"{title} Chart")
        chart_window.configure(bg=COLORS["background"])
        
        # Add a title to the chart window
        chart_title = tk.Label(chart_window, text=f"{title} Reference Chart", 
                              font=("Segoe UI", 14, "bold"),
                              bg=COLORS["background"], fg=COLORS["text_light"])
        chart_title.pack(pady=10)
        
        # Create a frame for the chart with a border
        chart_frame = tk.Frame(chart_window, bg=COLORS["primary"], bd=0)
        chart_frame.pack(padx=15, pady=15)
        
        # Create a label widget to display the chart
        chart_label = tk.Label(chart_frame, image=chartTk, bd=0)
        chart_label.image = chartTk  # Keep a reference
        chart_label.pack(padx=3, pady=3)

        # Add a close button
        close_btn = tk.Button(chart_window, text="Close", font=("Segoe UI", 10, "bold"),
                             bg=COLORS["warning"], fg=COLORS["text_light"],
                             command=chart_window.destroy, padx=20, pady=5)
        close_btn.pack(pady=15)

        # Update the current chart window
        current_chart_window = chart_window
        
        # Center the window on screen
        chart_window.update_idletasks()
        width = chart_window.winfo_width()
        height = chart_window.winfo_height()
        x = (chart_window.winfo_screenwidth() // 2) - (width // 2)
        y = (chart_window.winfo_screenheight() // 2) - (height // 2)
        chart_window.geometry(f'{width}x{height}+{x}+{y}')
        
    except Exception as e:
        status_label.config(text=f"Error loading chart: {e}")

# Function to clear the text display with animation
def clear_text():
    global saved_symbols, current_word
    
    # Flash effect before clearing
    original_bg = text_display.cget("bg")
    text_display.config(bg=COLORS["warning"])
    window.after(100, lambda: text_display.config(bg=original_bg))
    
    # Clear after a short delay
    window.after(200, lambda: text_display.delete(1.0, tk.END))
    saved_symbols = []
    current_word = ""
    status_label.config(text="Xóa")

# Create language selection buttons with modern styling
button_frame = tk.Frame(control_panel, bg=COLORS["background"])
button_frame.pack(pady=10)

# First row of buttons - Language selection
lang_frame = tk.Frame(button_frame, bg=COLORS["background"])
lang_frame.pack(pady=5)

# Create language selection buttons with modern styling
asl_btn = tk.Button(lang_frame, text="Ngôn ngữ kí hiệu mỹ", command=switch_to_code1,
                   font=("Segoe UI", 11, "bold"), width=20, height=2,
                   bg=COLORS["accent"], fg=COLORS["text_light"],
                   relief=tk.FLAT, borderwidth=0)
asl_btn.pack(side=tk.LEFT, padx=5)

isl_btn = tk.Button(lang_frame, text="Ngôn ngữ kí hiệu Ấn", command=switch_to_code2,
                   font=("Segoe UI", 11, "bold"), width=20, height=2,
                   bg=COLORS["primary"], fg=COLORS["text_light"],
                   relief=tk.FLAT, borderwidth=0)
isl_btn.pack(side=tk.LEFT, padx=5)

rsl_btn = tk.Button(lang_frame, text="Ngôn ngữ kí hiệu Nga", command=switch_to_code3,
                   font=("Segoe UI", 11, "bold"), width=20, height=2,
                   bg=COLORS["primary"], fg=COLORS["text_light"],
                   relief=tk.FLAT, borderwidth=0)
rsl_btn.pack(side=tk.LEFT, padx=5)

# Second row of buttons - Charts and Clear
chart_frame = tk.Frame(button_frame, bg=COLORS["background"])
chart_frame.pack(pady=5)

clear_btn = tk.Button(chart_frame, text="Xóa", command=clear_text,
                     font=("Segoe UI", 10, "bold"), width=15, height=1,
                     bg=COLORS["warning"], fg=COLORS["text_light"],
                     relief=tk.FLAT, borderwidth=0)
clear_btn.pack(side=tk.LEFT, padx=5)

# Create code label with modern styling
code_label = tk.Label(control_panel, text="Ngôn ngữ kí hiệu mỹ", 
                     font=("Segoe UI", 16, "bold"),
                     bg=COLORS["background"], fg=COLORS["text_light"])
code_label.pack(pady=5)

# Initialize variables for timing and storing symbols
saved_symbols = []
current_word = ""
current_symbol = None
symbol_start_time = None
last_hand_time = time.time()
hand_present = False
recognition_active = False
current_chart_window = None

# Define video loop function with enhanced visuals
def video_loop():
    global use_code1, use_code2, use_code3, saved_symbols, current_word
    global current_symbol, symbol_start_time, last_hand_time, hand_present, recognition_active
    
    success, img = cap.read()
    if not success:
        status_label.config(text="Error: Cannot read from camera")
        window.after(10, video_loop)
        return
        
    imgOutput = img.copy()
    detected_symbol = None
    hands_detected = False

    # Add a stylish border to the video frame
    border_thickness = 5
    cv2.rectangle(imgOutput, (0, 0), (imgOutput.shape[1], imgOutput.shape[0]), 
                 (49, 140, 231), border_thickness)  # RGB values for COLORS["primary"]

    # Run code 1, code 2, or code 3 based on button state
    if use_code1:
        # Code for Ngôn ngữ kí hiệu mỹ
        hands, img = detector1.findHands(img)
        if hands:
            hands_detected = True
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                
                # Draw a more stylish bounding box and label
                # Rounded rectangle for label background
                alpha = 0.7  # Transparency
                overlay = imgOutput.copy()
                cv2.rectangle(overlay, (x - offset, y - offset - 60), 
                             (x - offset + 120, y - offset), 
                             (49, 140, 231), cv2.FILLED)  # Primary color
                cv2.addWeighted(overlay, alpha, imgOutput, 1 - alpha, 0, imgOutput)
                
                # Add the letter with a shadow effect
                cv2.putText(imgOutput, labels[index], (x - offset + 5, y - offset - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)  # Shadow
                cv2.putText(imgOutput, labels[index], (x - offset + 5, y - offset - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)  # Text
                
                # Draw a stylish bounding box
                cv2.rectangle(imgOutput, (x - offset, y - offset), 
                             (x + w + offset, y + h + offset), 
                             (49, 140, 231), 3)  # Primary color
                
                # Add corner highlights to the bounding box
                corner_length = 20
                # Top-left
                cv2.line(imgOutput, (x - offset, y - offset), (x - offset + corner_length, y - offset), 
                        (155, 89, 182), 5)  # Accent color
                cv2.line(imgOutput, (x - offset, y - offset), (x - offset, y - offset + corner_length), 
                        (155, 89, 182), 5)
                # Top-right
                cv2.line(imgOutput, (x + w + offset, y - offset), (x + w + offset - corner_length, y - offset), 
                        (155, 89, 182), 5)
                cv2.line(imgOutput, (x + w + offset, y - offset), (x + w + offset, y - offset + corner_length), 
                        (155, 89, 182), 5)
                # Bottom-left
                cv2.line(imgOutput, (x - offset, y + h + offset), (x - offset + corner_length, y + h + offset), 
                        (155, 89, 182), 5)
                cv2.line(imgOutput, (x - offset, y + h + offset), (x - offset, y + h + offset - corner_length), 
                        (155, 89, 182), 5)
                # Bottom-right
                cv2.line(imgOutput, (x + w + offset, y + h + offset), (x + w + offset - corner_length, y + h + offset), 
                        (155, 89, 182), 5)
                cv2.line(imgOutput, (x + w + offset, y + h + offset), (x + w + offset, y + h + offset - corner_length), 
                        (155, 89, 182), 5)
                
                detected_symbol = labels[index]
            except:
                pass

    elif use_code2:
        # Code for Ngôn ngữ kí hiệu Ấn
        hands, img = detector2.findHands(img)
        if len(hands) >= 1:
            hands_detected = True
            try:
                if len(hands) == 1:
                    x1, y1, w1, h1 = hands[0]['bbox']
                    x, y, w, h = x1, y1, w1, h1
                else:
                    x1, y1, w1, h1 = hands[0]['bbox']
                    x2, y2, w2, h2 = hands[1]['bbox']
                    x, y, w, h = min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2) - min(x1, x2), max(y1 + h1, y2 + h2) - min(
                        y1, y2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier2.getPrediction(imgWhite, draw=False)
                
                # Draw a more stylish bounding box and label (same as ASL but with different color)
                alpha = 0.7
                overlay = imgOutput.copy()
                cv2.rectangle(overlay, (x - offset, y - offset - 60), 
                             (x - offset + 120, y - offset), 
                             (46, 204, 113), cv2.FILLED)  # Secondary color
                cv2.addWeighted(overlay, alpha, imgOutput, 1 - alpha, 0, imgOutput)
                
                cv2.putText(imgOutput, labels[index], (x - offset + 5, y - offset - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)  # Shadow
                cv2.putText(imgOutput, labels[index], (x - offset + 5, y - offset - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)  # Text
                
                cv2.rectangle(imgOutput, (x - offset, y - offset), 
                             (x + w + offset, y + h + offset), 
                             (46, 204, 113), 3)  # Secondary color
                
                # Add corner highlights
                corner_length = 20
                # Top-left
                cv2.line(imgOutput, (x - offset, y - offset), (x - offset + corner_length, y - offset), 
                        (155, 89, 182), 5)
                cv2.line(imgOutput, (x - offset, y - offset), (x - offset, y - offset + corner_length), 
                        (155, 89, 182), 5)
                # Top-right
                cv2.line(imgOutput, (x + w + offset, y - offset), (x + w + offset - corner_length, y - offset), 
                        (155, 89, 182), 5)
                cv2.line(imgOutput, (x + w + offset, y - offset), (x + w + offset, y - offset + corner_length), 
                        (155, 89, 182), 5)
                # Bottom-left
                cv2.line(imgOutput, (x - offset, y + h + offset), (x - offset + corner_length, y + h + offset), 
                        (155, 89, 182), 5)
                cv2.line(imgOutput, (x - offset, y + h + offset), (x - offset, y + h + offset - corner_length), 
                        (155, 89, 182), 5)
                # Bottom-right
                cv2.line(imgOutput, (x + w + offset, y + h + offset), (x + w + offset - corner_length, y + h + offset), 
                        (155, 89, 182), 5)
                cv2.line(imgOutput, (x + w + offset, y + h + offset), (x + w + offset, y + h + offset - corner_length), 
                        (155, 89, 182), 5)
                
                detected_symbol = labels[index]
            except:
                pass

    elif use_code3:
        # Code for Ngôn ngữ kí hiệu Nga
        hands, img = detector3.findHands(img)
        if hands:
            hands_detected = True
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier3.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier3.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels_r[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255),
                            2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                
                detected_symbol = labels_r[index]
            except:
                pass

    # Handle hand presence detection
    if hands_detected:
        last_hand_time = time.time()
        hand_present = True
    else:
        hand_present = False

    # Handle symbol detection and timing
    if detected_symbol:
        # If this is a new symbol or we've been showing the same symbol for a while
        if current_symbol != detected_symbol:
            current_symbol = detected_symbol
            symbol_start_time = time.time()
            recognition_active = True
        elif recognition_active and time.time() - symbol_start_time >= 3:  
            # Add the symbol to the current word
            current_word += current_symbol
            
            # Update the display
            text_display.delete(1.0, tk.END)
            text_display.insert(tk.END, " ".join(saved_symbols) + " " + current_word)
            
            # Reset the timer and symbol
            symbol_start_time = time.time()
            current_symbol = None
            recognition_active = False
    
    # Add a space if no hand is detected for 5 seconds and we have a current word
    if not hand_present and time.time() - last_hand_time >= 3 and current_word:
        saved_symbols.append(current_word)
        current_word = ""
        
        # Update the display
        text_display.delete(1.0, tk.END)
        text_display.insert(tk.END, " ".join(saved_symbols))

    # Display status information on the video
    status_text = ""
    if hand_present:
        if current_symbol:
            remaining_time = max(0, 3 - (time.time() - symbol_start_time))
            status_text = f"Recognizing: {current_symbol} ({remaining_time:.1f}s)"
        else:
            status_text = "Hand detected, waiting for stable sign"
    else:
        if current_word:
            remaining_time = max(0, 3 - (time.time() - last_hand_time))
            status_text = f"Waiting for space ({remaining_time:.1f}s)"
        else:
            status_text = "No hand deteced "

    # Display status and current text on the video
    cv2.rectangle(imgOutput, (10, 10), (imgOutput.shape[1]-10, 90), (0, 0, 0), cv2.FILLED)
    cv2.putText(imgOutput, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    display_text = " ".join(saved_symbols)
    if current_word:
        display_text += " " + current_word
    if current_symbol and recognition_active:
        display_text += " [" + current_symbol + "]"
        
    # Split text into multiple lines if too long
    if len(display_text) > 40:
        words = display_text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) > 40:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        if current_line:
            lines.append(current_line)
            
        for i, line in enumerate(lines[-2:]):  # Show only last 2 lines
            cv2.putText(imgOutput, line, (15, 65 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(imgOutput, display_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Convert the OpenCV image to a PIL image
    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgOutput)

    # Convert the PIL image to a Tkinter image
    imgTk = ImageTk.PhotoImage(image=imgPIL)

    # Update the label with the new image
    label.config(image=imgTk)
    label.image = imgTk

    # Call the video_loop function after 10ms
    window.after(10, video_loop)

# Start the video loop
video_loop()

# Start the Tkinter event loop
window.mainloop()
