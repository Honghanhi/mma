import tensorflow as tf
import tensorflow.keras
import numpy as np
import cv2
import time


class UTF8Classifier:
    """
    A classifier for UTF-8 encoded text labels using TensorFlow models.
    Enhanced with modern visualization features and improved prediction display.
    """
    
    def __init__(self, modelPath, labelsPath=None):
        """
        Initialize the UTF8Classifier with a model and optional labels.
        
        Args:
            modelPath (str): Path to the TensorFlow model file
            labelsPath (str, optional): Path to the labels file with UTF-8 encoding
        """
        self.model_path = modelPath
        np.set_printoptions(suppress=True)
        
        # Load the model
        print(f"Loading model from {self.model_path}...")
        self.model = tensorflow.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        
        # Initialize data array for predictions
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # Load labels if provided
        self.labels_path = labelsPath
        self.list_labels = []
        if self.labels_path:
            try:
                with open(self.labels_path, "r", encoding="utf-8") as label_file:
                    self.list_labels = [line.strip() for line in label_file]
                print(f"Loaded {len(self.list_labels)} labels successfully")
            except Exception as e:
                print(f"Error loading labels: {e}")
        else:
            print("No Labels File Provided")
            
        # Colors for visualization
        self.colors = {
            'primary': (0, 120, 255),     # Orange
            'secondary': (255, 255, 255), # White
            'background': (0, 0, 0),      # Black
            'highlight': (0, 255, 0)      # Green
        }
        
        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_prediction_time = time.time()
        self.confidence_threshold = 0.7

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=1.2, color=None):
        """
        Get prediction from the model for the given image.
        
        Args:
            img: Input image
            draw (bool): Whether to draw the prediction on the image
            pos (tuple): Position to draw the text
            scale (float): Scale of the text
            color: Color of the text (if None, use default color scheme)
            
        Returns:
            tuple: (prediction_array, index_of_highest_prediction)
        """
        # Preprocess the image
        imgS = cv2.resize(img, (224, 224))
        image_array = np.asarray(imgS)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        
        # Get prediction
        prediction = self.model.predict(self.data, verbose=0)
        indexVal = np.argmax(prediction)
        confidence = prediction[0][indexVal]
        
        # Draw prediction on image if requested
        if draw and self.labels_path and len(self.list_labels) > indexVal:
            # Use default color scheme if no color is provided
            if color is None:
                color = self.colors['primary']
                
            # Calculate text size for background rectangle
            text = str(self.list_labels[indexVal])
            text_size = cv2.getTextSize(text, self.font, scale, 2)[0]
            
            # Draw background rectangle
            cv2.rectangle(img, 
                         (pos[0] - 10, pos[1] - text_size[1] - 10),
                         (pos[0] + text_size[0] + 10, pos[1] + 10),
                         self.colors['background'], 
                         cv2.FILLED)
            
            # Draw border for the rectangle
            cv2.rectangle(img, 
                         (pos[0] - 10, pos[1] - text_size[1] - 10),
                         (pos[0] + text_size[0] + 10, pos[1] + 10),
                         color, 
                         2)
            
            # Draw the prediction text
            cv2.putText(img, text, pos, self.font, scale, self.colors['secondary'], 2)
            
            # Draw confidence percentage if above threshold
            if confidence > self.confidence_threshold:
                confidence_text = f"{confidence * 100:.1f}%"
                cv2.putText(img, confidence_text, 
                           (pos[0], pos[1] + 25), 
                           self.font, scale * 0.7, 
                           self.colors['highlight'], 1)
        
        return list(prediction[0]), indexVal
