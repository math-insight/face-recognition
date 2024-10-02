import threading
from inference import classify


class InferenceProcessor:
    def __init__(self, models):
        self.models = models
        self.last_prediction = None
        self.processing = False
        self.lock = threading.Lock()
        
    def process_frame(self, frame):
        with self.lock:
            if self.processing:
                return self.last_prediction
            
            self.processing = True
            
        try:
            result = classify(frame, self.models['face_classifier'],
                              self.models['emotion_model'],
                              self.models['age_model'],
                              self.models['gender_model'])
            
            with self.lock:
                self.last_prediction = result
        finally:
            with self.lock:
                self.processing = False
                
        return result