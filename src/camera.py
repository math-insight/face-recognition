import cv2
import threading


class Camera:
    def __init__(self):
        self.frame = None
        self.run_camera = True
        self.camera = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.start()
        
    def _capture_frames(self):
        while self.run_camera:
            success, img = self.camera.read()
            if success:
                self.frame = img
                
    def get_frame(self):
        return self.frame
    
    def stop(self):
        self.run_camera = False
        self.thread.join()
        self.camera.release()