from flask import Flask, Response
from camera import Camera
from inferenceprocessor import InferenceProcessor
from inference import load_models
import cv2
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
camera = None
inference_processor = None

def init_camera_and_models():
    global camera, inference_processor
    camera = Camera()
    models = load_models()
    inference_processor = InferenceProcessor(models)

def gen_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(inference_processor.process_frame, frame.copy())
            
        classified_frame = inference_processor.last_prediction if inference_processor.last_prediction is not None else frame
        
        _, jpeg = cv2.imencode('.jpg', classified_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Podgląd z kamery</title>
        <style>
            * {
                box-sizing: border-box;
            }
            body, html {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
            }
            #video-container {
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #000;
            }
            #camera-feed {
                height: 100vh;
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <div id="video-container">
            <img id="camera-feed" src="/video_feed" alt="Podgląd z kamery">
        </div>
    </body>
    </html>
    """
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def shutdown_camera():
    global camera
    if camera:
        camera.stop()