from app import app, init_camera_and_models, shutdown_camera


if __name__ == '__main__':
    init_camera_and_models()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        shutdown_camera()