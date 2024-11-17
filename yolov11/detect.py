from ultralytics import YOLO


if __name__ == '__main__':  
    # Load a model
    model = YOLO("runs/train/exp4/weights/best.pt")  # load a pretrained model (recommended for training)

    model.predict(source = '../dataset/object_detection/test',
                  imgsz = 640,
                  project = "run/detect",
                  name = 'exp',
                  save = True,
                  save_txt = True,
                  )
