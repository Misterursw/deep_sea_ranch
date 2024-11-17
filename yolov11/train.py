from ultralytics import YOLO


if __name__ == '__main__':  
    # Load a model
    model = YOLO("yolo11n.yaml")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        cache = False,
        data = "data.yaml",
        imgsz = 640,
        epochs = 10,
        batch = 4,
        close_mosaic = 0,
        workers = 4,
        optimizer = "SGD",
        resume = True,
        project = "runs/train",
        name = "exp",
    )
