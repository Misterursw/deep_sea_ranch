from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('runs/train/exp4/weights/best.pt')
    model.val(data = "data.yaml",
              split = 'val',
              imgsz = 640,
              batch = 4,
              project = "runs/val",
              name = 'exp',
    )
    
