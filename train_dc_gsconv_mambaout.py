from ultralytics import YOLO
import os
os.environ["ULTRALYTICS_FONT"] = "Arial"

DATA = r"D:\pycharm\pycharmprojects\YOLO12\dataset\dataset_new_apple_split\data.yaml"


def train_one(model_yaml, name):
    model = YOLO(model_yaml).load("yolov12n.pt")
    model.train(
        data=DATA,
        epochs=50,
        batch=4,
        imgsz=640,
        device=0,
        workers=2,
        cache=False,
        name=name,
        single_cls=False,
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.2,
        degrees=0,
        shear=0,
        perspective=0,
        fliplr=0.5,
        flipud=0.0,
    )


if __name__ == "__main__":
    train_one("YOLOv12-turbo-DilatedContext-GSConv-Mambaout.yaml", "dc_gsconv_mambaout_main")
    print("improved(DilatedContext) + GSConv + Mambaout 训练完成")
