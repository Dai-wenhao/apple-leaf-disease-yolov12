from ultralytics import YOLO
import os
os.environ["ULTRALYTICS_FONT"] = "Arial"

DATA = r"D:\pycharm\pycharmprojects\YOLO12\dataset\dataset_new_apple_split\data.yaml"
WEIGHT = r"D:\pycharm\pycharmprojects\YOLO12\runs\detect\dc_gsconv_mambaout_main\weights\best.pt"


def main():
    model = YOLO(WEIGHT)
    metrics = model.val(data=DATA, split="test", imgsz=640, device=0, verbose=False, plots=False)

    print("\n========== full(完整模型: DilatedContext+GSConv+Mambaout) 测试集结果 ==========")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}   <-- 填消融表用这个")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"Speed:     {metrics.speed['inference']:.2f} ms")
    print("\n各类别 mAP50-95:")
    for i, name in metrics.names.items():
        print(f"  {name}: {metrics.box.ap[i]:.4f}")


if __name__ == "__main__":
    main()
