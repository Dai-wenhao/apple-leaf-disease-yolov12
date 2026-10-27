from ultralytics import YOLO
import os
os.environ["ULTRALYTICS_FONT"] = "Arial"

DATA = r"D:\pycharm\pycharmprojects\YOLO12\dataset\dataset_new_apple_split\data.yaml"

# 待评估的模型（测试集口径）。基线路径需你确认后替换。
models = {
    "基线(turbo)":                r"runs/detect/train_yolov12n_baseline_exp 5.30/weights/best.pt",
    "improved(DilatedContext)":   r"runs/detect/dc_main/weights/best.pt",
    "improved+GSConv":            r"runs/detect/dc_gsconv_main/weights/best.pt",
    "full(+Mambaout)":            r"runs/detect/dc_gsconv_mambaout_main/weights/best.pt",
}


def main():
    print("模型\t\t\tmAP50\tmAP50-95\tSpeed(ms)")
    print("-" * 60)
    for name, weight in models.items():
        if not os.path.exists(weight):
            print(f"{name}: 权重不存在，跳过")
            continue
        model = YOLO(weight)
        metrics = model.val(data=DATA, split="test", imgsz=640, device=0, verbose=False, plots=False)
        print(f"{name}\t{metrics.box.map50:.4f}\t{metrics.box.map:.4f}\t{metrics.speed['inference']:.2f}")


if __name__ == "__main__":
    main()
