import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import Jetson.GPIO as GPIO
import time as time          #引用需要用的库
# 設定pin18讀取按鈕數值，再用pin12輸出高、低電位
relay_pin = 11
but_pin0 = 19
but_pin1 = 21
but_pin2 = 23

#GPIO.setup(relay_pin, GPIO.OUT,initial=GPIO.LOW)
#def main():

GPIO.setmode(GPIO.BOARD)
# 設定為BOARD模式
GPIO.setup(relay_pin, GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(relay_pin, GPIO.OUT)
# 設定relay_pin為輸出

GPIO.setup(but_pin0, GPIO.IN)
GPIO.setup(but_pin1, GPIO.IN)
GPIO.setup(but_pin2, GPIO.IN)
# 設定but_pin為輸入



# 將LED燈的電位初始化為低電位

value0 = GPIO.input(but_pin0)
value1 = GPIO.input(but_pin1)
value2 = GPIO.input(but_pin2)
s=0
GPIO.output(relay_pin,GPIO.LOW)

# 定義禁區範圍，這裡使用矩形範圍,4個座標,[左上],[右上],[右下],[左下]
restricted_zone = np.array([[105, 50], [570, 50], [570, 380], [105, 380]], np.int32)
restricted_zone = restricted_zone.reshape((-1, 1, 2))

def detect(save_img=False):
    s = 0  # 初始化 s
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    view_img = True
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = True  # 使用USB攝影機

    # 目錄類的設定
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader，使用 USB 相機的數據流
    cap = cv2.VideoCapture("/dev/video0")

    if not cap.isOpened():
        print("無法打開 USB 攝像頭")
        return

    try:
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        while True:
            value0 = GPIO.input(but_pin0)
            value1 = GPIO.input(but_pin1)
            value2 = GPIO.input(but_pin2)
            print(s)  # 用於偵錯顯示狀態 's'
            time.sleep(1/30)

            if s == 0:
                if value0 == GPIO.HIGH:
                    s = 1
                    GPIO.output(relay_pin, GPIO.HIGH)  # 啟動繼電器
            elif s == 1:
                if value1 == GPIO.HIGH:
                    s = 2
                    GPIO.output(relay_pin, GPIO.LOW)  # 停止繼電器

            elif s == 2:
                if value2 == GPIO.HIGH:
                    s = 0  # 重置狀態

            if s == 0:
                print("狀態：待機")
            elif s == 1:
                print("狀態：啟動")
            elif s == 2:
                print("狀態：停止")

            ret, frame = cap.read()

            #紅色透明濾鏡的禁區
            overlay = frame.copy()
            cv2.fillPoly(overlay, [restricted_zone], (0, 0, 255))
            alpha = 0.2  # 透明度
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            img = cv2.resize(frame, (imgsz, imgsz))
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = img.unsqueeze(0)

            # Inference
            with torch.no_grad():
                pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Process detections
            for det in pred:  # detections per image
                if len(det):
                    # 檢測到物體後，縮放邊界框到原圖尺寸
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                    # 檢測是否有人類（類別 0 是人類）
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls) == 0:  # 人類類別為0
                            # 檢查人是否進入禁區
                            center_x = int((xyxy[0] + xyxy[2]) / 2)
                            center_y = int((xyxy[1] + xyxy[3]) / 2)
                            if cv2.pointPolygonTest(restricted_zone, (center_x, center_y), False) >= 0:
                                print("人進入禁區，觸發 stop 訊號")
                                s = 2
                                GPIO.output(relay_pin, GPIO.LOW)# 急停開
                            plot_one_box(xyxy, frame, label=f'Person {conf:.2f}', color=colors[int(cls)], line_thickness=2)

            # 顯示圖像
            if view_img:
                cv2.imshow("B1USB Camera", frame)
                if cv2.waitKey(1) == ord('q'):  # 按 'q' 鍵退出
                    break

    finally:
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # 使用USB相機
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='dont trace model')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()