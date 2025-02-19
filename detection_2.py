import cv2, numpy as np
from ultralytics import YOLO
#路徑字典={物件追蹤編號:[移動路徑座標]}
from collections import defaultdict
track_history = defaultdict(lambda: [])

model = YOLO("yolov8n.pt")
names = model.names

cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)
target = "video/city.mp4"
cap = cv2.VideoCapture(target)

while cap.isOpened():
    r, frame = cap.read()
    #成功讀取影像
    if r==False:
        break
        #       啟動辨識+追蹤      持續          不顯示結果
    results = model.track(frame, persist=True, verbose=False)

    for box in results[0].boxes.data:
        x1 = int(box[0]) #左
        y1 = int(box[1]) #上
        x2 = int(box[2]) #右
        y2 = int(box[3]) #下
        trackid = int(box[4]) #追蹤編號
        r = round(float(box[5]),2) #信任度
        n = names[int(box[6])] #名字
       
        #追蹤物件(ClassID=0)
        if n in ['car','bus','truck']:

            #劃出box
            cv2.rectangle(frame,(x1,y1), (x2,y2), (0, 255, 0),5)

            #叫出物件的移動軌跡座標
            track = track_history[trackid]

            #加一個點，物件的中心點為軌跡座標
            track.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
           
            #fps=30, 1秒30點, 10秒=300點
            if len(track) > 300:
                track.pop(0) #刪除最早的那個軌跡

            #現在位置     倒數第一個軌跡座標,半徑,顏色,   填滿(正:外框寬度,負:填滿)
            cv2.circle(frame, (track[-1]), 7, (0,0,255), -1)
            # 軌跡List轉換成陣列
            points = np.array(track)
            #移動路徑                      不封閉:頭尾不相接     顏色            寬度
            cv2.polylines(frame, [points], isClosed=False, color=(0,0,255), thickness=2)

    cv2.imshow('YOLOv8',frame)

    if cv2.waitKey(1) == 27: #ESC退出
        break
