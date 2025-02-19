from ultralytics import YOLO
import cv2,time,numpy as np
from shapely.geometry import Polygon #區域重疊偵測工具
#設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

target='video/market.mp4'
model = YOLO('yolov8n.pt')  # 預設模型：n,s,m,l,x 五種大小

names=model.names #認識的80物件 字典：編號及名稱
print(names)

#區域三維陣列
area=[
    [[658, 59],[800, 59],[735, 244],[708, 321],[700, 382],[681, 471],[688, 563],[645, 836],[390, 832],[490, 501],[628, 216]], #車道0
    [[1029, 59],[1184, 59],[1260, 276],[1301, 486],[1356, 707],[1401, 886],[1089, 892],[1074, 574],[1031, 178]], #車道1
    [[1458, 59],[1682, 59],[1815, 295],[1900, 528],[1912, 622],[1912, 843],[1818, 843],[1815, 713],[1733, 561],[1585, 298],[1464, 155]], #車道2
    ]

#繪製區域   影像,區域座標,顏色,寬度
def drawArea(f,area,color,th):
    for a in area:
        v =  np.array(a, np.int32)
        cv2.polylines(f, [v], isClosed=True, color=color, thickness=th)
    return f

#取得重疊比例  物件,區域
def inarea(object,area):
    inAreaPercent=[] #area陣列，物件在所有區域的比例
    #把物件座標變成多邊形  [x1,y1]左上           [x2,y1]右上            [x2,y2]右下          [x1,y2]左下
    b=[[object[0],object[1]],[object[2],object[1]],[object[2],object[3]],[object[0],object[3]]]
    for i in range(len(area)):        
        poly1 = Polygon(b)
        poly2 = Polygon(area[i])
        intersection_area = poly1.intersection(poly2).area #重疊區域部分畫素
        poly1Area = poly1.area #物件區域畫素
        #union_area = poly1.union(poly2).area
        overlap_percent = (intersection_area / poly1Area) * 100
        inAreaPercent.append(overlap_percent)
    return inAreaPercent



cap=cv2.VideoCapture(target)

while 1:
    st=time.time()  
    r,frame = cap.read()
    if r==False: #讀取失敗
        break
    results = model(frame, verbose=False) #YOLO辨識verbose=False不顯示文字結果
    # frame= results[0].plot() #畫出辨識結果，[0]第一張照片

    #三個區域分開不同顏色
    frame = drawArea(frame,[area[0]],(255,0,0),3)
    frame = drawArea(frame,[area[1]],(0,255,0),3)
    frame = drawArea(frame,[area[2]],(0,0,255),3)

    carCount=[0,0,0] #初始汽車數量
    for box in results[0].boxes.data:
        x1 = int(box[0]) #左
        y1 = int(box[1]) #上
        x2 = int(box[2]) #右
        y2 = int(box[3]) #下
        r = round(float(box[4]),2) #信任度
        n = names[int(box[5])] #名字
        # 跳過非汽車物件
        if n not in ['person'] or r<=0.5:
            continue #下一個
        # 進入區域才畫框
        # 計算物件是否進入區域
        tempObj = [x1,y1,x2,y2,r,n] #重新組合物件
        # 計算物件與區域的重疊比例    物件,區域
        ObjInArea = inarea(tempObj,area)
        #物件在區域0的比例>=25
        #區域0 藍色
        if ObjInArea[0]>=25:
            # 自己畫外框   影像   左上角   右下角
            cv2.rectangle(frame, (x1,y1), (x2,y2),(255,0,0),2 )
            # 寫上物件名稱
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            carCount[0]+=1
        #區域1 綠色
        if ObjInArea[1]>=25:
            # 自己畫外框   影像   左上角   右下角
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2 )
            # 寫上物件名稱
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            carCount[1]+=1
        #區域2 紅色
        if ObjInArea[2]>=25:
            # 自己畫外框   影像   左上角   右下角
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,0,255),2 )
            # 寫上物件名稱
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            carCount[2]+=1

    #白色背景                                           粗細=填滿顏色
    cv2.rectangle(frame,(10,20),(200,180),(255,255,255),-1)
   
    #區域0
    cv2.putText(frame, 'Area0=' + str(carCount[0]), (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
    #區域1
    cv2.putText(frame, 'Area1=' + str(carCount[1]), (20, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
    #區域2
    cv2.putText(frame, 'Area2=' + str(carCount[2]), (20, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
    #合計人數
    cv2.putText(frame, 'Total=' + str(carCount[0]+carCount[1]+carCount[2]), (20, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)

    et=time.time()  
    FPS=round((1/(et-st)),1)
    #在畫面寫字  影像       文字內容       位置(x,y)     字型               大小  顏色(BGR)   粗細   樣式
    cv2.putText(frame, 'FPS=' + str(FPS), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv8', frame)
    key=cv2.waitKey(1)
    if key==27:
        break
