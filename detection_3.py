from ultralytics import YOLO
import cv2,time
import numpy as np
from shapely.geometry import Polygon
#設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

target='video/highway.mp4'
model = YOLO('yolov8n.pt')  # n,s,m,l,x 五種大小

area=[
    [[250,465],[575,465],[531,609],[20,609]], #南下
    [[706,465],[1100,465],[1400,609],[734,609]] #北上
]

#繪製區域
def drawArea(f,area,color,th):
    for a in area:
        v =  np.array(a, np.int32)
        cv2.polylines(f, [v], isClosed=True, color=color, thickness=th)
    return f

#取得重疊比例
def inarea(object,area):
    inAreaPercent=[]
    b=[[object[0],object[1]],[object[2],object[1]],[object[2],object[3]],[object[0],object[3]]]
    for i in range(len(area)):        
        poly1 = Polygon(b)
        poly2 = Polygon(area[i])
        intersection_area = poly1.intersection(poly2).area
        poly1Area = poly1.area        
        #union_area = poly1.union(poly2).area
        overlap_percent = (intersection_area / poly1Area) * 100
        inAreaPercent.append(overlap_percent)
    return inAreaPercent

names=model.names

cap=cv2.VideoCapture(target)
SouthTrackList=[] #南下
NorthTrackList=[] #北上

while 1:
    try:
        st=time.time()  
        r,frame = cap.read()
        if not r: # r:ret=true讀取成功, false讀取失敗
            break

        results = model.track(frame,persist=True, verbose=False)
        
        #frame= results[0].plot()
        #自己畫bbox(外框)
        for data in results[0].boxes.data:
            x1,y1,x2,y2=int(data[0]),int(data[1]),int(data[2]),int(data[3])
            tid=int(data[4])
            r=round(float(data[5]),2)
            n=names[int(data[6])]
            #畫區域
            drawArea(frame,[area[0]],(0,255,0),5)
            drawArea(frame,[area[1]],(255,0,0),5)

            #取得各區域重疊比例
            p = inarea((x1,y1,x2,y2),area)

            #畫框(1.車類 2.區域內)
            if n in ['car','truck','bus']:
                if p[0]>=30: #南下 綠
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                    #寫上物件名與編號
                    cv2.putText(frame,n + '(#' + str(tid) + ')',(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                    if tid not in SouthTrackList:
                        SouthTrackList.append(tid)

                if p[1]>=30: #北上 藍
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
                    #寫上物件名與編號
                    cv2.putText(frame,n + '(#' + str(tid) + ')',(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                    if tid not in NorthTrackList:
                        NorthTrackList.append(tid)

        SouthCount,NorthCount = len(SouthTrackList),len(NorthTrackList)
        cv2.putText(frame, 'South:' + str(SouthCount), (300, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'North:' + str(NorthCount), (800, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)


        et=time.time()    
        FPS=round((1/(et-st)),1)
        cv2.putText(frame, 'FPS=' + str(FPS), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('YOLOv8', frame)
        key=cv2.waitKey(1)
        if key==27:
            break
        
    except Exception as e :
        print(e)
        break
