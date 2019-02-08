import cv2
import numpy as np
def expend_lines(image):
    height=image.shape[0]
    MIN_K=0.2
    draw_lines=[]
    #gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # 通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(image, 1, np.pi /180,20,10,10)
    #消除斜率相近的直线
    total_k=[]
    flag=True
    for line in lines:
        x1,y1,x2,y2 = line[0]
        #计算直线的斜率
        k=(y2-y1)/(x2-x1)
        for i in range(len(total_k)):
            distance=abs(k-total_k[i])
            if distance<MIN_K:
                flag=False
                break
        if flag:
            draw_lines.append([x1,y1,x2,y2])
            total_k.append(k)
        flag=True
    #找到起始点
    start_point=[]
    for i in range(len(draw_lines)):
        y1=draw_lines[i][1]
        y2=draw_lines[i][3]
        if y1>y2:
            start_point.append([draw_lines[i][0],draw_lines[i][1]])
        else:
            start_point.append([draw_lines[i][2],draw_lines[i][3]])
    #画延长线
    for i in range(len(draw_lines)):
        x1=draw_lines[i][0]
        y1=draw_lines[i][1]
        y2=160
        x2=int(x1+((y2-y1)/total_k[i]))
        y3=height
        x3=int(x2-((y2-y3)/total_k[i]))
        #cv2.line(image,(x2,y2),(x3,y3),(0,0,255),2)
    print(total_k)
    print(start_point)
    #cv2.imshow("line_detect_possible_demo",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
"""
#获取到白色像素的坐标
for row in range(heights):
    for col in range(weights):
        for c in range(channels):
            pv=image[row,col,c]
            if pv==255:
                points.append((col,row))
                test_image[row,col,c]=pv
                print((col,row))
                cv2.imshow('test',test_image)
                cv2.waitKey(1)
"""    
if __name__ == "__main__":
    image=np.array(cv2.imread('binary_image.png'))
    expend_lines(image)
