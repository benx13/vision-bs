import cv2
import numpy as np
from random import randint   

WALL_X = 1280
PADDLE_SIZE = 50
BALL_VELOCITY = 5

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading xml file")
    exit(0)

def game():
    ball_dx = 12 
    ball_dy = BALL_VELOCITY 

    ball_x1 = 90 
    ball_x2 = 100 
    ball_y1 = 150 
    ball_y2 = 160 

    paddle_x = 0
    paddle_y = 600

    bricks_x = 100
    bricks_y = 50

    score_points=0

    briks = []

    for i in range(4):
        briks.append([])
        for j in range(18):
            briks[i].append([])
        for j in range(18):
            brick_x = bricks_x + 60*j
            brick_y = bricks_y + 20*i
            briks[i][j] = str(brick_x)+"_"+str(brick_y)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read( )
        
        frame = cv2.resize(frame, (1280, 720))
        
        hsv = cv2.cvtColor( frame ,cv2.COLOR_BGR2HSV ) 
        frame = cv2.flip(frame, 1)

        lower, upper = (80, 130, 80), (150, 255, 255)
        
        mask = cv2.inRange(hsv, lower, upper) 

        kernel = np.ones((5, 5), np.uint8 )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=30)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=30)

        contours, _ = cv2.findContours( mask ,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE )
        
        if not contours:
            cv2.putText(frame, "Hold a blue object in front of the camera to play", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            big_countour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(big_countour)
            game_image = cv2.rectangle( frame,( WALL_X-(paddle_x-PADDLE_SIZE) ,paddle_y ), ( WALL_X-(paddle_x+PADDLE_SIZE) ,paddle_y+10 ), ( 255 ,255 ,255 ), -1 )
            cv2.rectangle(frame,(WALL_X-x-w,y+h), (WALL_X-x,y),(255, 255, 255) ,2)
            paddle_x = int((x+(w/2)))
        
        ball_x1 = ball_x1 + ball_dx
        ball_y1 = ball_y1 + ball_dy
        ball_y2 = ball_y2 + ball_dy
        ball_x2 = ball_x2 + ball_dx

        game_image = cv2.circle(frame, (ball_x1, ball_y1), 8, (255, 255, 255), -1)
        
        for i in range(4):
            for j in range(18):
                rec = briks[i][j]
                if rec != []:
                    rec1 = str(rec)
                    rec_1 = rec1.split("_")
                    x12 = int(rec_1[0])
                    y12 = int(rec_1[1])
                game_image = cv2.rectangle(frame,(x12,y12),(x12+50,y12+10),(210,90+(10*j),110+(20*j)),-1)
       
        if ( ball_x2 >= WALL_X):
            ball_dx = -(randint(1, 5))
            
        for i in range(4):
            for j in range(18):
                brick = briks[i][j]
                if brick != []:
                    brick_ = str(brick)
                    brick__ = brick_.split("_")
                    brick_x = int (brick__[0])
                    brick_y = int (brick__[1])
                    if (((brick_x <= ball_x2 and brick_x+50 >=ball_x2) or (brick_x <= ball_x1 and brick_x+50 >=ball_x1)) and ball_y1<=brick_y ) or (ball_y1<=50):
                        ball_dy = BALL_VELOCITY
                        briks[i][j]=[]
                        score_points = score_points+1
                        break                       
                       
        cv2.putText(game_image, "SCORE : "+str(score_points), (230, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 120, 120), 2)
                         
        if ( ball_x1 <= 0 ):
            ball_dx = randint(1,5)

        if ( ball_y2 >= paddle_y ):
            if (WALL_X-( paddle_x-PADDLE_SIZE ) >= ball_x2 and WALL_X-( paddle_x+PADDLE_SIZE ) <= ball_x2) or (WALL_X-( paddle_x-PADDLE_SIZE ) >= ball_x1 and WALL_X-( paddle_x+PADDLE_SIZE ) <= ball_x1):
                ball_dy = -BALL_VELOCITY
        
        if ball_y2 > paddle_y+20:
            cv2.putText(game_image ,"GAME OVER! pres 'r' to play again" ,(500, 700) ,cv2.FONT_HERSHEY_SIMPLEX ,1 ,(255, 255, 255) ,2 )        
            if ball_y2 > paddle_y+120:
                break
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    while True:
        if cv2.waitKey(1) & 0xFF == ord("r"):
            break
while True:
    game()