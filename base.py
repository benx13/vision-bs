import cv2
import numpy as np
from random import randint   

INDEX2COLOR_RANGE = {0:[(50, 100, 100),(113, 255, 255)],
                     1:[(0, 50, 0), (100, 255, 100)],
                     2:[(0, 50, 50), (10, 255, 255)]}
INDEX2COLOR = {0:'blue', 1:'green', 2:'red'}

WALL_X = 1280
PADDLE_SIZE = 25
BALL_VELOCITY = 5
COLOR = 0
TRACK = 0

def onpaddlebar(x):
    global PADDLE_SIZE
    PADDLE_SIZE = x

def onballvelocity(x):
    global BALL_VELOCITY
    BALL_VELOCITY = x

def oncolor(x):
    global COLOR
    COLOR = x

def ontrack(x):
    global TRACK
    TRACK = x

cv2.namedWindow('frame')
cv2.createTrackbar('Paddle size:', 'frame' , 50, 300, onpaddlebar)
cv2.createTrackbar('Ball velocity:', 'frame' , 5, 30, onballvelocity)
cv2.createTrackbar('0:Color, 1:Face: ', 'frame', 0, 2, ontrack)
cv2.createTrackbar('0:B, 1:G, 2:R: ', 'frame', 0, 2, oncolor)

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading xml file")
    exit(0)

def game():
    # Ball movement in x and y direction in pixels per frame
    ball_dx = 12 
    ball_dy = BALL_VELOCITY 

    # Ball top left and bottom right coordinates
    ball_x1 = 90 
    ball_x2 = 100 
    ball_y1 = 150 
    ball_y2 = 160 

    # Paddle coordinates
    paddle_x = 350
    paddle_y = 600

    # Bricks starting x and y coordinates
    bricks_x = 100
    bricks_y = 50

    score_points=0

    # Initialize empty list of bricks
    briks = []

    for i in range(4):
        briks.append([])
        for j in range(18):
            briks[i].append([])
        for j in range(18):
            brick_x = bricks_x + 60*j
            brick_y = bricks_y + 20*i
            briks[i][j] = str(brick_x)+"_"+str(brick_y)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Main game loop
    while True:
        _, frame = cap.read( )
        if(TRACK == 0):
            # Convert frame to HSV color space
            hsv = cv2.cvtColor( frame ,cv2.COLOR_BGR2HSV ) 

            # Define lower and upper bounds for blue color in HSV color space
            lower = np.array(INDEX2COLOR_RANGE[COLOR][0])
            upper = np.array(INDEX2COLOR_RANGE[COLOR][1])        
            
            # Create mask for blue color
            mask = cv2.inRange(hsv, lower, upper) 

            # Apply closing and opening transformations to mask to remove noise and smooth edges
            kernel = np.ones((5, 5), np.uint8 )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=30)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=30)

            # Find contours in mask
            contours, _ = cv2.findContours( mask ,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE )
            
            # If no blue object is detected, display message to hold a blue object in front of the camera        
            if not contours:
                cv2.putText(frame, "Hold a "+INDEX2COLOR[COLOR]+" object in front of the camera to play", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Find the largest contour in the mask
                big_countour = max(contours, key=cv2.contourArea)
                # Calculate bounding rectangle for the contour
                x, y, w, h = cv2.boundingRect(big_countour)
                # Draw the paddle on frame
                game_image = cv2.rectangle( frame,( WALL_X-(paddle_x-PADDLE_SIZE) ,paddle_y ), ( WALL_X-(paddle_x+PADDLE_SIZE) ,paddle_y+10 ), ( 255 ,255 ,255 ), -1 )
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                paddle_x = int((x+(w/2)))
        if(TRACK == 1):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 4, 3)
            if faces == ():
                cv2.putText(frame, "Hold a face in front of the camera to play", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                game_image = cv2.rectangle( frame,( WALL_X-(paddle_x-PADDLE_SIZE) ,paddle_y ), ( WALL_X-(paddle_x+PADDLE_SIZE) ,paddle_y+10 ), ( 255 ,255 ,255 ), -1 )
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
                paddle_x = int((x+(w/2)))
        # Update the ball coordinates based on the ball movement
        ball_x1 = ball_x1 + ball_dx
        ball_y1 = ball_y1 + ball_dy
        ball_y2 = ball_y2 + ball_dy
        ball_x2 = ball_x2 + ball_dx

        #Draw ball on frame
        game_image = cv2.circle(frame, (ball_x1, ball_y1), 8, (255, 255, 255), -1)
        
        #Draw bricks on frame
        for i in range(4):
            for j in range(18):
                rec = briks[i][j]
                if rec != []:
                    rec1 = str(rec)
                    rec_1 = rec1.split("_")
                    x12 = int(rec_1[0])
                    y12 = int(rec_1[1])
                game_image = cv2.rectangle(frame,(x12,y12),(x12+50,y12+10),(210,90+(10*j),110+(20*j)),-1)
       
        #if ball hits right wall invert direction
        if ( ball_x2 >= WALL_X):
            ball_dx = -(randint(1, 5))
            
        #for each brick check check if ball hits brick increase score 
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
                         
        #check if the ball hits left corner
        if ( ball_x1 <= 0 ):
            ball_dx = randint(1,5)

        #check if the ball hits paddle and revenrse y direction
        if ( ball_y2 >= paddle_y ):
            if (WALL_X-( paddle_x-PADDLE_SIZE ) >= ball_x2 and WALL_X-( paddle_x+PADDLE_SIZE ) <= ball_x2) or (WALL_X-( paddle_x-PADDLE_SIZE ) >= ball_x1 and WALL_X-( paddle_x+PADDLE_SIZE ) <= ball_x1):
                ball_dy = -BALL_VELOCITY
        
        #if ball goes below y print game over
        if ball_y2 > paddle_y:
            cv2.putText(game_image ,"GAME OVER! pres 'r' to play again" ,(500, 700) ,cv2.FONT_HERSHEY_SIMPLEX ,1 ,(255, 255, 255) ,2 )        
            #when ball hits bottom of screen break the game
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