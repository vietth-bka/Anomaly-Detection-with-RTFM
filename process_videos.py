# Importing all necessary libraries
import cv2
import os
import numpy as np

def putText(image, anomal):
    font = cv2.FONT_HERSHEY_SIMPLEX
    w, h = image.shape[1], image.shape[0]
    # org
    org = (10, h-20)

    # fontScale
    fontScale = 3

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    if anomal is True:
        image[h-100:, 0:350, :] = np.array([0, 0, 255])

        # Using cv2.putText() method
        image = cv2.putText(image, 'Anomal', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    else:
        image[h-100:, 0:350, :] = np.array([0, 255, 0])

        # Using cv2.putText() method
        image = cv2.putText(image, 'Normal', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    
    return image

# Read the video from specified path
cam = cv2.VideoCapture("/I3D_extractor/demovideos/NVR-VietDuy_ori.mp4")
scores = np.load('/RTFM/scores/scores_VietDuy_NAdam.npy')
print('scores:', len(scores))

if (cam.isOpened() == False): 
    print("Error reading video file")

frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
# frame
size = (frame_width, frame_height)
result = cv2.VideoWriter('out_videos/VietDuy_NAdam.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
currentframe = 0

while(True):
	
	# reading from frame
    ret,frame = cam.read()

    if ret:
        currentframe += 1
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name, frame.shape)
        print(scores[currentframe])

        if scores[currentframe] >= 0.85:
            print('true')
            frame = putText(frame, True)
        else:
            print('false')
            frame = putText(frame, False)
        result.write(frame)
        # writing the extracted images
        # cv2.imwrite('out_videos/test.jpg', frame)
        # break

        # increasing counter so that it will
        # show how many frames are created
    else:
        break

print(currentframe)

# Release all space and windows once done
result.release()
cam.release()
cv2.destroyAllWindows()
