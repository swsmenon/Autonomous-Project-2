# %%
#pip install opencv-python

# %%
#pip install tqdm

# %%
#pip install jupyter

# %%
#pip install matplotlib

# %%
import os
import re
import cv2
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import re
import glob
import os


# %%


# %%
# get file names of frames
col_frames = os.listdir('/home/weston/project/Autonomous-Project-2-main/frames/')
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# load frames
col_images=[]
for i in tqdm(col_frames):
    img = cv2.imread('/home/weston/project/Autonomous-Project-2-main/frames/'+i)
    col_images.append(img)

# %%
# specify frame index
idx = 457

# plot frame
plt.figure(figsize=(10,10))
plt.imshow(col_images[idx][:,:,0], cmap= "gray")
#plt.show()

# %%
# create a zero array
stencil = np.zeros_like(col_images[idx][:,:,0])

# specify coordinates of the polygon
polygon = np.array([[50,270], [220,160], [360,160], [480,270]])

# fill polygon with ones
cv2.fillConvexPoly(stencil, polygon, 1)

# %%
# plot polygon
plt.figure(figsize=(10,10))
plt.imshow(stencil, cmap= "gray")
#plt.show()

# %%
# apply polygon as a mask on the frame
img = cv2.bitwise_and(col_images[idx][:,:,0], col_images[idx][:,:,0], mask=stencil)

# plot masked frame
plt.figure(figsize=(10,10))
plt.imshow(img, cmap= "gray")
#plt.show()

# %%
# apply image thresholding
ret, thresh = cv2.threshold(img, 130, 145, cv2.THRESH_BINARY)

# plot image
plt.figure(figsize=(10,10))
plt.imshow(thresh, cmap= "gray")
#plt.show()

# %%
lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)

# create a copy of the original frame
dmy = col_images[idx][:,:,0].copy()

# draw Hough lines
for line in lines:
  x1, y1, x2, y2 = line[0]
  cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)

# plot frame
plt.figure(figsize=(10,10))
plt.imshow(dmy, cmap= "gray")
#plt.show()

# %%
cnt = 0

for img in tqdm(col_images):
  
  # apply frame mask
  masked = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)
  
  # apply image thresholding
  ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)

  # apply Hough Line Transformation
  lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
  dmy = img.copy()
  
  # Plot detected lines
  try:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(dmy, (x1, y1), (x2, y2), (0, 255, 0), 3)
  
    cv2.imwrite('/home/weston/project/Autonomous-Project-2-main/detected/'+str(cnt)+'.png',dmy)
  
  except TypeError: 
    cv2.imwrite('/home/weston/project/Autonomous-Project-2-main/detected/'+str(cnt)+'.png',img)

  cnt+= 1

# %%
# input frames path
pathIn= '/home/weston/project/Autonomous-Project-2-main/detected/'

# output path to save the video
pathOut = 'roads_v2.mp4'

# specify frames per second
fps = 30.0

# %%
from os.path import isfile, join

# get file names of the frames

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort(key=lambda f: int(re.sub('\D', '', f)))

# %%
frame_list = []

for i in tqdm(range(len(files))):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    #height, width, layers = img.shape
    #size = (width,height)
    #img = cv2.imread(filename)
   # print(img.shape)
    #height, width, layers = img.shape
    size = 480, 270
   
    #inserting the frames into an image array
    frame_list.append(img)

# %%
 
img_array = []
for filename in sorted(glob.glob('/home/weston/project/Autonomous-Project-2-main/detected/*.png'), key=os.path.getmtime):
    
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
   
 
out = cv2.VideoWriter('project_Autonomous1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 50, size)

#print(img_array)
for i in range(len(img_array)):
    
    out.write(img_array[i])
out.release()

# %%
cap = cv2.VideoCapture('project_Autonomous1.avi')

# Check if camera opened successfully
if (cap.isOpened()== False):

	print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):

	  # Capture frame-by-frame
	ret, frame = cap.read()
      
	if ret == True:
	# Display the resulting frame
		cv2.imshow('OutputOfLaneDetection',frame)
    	# Press Q on keyboard to  exit
		if cv2.waitKey(75) & 0xFF == ord('q'):
			break
	else:
		break
    
	
	# Break the loop


	# When everything done, release the video capture object

cap.release()

	 

	# Closes all the frames

cv2.destroyAllWindows()



# %%
