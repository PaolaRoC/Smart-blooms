# -*- coding: utf-8 -*-
"""
@author: Paola
"""

# Importing all necessary libraries
import cv2
import os

   
def save_frame_range(in_adress, out_adress, start_frame, stop_frame, step_frame):
    cam= cv2.VideoCapture(in_adress)
    
  
    try:
    	
    	# creating a folder named data
    	if not os.path.exists(out_adress):
    		os.makedirs(out_adress)
    
    # if not created then raise error
    except OSError:
    	print ('Error: Creating directory')
    
    # frame
    currentframe = 0
    
    for n in range(start_frame, stop_frame, step_frame):
        
        cam.set(cv2.CAP_PROP_POS_FRAMES, n)
        
    	# reading from frame
        ret,frame = cam.read()
        os.chdir(out_adress)
        if ret:
    		# if video is still left continue creating images
            
            
            name = out_adress.split('\\')[-1]+'_frame_' + str(currentframe) + '.jpeg'
            
    	
    
    		# writing the extracted images
            cv2.imwrite(name, frame)
    
    		# increasing counter so that it will
    		# show how many frames are created
            currentframe += 1
        else:
            return
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

path_in = r"...\dataset404\nuevos_videos\20230410_114709.mp4"
path_out = r"...\dataset404\nuevas_imagenes"

stop_fr= 73*30  #seconds * frame/seconds
save_frame_range(path_in,path_out,0,stop_fr,15)
