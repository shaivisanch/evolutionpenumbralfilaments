#IR Fever Detector, v2 (pure IR)
#author: Isaac Berez
#last updated: 10/28/20

#description: 
#This script measures forehead temperature using an IR camera and TR (thermal reference) surface.
#It assumes a constant thermal reference surface of 40 degC to account any thermal fluctuations of the IR camera.
#Face detection is executed with a tenserflow lite model trained on and designed for thermal face detection.
#More information about the model can be found here: https://github.com/maxbbraun/thermal-face
#The camera is accessed using a custom python wrapper for libUVC called vidcap_C.
#Calibration files are read in to convert the camera's raw counts/pixel into temp/pixel.


import time
import vidcap_c
import vidcap
import cv2
import pandas as pd
from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np
import time
from matplotlib import pyplot as plt 
import pyautogui
import datetime





#note: run using thermal face venv

#load gain and offset matrixes
#gains matrix
#gains = pd.read_csv('/home/ir-cam-4/Desktop/LWIR_fever_proj/gainsMatrix_S0060422.csv', header=None)
#gains = pd.read_csv('/home/ir-cam-4/Desktop/LWIR_fever_proj/gainsMatrix_S0060420.csv', header=None)
gains = pd.read_csv('/home/ir-cam-4/Desktop/LWIR_fever_proj/gainsMatrix_S0060420_oct_cal.csv', header=None)

#print('gains', type(gains))
#print('gains single value', gains.loc[1][1])

#offsets matrix
#offsets = pd.read_csv('/home/ir-cam-4/Desktop/LWIR_fever_proj/offsetsMatrix_S0060422.csv', header=None)
#offsets = pd.read_csv('/home/ir-cam-4/Desktop/LWIR_fever_proj/offsetsMatrix_S0060420.csv', header=None)
offsets = pd.read_csv('/home/ir-cam-4/Desktop/LWIR_fever_proj/offsetsMatrix_S0060420_oct_cal.csv', header=None)

#load thermal model
print ('Loading face detection model...')
face_detector = tflite.Interpreter(model_path='/home/ir-cam-4/Desktop/LWIR_fever_proj/thermal-face/thermal_face_automl_edge_fast.tflite')
print ('Done. Using', face_detector, 'face detector model.')
face_detector.allocate_tensors()

#get input and output tensors
input_details = face_detector.get_input_details()
output_details = face_detector.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#use libuvc and vidcap_c to capture raw boson pixels
print ('Found UVC devices: \n', vidcap_c.uvc_devices())

#access boson
print ('Initializing Boson capture...')
ir_cap = vidcap_c.UVCdevice(0x9cb,0x4007,'')  
if ir_cap.start(2,1) != 0:
    raise 'could not start boson capture'
a = time.time()
print ('Done.')

#function to convert skin temperature to core temperture 
axConvRatio = 0.43
axConvOffset = 22.57
def convert_to_core(temp_values,ratio,offset):
    return (temp_values*ratio) + offset

#histogram plot function used to get peak histogram of forehead temps
bins = np.arange(15.5,40,.05).tolist()
def plot_hist(forehead_values,face_values,bins,debug_mode):
    #plot live face and forehead hist distributions by setting plot to true
    
    forehead = np.array(forehead_values)
    face = np.array(face_values)
    if debug_mode == True:
        counts, bin_loc, _ = plt.hist(forehead, bins, alpha = 0.5, label='forehead')
        counts_2, bin_loc_2, _2 = plt.hist(face,bins, alpha = 0.5, label='face')
        forehead_hist_peak_temp = bin_loc[counts[0].argmax()]
        plt.title('Temp Histogram')
        plt.legend(loc='upper right')
        plt.pause(0.05)
        plt.clf()
    else:
        counts, bin_loc = np.histogram(forehead, bins)
        forehead_hist_peak_temp = bin_loc[counts[0].argmax()]
    return forehead_hist_peak_temp

#Debug mode - change to true to show live forehead and face histograms, measured blackbody temperature
debug_mode = False
print ('Starting Fever Detect GUI v2')
while True:
    ir_frame = ir_cap.frame
    #print ('data type IR frame', type(ir_frame))
    #print ('data type IR frame single val', type(ir_frame[1,1]))
    
    b = time.time()
    #print('frame time', b - a)
    b = a
    img_array = np.array(ir_frame)
    #print ('data type img array pre transpose', type(img_array))
    #print ('data type img array pre transpose single val', type(img_array[1,1]))
    #print ('shape', img_array.shape)
    #img_array_2d_rt = np.rot90(img_array, k=3)
    img_array_2d_rt=(np.transpose(img_array)).astype(np.float)
    #print ('data type img array', type(img_array_2d_rt))
    #print ('data type img array single val', type(img_array_2d_rt[1,1]))
    #print (np.amin(img_array_2d_rt),np.amax(img_array_2d_rt))
    #new_arry = ((img_array_2d_rt - img_array_2d_rt.min()) * (1/(img_array_2d_rt.max() - img_array_2d_rt.min()) * 255)).astype('uint8')
    #cv2.imshow('new array', new_arry)
    #print ('new_array shape', new_arry.shape)
    raw_shape = img_array_2d_rt.shape
    #print (raw_shape)
    
    #16 bit range
    rg = (25000, 30000)
    # copied from kalles code, scales 16 bit range values to go from 0 to 255
    # removed .clip(0.0, 255.0) before .astype(np.uint8) in kalle's code
    u8 = ((img_array_2d_rt.astype(np.float) - rg[0]) * (255.0/(rg[1]-rg[0]))).astype(np.uint8)
    #u8 = img_array_2d_rt/256
    
    #u8 expected shape
    exp_shape = (256,320)

    #catch occasional opencv bug where u8 passes into cvt color with shape (,0)
    if u8.shape != exp_shape:
        print ('Caught opencv bug... Retrying.')
        continue
        

    #print ('exp shape type', type(exp_shape))
    cv2_img = cv2.cvtColor(cv2.applyColorMap(u8,cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB)
    #print ('cv2 img shape', cv2_img.shape)
    #resize img 
    cv2_img_model = cv2.resize(cv2_img,(width, height))  
    #print (np.amax(img_array),np.amin(img_array))
    input_data = np.array(np.expand_dims(cv2_img_model, axis=0),dtype=np.uint8)
    #run the model with the resized frame
    face_detector.set_tensor(input_details[0]['index'], input_data)
    face_detector.invoke()
    
    #get results
    output_data = face_detector.get_tensor(output_details[0]['index'])
    rects = face_detector.get_tensor(output_details[0]['index'])
    scores = face_detector.get_tensor(output_details[2]['index'])

    #compute temp matrix
    temps_img = np.add(np.multiply(img_array_2d_rt,gains), offsets)
    #plt.figure()
    #plt.imshow(temps_img)
    #plt.colorbar()
    #plt.pause(.5)
    #plt.close()
    
    temps_img_array = np.array(temps_img)
    #print ('temps img mean', np.mean(temps_img))
    #print ('temps img single val', type(temps_img.loc[1][1]))
    temps_img_model = np.resize(temps_img, (width,height))
    #print ('mean temps image model', np.mean(temps_img_model))

    
    #detect blackbody 
    #convert model inputted image to grayscale
    blackbody_img = cv2.cvtColor(cv2_img_model,cv2.COLOR_RGB2GRAY)
    #cv2.imshow('BLACKBODY',blackbody_img)

    #remove noise with blurring function
    blackbody_img_gb = cv2.GaussianBlur(blackbody_img,(5,5),0)
    #cv2.imshow('Gaussian Blurred', blackbody_img_gb)
    
    
    #method 1 - edge/contour detection
    edges = cv2.Canny(blackbody_img_gb,100,200)
    #cv2.imshow('edges',edges)
    dilateSize = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateSize,dilateSize))
    edges_morphed = edges
    edges_morphed = cv2.dilate(edges_morphed, kernel, iterations=5)
    edges_morphed = cv2.erode(edges_morphed,kernel,iterations=5)
    #cv2.imshow("edges_morphed",edges_morphed)

    contours, hierarchy = cv2.findContours(edges_morphed,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print ('length contours ', len(contours))
    #print ('countours', contours)
    clone = cv2_img_model.copy()
    

    for i in contours:
        #print('contour area', cv2.contourArea(i))
        #print ('countour circum', cv2.arcLength(i,True))
        x,y,w,h = cv2.boundingRect(i)
        #identify blackbody based on area and perimeter
        #lab area 450 - 750, length 85-110
        #bldg 252 
        if 450<cv2.contourArea(i)<750 and 80<cv2.arcLength(i,True)<150:
            #print('contour area', cv2.contourArea(i))
            #print ('width,height', w,h)
            #print ('perimeter:', cv2.arcLength(i,True))
            
            #tuck in edges to ensure blackbody surface only
            w_adj = w-4
            h_adj = h-10
            x_offset = 4
            y_offset = 5
            cv2.rectangle(cv2_img_model,(x+x_offset,y+y_offset),(x+w_adj,y+h_adj),(0,0,255),1)
            
            #scale blackbody coordinates to match orginal raw image size
            x_scaled = int(round(x*(cv2_img.shape[1]/cv2_img_model.shape[0])))
            y_scaled = int(round(y*(cv2_img.shape[0]/cv2_img_model.shape[0])))
            w_scaled = int(round(w*(cv2_img.shape[1]/cv2_img_model.shape[0])))
            h_scaled = int(round(h*(cv2_img.shape[0]/cv2_img_model.shape[0])))
            #print ('w type', type(w))
            #print ('w_adj type', type(w_adj))
            

            blackbody_mask = np.zeros((cv2_img.shape), dtype=np.uint8)
            pts = np.array( [[[x_scaled+x_offset,y_scaled+h_adj],[x_scaled+x_offset,y_scaled+y_offset],[x_scaled+x_offset+w_adj,y_scaled+y_offset],[x_scaled+x_offset+w_adj,y_scaled+h_adj]]], dtype=np.int32 )
            cv2.fillPoly(blackbody_mask, pts, (255,255,255))
            blackbody_temps = temps_img_array[np.where((blackbody_mask == (255,255,255)).all(axis=2))]
            avg_measured_blackbody_temp = np.mean(blackbody_temps)
            #print ('avg measured blackbody temp', avg_measured_blackbody_temp)
            #print ('std measured blackbody temp', np.std(blackbody_temps))
            blackbody_actual_temp = 40 
            blackbody_offset = blackbody_actual_temp - avg_measured_blackbody_temp
            #print ('blackbody offset', blackbody_offset)
            temps_img_array += blackbody_offset
    
            #cv2.imshow('blackbody mask', blackbody_mask)

    #cv2.imshow('blackbody bounding box',cv2_img_model)
    

    '''
    #method 2 - detect blackbody as top 10% highest temp pixel values
    threshImg = 1*(temps_img > (np.amin(temps_img) + 0.9*(np.amax(temps_img)-np.amin(temps_img))))
    print ('threshImg data type', type(threshImg))
    print ('threshImg data type single val', threshImg.loc[1][1])
    threshImg_mask = np.array(threshImg,dtype=np.uint8)
    print (threshImg_mask)
    cv2.imshow('threshimg', 255*threshImg_mask)
    '''

    '''
    #home testing debug
    #comment out when using blackbody
    temps_img_model+=16
    '''

    #initialize text for screen
    forehead_text = ''
    position_text = ''
    fail_text = ''
    color = 0
    #filter results to draw face bounding box only if above threshold confidence score
    threshold = 0.5
    for index,score in enumerate(scores[0]):
        #single face detect (index<1)
        if score > threshold and index<1:
            #get face bounding box coodinates from model
            face_box = rects[index][index]
            y_min = int(face_box[0] * height)
            x_min = int(face_box[1] * width)
            y_max = int(face_box[2] * height)
            x_max = int(face_box[3] * width)
            cv2.rectangle(cv2_img_model, (x_min, y_min), (x_max, y_max), (34, 139, 34), 2)
            
            #scale face coordinates back up to orignal raw image size
            y_min_scaled = int(round(y_min*(cv2_img.shape[0]/cv2_img_model.shape[0])))
            x_min_scaled = int(round(x_min*(cv2_img.shape[1]/cv2_img_model.shape[0])))
            y_max_scaled = int(round(y_max*(cv2_img.shape[0]/cv2_img_model.shape[0])))
            x_max_scaled = int(round(x_max*(cv2_img.shape[1]/cv2_img_model.shape[0])))

            #get face pixels and temps on original raw image
            mask_face_scaled = np.zeros((cv2_img.shape), dtype=np.uint8)
            pts = np.array( [[[x_min_scaled,y_max_scaled],[x_min_scaled,y_min_scaled],[x_max_scaled,y_min_scaled],[x_max_scaled,y_max_scaled]]], dtype=np.int32 )
            cv2.fillPoly(mask_face_scaled, pts, (255,255,255))
            face_temp_values = temps_img_array[np.where((mask_face_scaled == (255,255,255)).all(axis=2))]
            
            #print ('face avg temp:', np.mean(face_temp_values))

            #convert face skin temps to core temps
            face_core_temp_values = convert_to_core(face_temp_values,axConvRatio,axConvOffset)
            mean_face_core_temp = np.format_float_positional(np.mean(face_core_temp_values), precision=2)
            #print ('face avg temp after core conversion:', mean_face_core_temp)

            #define forehead region coordinates from face bounding box
            y_lower = y_max - int(0.8*(y_max-y_min))
            x_left = x_min+int(.04*x_min)
            y_upper = y_min+int(.04*y_min)
            x_right = x_max-int(.02*x_max)
            cv2.rectangle(cv2_img_model, (x_left,y_lower),(x_right,y_upper),(255,0,0), 1)

            #scale forehead coordinates back up to original raw image size (256 pixels vertical, 320 pixels horizonatol for Boson 320)
            y_lower_scaled = int(round(y_lower*(cv2_img.shape[0]/cv2_img_model.shape[0])))
            x_left_scaled = int(round(x_left*(cv2_img.shape[1]/cv2_img_model.shape[0])))
            y_upper_scaled = int(round(y_upper*(cv2_img.shape[0]/cv2_img_model.shape[0])))
            x_right_scaled = int(round(x_right*(cv2_img.shape[1]/cv2_img_model.shape[0])))

            #get forehead pixels and temps
            mask_forehead_scaled = np.zeros((cv2_img.shape), dtype=np.uint8)
            pts = np.array( [[[x_left_scaled,y_lower_scaled],[x_left_scaled,y_upper_scaled],[x_right_scaled,y_upper_scaled],[x_right_scaled,y_lower_scaled]]], dtype=np.int32 )
            cv2.fillPoly(mask_forehead_scaled, pts, (255,255,255))
            #cv2.imshow('mask_forehead', mask_forehead)
            forehead_temp_values = temps_img_array[np.where((mask_forehead_scaled == (255,255,255)).all(axis=2))]
            #print('forehead avg temp', np.mean(forehead_temp_values))
            
            #convert skin forehead temp to core temp
            forehead_core_temp_values = convert_to_core(forehead_temp_values,axConvRatio,axConvOffset)
            mean_forehead_core_temp = np.format_float_positional(np.mean(forehead_temp_values), precision=2)
            #print ('forehead avg temp after core conversion:', mean_forehead_core_temp)

            #plot histograms of core converted face and forehead temps and get forehead peak value
            fore_peak_hist_temp = plot_hist(forehead_core_temp_values,face_core_temp_values,bins,debug_mode)
            #print ('fore peak', fore_peak_hist_temp)
            #get max temp value of forehead and face
            maxforeT = max(forehead_core_temp_values)
            maxfaceT = max(face_core_temp_values)
            #print ('maxforeT', maxforeT)
            #print ('maxfaceT', maxfaceT)

            #greg's alg for determing forehead temp (adapted from LM_FEVER_GUI_MTCNN.m)
            if maxfaceT and maxforeT > 35:
                if maxforeT - fore_peak_hist_temp < 1:
                    estTemp = fore_peak_hist_temp
                    estTemp_F = (estTemp*(9/5))+32
                    #print ('est temp is fore peak hist temp', estTemp)
                elif abs(maxforeT-maxfaceT) < 0.5:
                    #estTemp = fore_peak_hist_temp
                    #print ('est temp in abs', estTemp)
                    #print ('maxforeT', maxforeT, 'maxfaceT', maxfaceT)
                    estTemp = maxforeT - 0.5
                    estTemp_F = (estTemp*(9/5))+32
                    #print ('est temp is abs maxfore-maxface', estTemp)
                else:
                    #print ('retest')
                    break
    

            #add forehead temp to screen
            formated_estTemp = np.format_float_positional(estTemp,precision=1)
            formated_estTemp_F = np.format_float_positional(estTemp_F,precision=1)
            forehead_text = str(formated_estTemp +'C' + ' | ' +formated_estTemp_F + 'F' )

            #display temp in red if above 38 deg C, green otherwise
            #print ('type formated est temp', type(formated_estTemp))
            if estTemp > 38:
                color += 1
            else:
                color += 0

            #compute face area
            face_area = (y_max-y_min)*(x_max-x_min)
            #print ('face area', face_area)
            
            #add green border if face area is within reasonable range, yellow if too close, red if too far
            if 1400<face_area<1800:
                #goldilocks zone
                cv2.rectangle(cv2_img_model,(0,5), (cv2_img_model.shape[0]-1,cv2_img_model.shape[1]-26),(0,255,0),2)
                #code for autocapture
                '''
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
                capture = pyautogui.screenshot()
                capture = cv2.cvtColor(np.array(capture),cv2.COLOR_RGB2BGR)
                cv2.imwrite('/home/ir-cam-4/Desktop/LWIR_fever_proj/fever/auto_captured_images/autocap_' + st +'.png',capture)
                '''
            elif face_area<1400:
                #too close
                cv2.rectangle(cv2_img_model,(0,5), (cv2_img_model.shape[0]-1,cv2_img_model.shape[1]-26),(0,255,255),2)
            else:
                #too far
                cv2.rectangle(cv2_img_model,(0,5), (cv2_img_model.shape[0]-1,cv2_img_model.shape[1]-26),(0,0,255),2)

    #cv2.imshow('orig frame', cv2_img_model)

    #GUI display 
    #upsize frame since model input size is small (192x192)
    cv2_img_model_upsized = cv2.resize(cv2_img_model,(960,768))

    #rectangle for GUI text to be printed on
    x,y,w,h = 0,668,960,100
    cv2.rectangle(cv2_img_model_upsized, (x,y),(x+w,y+h),(0,0,0), -1)

    #add blackbody avg measured temp to screen if debug is True
    if debug_mode == True:
        formated_blackbody_Temp = str(int(round(avg_measured_blackbody_temp)))
        blackbody_text = str(formated_blackbody_Temp + 'C')
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org_tr = (x+int((9*w)/10), y + int((2*h)/3)) 
        fontScale = 1 #min(width,height)/(25/scale)
        color = (0, 0, 255) 
        thickness = 1
        cv2.putText(cv2_img_model_upsized,blackbody_text, org_tr, font,  
            fontScale, color, thickness, cv2.LINE_AA) 
    
    #add forehead text to screen
    org_estTemp = (x+int((3*w)/12), y + int((2*h)/3)) 
    forehead_text_color = ''
    color_estTemp = (255,255,255)
    if color == 1:
        color_estTemp = (0, 0, 255)
    if color == 0:
        color_estTemp = (0,255,0) 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 2
    thickness = 4
    cv2.putText(cv2_img_model_upsized,forehead_text, org_estTemp, font,  
            fontScale, color_estTemp, thickness, cv2.LINE_AA)   
    '''
    #add position text to screen
    org_pass_text = (x + int(w/2), y+int((2*h)/3)) 
    color_pass = (0, 255, 0) 
    cv2.putText(cv2_img_model_upsized,pass_text, org_pass_text, font,  
    fontScale, color_pass, thickness, cv2.LINE_AA)
    
    #add fail text to screen
    org_fail_text = (x + int(w/2), y+int((2*h)/3)) 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    color_fail = (0, 0, 255) 
    cv2.putText(cv2_img_model_upsized,fail_text, org_fail_text, font,  
    fontScale, color_fail, thickness, cv2.LINE_AA)
    '''
    cv2.imshow('Fever_Detect_GUI_v2',cv2_img_model_upsized)

    if cv2.waitKey(1) == ord('q'):
        break


#clean up
cv2.destroyAllWindows()
