import numpy as np
import uuid
import os
import torch.nn as nn
import numpy as np
import torch
import cv2

sign_classes = ['Hello', 'Thank you', 'ambulance', 'call', 'doctor', 'hurt', 'road']

joint_list = [[4,3,2],[3,2,1],[2,1,0], 
              [8,7,6],[7,6,5],[6,5,0], 
              [12,11,10],[11,10,9],[10,9,0], 
              [16,15,14],[15,14,13],[14,13,0], 
              [20,19,18],[19,18,17],[18,17,0]] 

def process_image(image, results, classifier):
    if len(results.multi_hand_landmarks) == 2:
        position_hand1 = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
        position_hand2 = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[1].landmark]).flatten()
        angles = get_angles(results).flatten()
        # frame_data = np.concatenate((position_hand1, position_hand2, angles), axis=0)
        frame_data = np.expand_dims(angles, axis=0)
        
        # frame_data = scaler1.transform(frame_data)
        frame_data = torch.tensor(frame_data, dtype=torch.float32)
        output = infer_realtime(classifier, frame_data)

        image = show_subtitle(image, sign_classes[output[0]])
    return image

def process_results(results, classifier):
    # position_hand1 = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
    # position_hand2 = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[1].landmark]).flatten()
    angles = get_angles(results).flatten()
    # frame_data = np.concatenate((position_hand1, position_hand2, angles), axis=0)
    frame_data = np.expand_dims(angles, axis=0)
    
    # frame_data = scaler1.transform(frame_data)
    frame_data = torch.tensor(frame_data, dtype=torch.float32)
    output, prob = infer_realtime(classifier, frame_data)

    return output[0], prob

def show_subtitle(image, sign_class):
    h, w, _ = image.shape
    text_loc = (int(0.5*w), int(0.9*h))
    # cv2.rectangle(image, (text_loc[0]-len(sign_class)*8-5, text_loc[1]-35), (text_loc[0]+len(sign_class)*8+5, text_loc[1]+15), (0, 0, 0), -1)
    x, y, w, h = 100, 100, 200, 100
    sub_img = image[text_loc[1]-50:text_loc[1]+20, text_loc[0]-len(sign_class)*20:text_loc[0]+len(sign_class)*20]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    # Putting the image back to its position
    image[text_loc[1]-50:text_loc[1]+20, text_loc[0]-len(sign_class)*20:text_loc[0]+len(sign_class)*20] = res
    
    cv2.putText(image, sign_class, (text_loc[0]-len(sign_class)*18, text_loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    return image

def infer_realtime(model, input):
    model.eval()  # Set the model to evaluation mode
        
    with torch.no_grad():
        output = model(input)
        prob, predicted = torch.max(output.data, 1)
    return predicted.numpy(), prob

def get_angles(results):
    all_angles = []
    for hand in results.multi_hand_landmarks:
        angle_list = []
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            if angle > 180.0:
                angle = 360 - angle
            angle_list.append(angle)
        all_angles.append(angle_list)
    return np.array(all_angles)