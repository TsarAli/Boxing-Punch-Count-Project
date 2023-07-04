from ultralytics import YOLO
import numpy as np
import math
import cv2

def update_score(results, prev_contact_b1g2, prev_contact_b2g1, score_boxer1, score_boxer2):
    boxes = results[0].boxes.xyxy
    boxes = boxes.detach().cpu().numpy()
    classes = results[0].boxes.cls
    classes = classes.detach().cpu().numpy()
    
    box_boxer1 = boxes[np.where(classes == 0)]
    box_boxer2 = boxes[np.where(classes == 1)]
    box_glove1 = boxes[np.where(classes == 2)]
    box_glove2 = boxes[np.where(classes == 3)]
    contact_b1g2 = False
    contact_b2g1 = False

    if box_boxer2.shape[0] != 0:
        if box_glove1.shape[0] == 1:
            contact_b2g1 = contact(box_boxer2[0], box_glove1[0])
        else:
            all_b2g1 = []
            for i in range(box_glove1.shape[0]):
                contact_b2g1 = contact(box_boxer2[0], box_glove1[i])
                all_b2g1.append(contact_b2g1)
            if any(all_b2g1):
                contact_b2g1 = True
            else:
                contact_b2g1 = False

    if box_boxer1.shape[0] != 0:
        if box_glove2.shape[0] == 1:
            contact_b1g2 = contact(box_boxer1[0], box_glove2[0])
        else:
            all_b1g2 = []
            for i in range(box_glove2.shape[0]):
                contact_b1g2 = contact(box_boxer1[0], box_glove2[i])
                all_b1g2.append(contact_b1g2)
            if any(all_b1g2):
                contact_b1g2 = True
            else:
                contact_b1g2 = False        

    if contact_b1g2 and not prev_contact_b1g2:
        score_boxer2 += 1
    if contact_b2g1 and not prev_contact_b2g1:
        score_boxer1 += 1

    return score_boxer1, score_boxer2, contact_b1g2, contact_b2g1

def contact(box1, box2):
    Intsec_X1 = max(box1[0], box2[0])
    Intsec_Y1 = max(box1[1], box2[1])
    Intsec_X2 = min(box1[2], box2[2])
    Intsec_Y2 = min(box1[3], box2[3])
    Intsec_w = Intsec_X2 - Intsec_X1
    Intsec_h = Intsec_Y2 - Intsec_Y1
    if Intsec_w >= 0 and Intsec_h >= 0:
        return True
    else:
        return False
    

def update_hit_ratio(results, prev_dis_b1g1, prev_dis_b2g2):
    boxes = results[0].boxes.xyxy
    boxes = boxes.detach().cpu().numpy()
    classes = results[0].boxes.cls
    classes = classes.detach().cpu().numpy()
    
    box_boxer1 = boxes[np.where(classes == 0)]
    box_boxer2 = boxes[np.where(classes == 1)]
    box_glove1 = boxes[np.where(classes == 2)]
    box_glove2 = boxes[np.where(classes == 3)]
    dis_b1g1 = 0
    dis_b2g2 = 0
    threshold_b1 = 0
    threshold_b2 = 0
    global boxer1_thrown
    global boxer2_thrown
    global score_1
    global score_2

    if box_boxer1.shape[0] != 0:
        threshold_b1 = box_boxer1[0][2] - box_boxer1[0][0]

        if box_glove1.shape[0] == 1:
            dis_b1g1 = dis(box_boxer1[0], box_glove1[0])
        elif box_glove1.shape[0] > 1:
            all_dis_b1g1 = []
            for i in range(box_glove1.shape[0]):
                dis_b1g1 = dis(box_boxer1[0], box_glove1[i])
                all_dis_b1g1.append(dis_b1g1)
            dis_b1g1 = max(all_dis_b1g1)

    if box_boxer2.shape[0] != 0:
        threshold_b2 = box_boxer2[0][2] - box_boxer2[0][0]

        if box_glove2.shape[0] == 1:
            dis_b2g2 = dis(box_boxer2[0], box_glove2[0])
        elif box_glove2.shape[0] > 1:
            all_dis_b2g2 = []
            for i in range(box_glove2.shape[0]):
                dis_b2g2 = dis(box_boxer2[0], box_glove2[i])
                all_dis_b2g2.append(dis_b2g2)
            dis_b2g2 = max(all_dis_b2g2) 

    if dis_b1g1 > threshold_b1 and prev_dis_b1g1 < threshold_b1:
        boxer1_thrown += 1
    if dis_b2g2 > threshold_b2 and prev_dis_b2g2 < threshold_b2:
        boxer2_thrown += 1

    hit_ratio_boxer1 = score_1 / boxer1_thrown if boxer1_thrown > 0 else 0
    hit_ratio_boxer2 = score_2 / boxer2_thrown if boxer2_thrown > 0 else 0

    return dis_b1g1, dis_b2g2, hit_ratio_boxer1, hit_ratio_boxer2

def dis(box1, box2):
    return math.sqrt(((box1[2]+box1[0])/2 - (box2[2]+box2[0])/2) ** 2 + ((box1[3]+box1[1])/2 - (box2[3]+box2[1])/2) ** 2)

#-----------------------------------------------
if __name__ == "__main__":
    video_path = "/home/featurize/work/ultralytics/boxing1_cut_final.mp4"
    model = YOLO("/home/featurize/work/ultralytics/runs/segment/train8/weights/last.pt")

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter("boxing1_cut_final_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    prev_contact_b1g2 = False
    prev_contact_b2g1 = False
    score_1 = 0
    score_2 = 0

    prev_dis_b1g1 = 0
    prev_dis_b2g2 = 0
    boxer1_thrown = 0
    boxer2_thrown = 0

    # i = 0
    while (1):
        ret, frm = cap.read()
        if not ret:
            break

        result = model.predict(frm)
        frm = result[0].plot()

        score_1, score_2, contact_b1g2, contact_b2g1 = update_score(result, prev_contact_b1g2, prev_contact_b2g1, score_1, score_2)
        dis_b1g1, dis_b2g2, hit_ratio_1, hit_ratio_2 = update_hit_ratio(result, prev_dis_b1g1, prev_dis_b2g2)  ##
        prev_contact_b1g2 = contact_b1g2
        prev_contact_b2g1 = contact_b2g1
        prev_dis_b1g1 = dis_b1g1
        prev_dis_b2g2 = dis_b2g2

        text_1 = f"Boxer1 valid hit: {score_1}, valid hit ratio: {hit_ratio_1*100:.2f}%"
        text_2 = f"Boxer2 valid hit: {score_2}, valid hit ratio: {hit_ratio_2*100:.2f}%"
        cv2.putText(frm, text_1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) #size,BGR,thickness
        cv2.putText(frm, text_2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        output.write(frm)

        # cv2.imwrite(f"output/{i}.jpg", frm)
        # i += 1

cap.release()
output.release()
cv2.destroyAllWindows()