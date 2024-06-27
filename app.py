from flask import Flask, json, jsonify
from flask.sansio.scaffold import T_template_context_processor
from flask_cors import CORS
from flask_cors.extension import make_after_request_function
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)
LR=int(input('''Are you on the left side or the right side?
Enter 1 for right and 0 for left:'''))
while LR>=2:
    LR=int(input('Do it again'))
after_y=1
after_ya=1
pre_1=0
pre_2=0
pre_1a=0
pre_2a=0
feedback=[]
name=Flask(__name__)
CORS(name)
@name.route('/test/<item>')
def video_processer(video_source):
    mpDraw = mp.solutions.drawing_utils
    model_path = "pose_landmark_full.task"
    num_poses = 2
    min_pose_detection_confidence = 0.5
    min_pose_presence_confidence = 0.5
    min_tracking_confidence = 0.5
    current_frame_output = None
    last_timestamp_ms = 0
    base_options = python.BaseOptions(model_asset_path=model_path) 
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_source)
    fps = int(cap.get(5))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    output = cv2.VideoWriter('Sam.avi', cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_landmarks_list = result.pose_landmarks
            frame = np.copy(mp_image.numpy_view())
            noses = []
            for person in range(len(pose_landmarks_list)):
                noses.append(pose_landmarks_list[person][0].x)
            noses.sort()
            for person in range(len(pose_landmarks_list)):
                pose_landmarks = landmark_pb2.NormalizedLandmarkList() # don't worry too much about these two lines
                pose_landmarks.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z) for landmark in pose_landmarks_list[person]
                ])
                if len(noses)==1 or len(noses)==0:
                    continue
                if (pose_landmarks_list[person][0].x == noses[LR]):
                    hip = pose_landmarks_list[person][23]
                    ankle = pose_landmarks_list[person][27]
                    knee = pose_landmarks_list[person][25]
                    angle=calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
                    if 90<angle:
                        feedback.append("You did a lunge")
                    arm_y=pose_landmarks_list[person][16].y
                    if arm_y-after_y>0.1:
                        feedback.append("You did a parry")
                    after_y=arm_y
                    knee_1=pose_landmarks_list[person][26].x
                    knee_2=pose_landmarks_list[person][25].x
                    if knee_1<knee_2 and pre_1>pre_2:
                        feedback.append("You did a flech")
#needs fixing           continue
                    if knee_1>knee_2 and pre_1<pre_2:
                        feedback.append("You did a flech")
                    pre_1=knee_1
                    pre_2=knee_2
#needs fixing       continue
                    mpDraw.draw_landmarks(frame, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, mp.solutions.drawing_styles.get_default_pose_landmarks_style())
                    print(pose_landmarks_list[person][1].x)
                    hipa = pose_landmarks_list[person][23]
                    anklea = pose_landmarks_list[person][27]
                    kneea = pose_landmarks_list[person][25]
                    anglea=calculate_angle((hipa.x, hipa.y), (kneea.x, kneea.y), (anklea.x, anklea.y))
                    if 90<anglea:
                        feedback.append("Your oponent lunged")
                    arm_ya=pose_landmarks_list[person][16].y
                    if arm_ya-after_ya>0.1:
                        feedback.append("Your oponent parried")
                    after_ya=arm_ya
                    knee_1a=pose_landmarks_list[person][26].x
                    knee_2a=pose_landmarks_list[person][25].x
                    if knee_1a<knee_2a and pre_1a>pre_2a:
                        feedback.append("Your oponent did a flech")
#needs fixing           continue
                    if knee_1a>knee_2a and pre_1a<pre_2a:
                        feedback.append("Your oponent did a flech")
                    pre_1a=knee_1a
                    pre_2a=knee_2a
#needs fixing       continue
                    mpDraw.draw_landmarks(frame, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, mp.solutions.drawing_styles.get_default_pose_landmarks_style())
                    print(pose_landmarks_list[person][1].x)
            output.write(frame)
        else:
            print("Stream disconnected")
            break
    cap.release()
    cv2.destroyAllWindows()
    return feedback
if __name__ == '__main__':
  name.run(host='0.0.0.0')