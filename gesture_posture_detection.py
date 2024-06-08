# import cv2
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose
# mp_holistic = mp.solutions.holistic

# # Define gesture recognition function (replace with your actual gesture recognition logic)
# def recognize_gesture(landmarks):
#     # Thumbs Up
#     if (landmarks[mp_holistic.HandLandmark.THUMB_TIP].y < landmarks[mp_holistic.HandLandmark.WRIST].y and
#         landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.THUMB_TIP].y):
#         return "Thumbs Up"
    
#     # Hand Waving (movement of hand)
#     elif landmarks[mp_holistic.HandLandmark.WRIST].visibility < 0.9:
#         return "Hand Waving"
    
#     # Peace Sign (index and middle fingers spread)
#     elif (landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y and
#           landmarks[mp_holistic.HandLandmark.THUMB_TIP].y > landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y):
#         return "Peace Sign"
    
#     # Pointing (index finger extended)
#     elif landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].visibility > 0.9:
#         return "Pointing"
    
#     # Counting fingers
#     else:
#         # Count number of visible finger tips
#         visible_fingers = sum(1 for lm in landmarks if lm.visibility > 0.9)
#         return f"{visible_fingers} Finger(s)"

# # Define posture recognition function
# def recognize_posture(pose_landmarks):
#     # For simplicity, we’ll check the position of the hips to determine sitting or standing
#     # If the y-coordinate of the hip landmarks is above a certain threshold, consider it standing; otherwise, sitting
#     hip_landmarks = [pose_landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]]
#     if all(hip.y > 0.7 for hip in hip_landmarks):
#         return "Sitting"
#     else:
#         return "Standing"

# # Start capturing video from webcam
# cap = cv2.VideoCapture(0)

# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
#      mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue

#         # Process image and find landmarks
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         pose_results = pose.process(image)
#         holistic_results = holistic.process(image)

#         # Draw pose and hand landmarks on the image
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         if pose_results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 pose_results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#             )
        
#         if holistic_results.left_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 holistic_results.left_hand_landmarks,
#                 mp_holistic.HAND_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
#             )
        
#         if holistic_results.right_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 holistic_results.right_hand_landmarks,
#                 mp_holistic.HAND_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
#             )

#         # Recognize gesture (replace with your actual gesture recognition logic)
#         if holistic_results.right_hand_landmarks:
#             gesture = recognize_gesture(holistic_results.right_hand_landmarks.landmark)
#             cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         # Recognize posture
#         if pose_results.pose_landmarks:
#             posture = recognize_posture(pose_results.pose_landmarks)
#             cv2.putText(image, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         # Display the image
#         cv2.imshow('Gesture, Posture and Pose Recognition', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Define gesture recognition function
def recognize_gesture(landmarks):
    thumb_tip = landmarks[mp_holistic.HandLandmark.THUMB_TIP]
    wrist = landmarks[mp_holistic.HandLandmark.WRIST]
    index_finger_tip = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_holistic.HandLandmark.PINKY_TIP]

    # Check if thumb is up and other fingers are down (Thumbs Up)
    if (thumb_tip.y < wrist.y and
        index_finger_tip.y > thumb_tip.y and
        middle_finger_tip.y > thumb_tip.y and
        ring_finger_tip.y > thumb_tip.y and
        pinky_tip.y > thumb_tip.y):
        return "Thumbs Up"
    
    # Check if index and middle fingers are up and the other fingers are down (Peace Sign)
    elif (index_finger_tip.y < wrist.y and
          middle_finger_tip.y < wrist.y and
          ring_finger_tip.y > index_finger_tip.y and
          pinky_tip.y > middle_finger_tip.y):
        return "Peace Sign"

    # Hand Waving (movement of hand)
    elif wrist.visibility < 0.9:
        return "Hand Waving"
    
    # Pointing (index finger extended)
    elif index_finger_tip.visibility > 0.9:
        return "Pointing"
    
    # Counting fingers
    else:
        # Count number of visible finger tips
        visible_fingers = sum(1 for lm in [thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip] if lm.visibility > 0.9)
        return f"{visible_fingers} Finger(s)"

# Define posture recognition function
def recognize_posture(pose_landmarks):
    # For simplicity, we’ll check the position of the hips to determine sitting or standing
    # If the y-coordinate of the hip landmarks is above a certain threshold, consider it standing; otherwise, sitting
    hip_landmarks = [pose_landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]]
    if all(hip.y > 0.7 for hip in hip_landmarks):
        return "Sitting"
    else:
        return "Standing"

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process image and find landmarks
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image)
        holistic_results = holistic.process(image)

        # Draw pose and hand landmarks on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if holistic_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                holistic_results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        if holistic_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                holistic_results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

        # Recognize gesture
        if holistic_results.right_hand_landmarks:
            gesture = recognize_gesture(holistic_results.right_hand_landmarks.landmark)
            cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Recognize posture
        if pose_results.pose_landmarks:
            posture = recognize_posture(pose_results.pose_landmarks)
            cv2.putText(image, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Gesture, Posture and Pose Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
