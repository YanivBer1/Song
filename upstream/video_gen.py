import cv2
import mediapipe as mp
import numpy as np

def extract_keypoints_from_video(video_path, verbose=False):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose_model = mp_pose.Pose()
    hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Pose estimation
        pose_results = pose_model.process(frame_rgb)
        if not pose_results.pose_landmarks:
            frame_idx += 1
            continue

        # Extract 33 body keypoints
        pose_landmarks = pose_results.pose_landmarks.landmark
        pose = np.array([[lm.x, lm.y] for lm in pose_landmarks])  # shape (33, 2)

        # Hand tracking
        left_hand = np.zeros((21, 2))
        right_hand = np.zeros((21, 2))

        hand_results = hands_model.process(frame_rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, hand_info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label = hand_info.classification[0].label  # 'Left' or 'Right'
                hand_array = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                if label == "Left":
                    left_hand = hand_array
                else:
                    right_hand = hand_array

        keypoints_sequence.append((pose, left_hand, right_hand))

        if verbose:
            print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()
    pose_model.close()
    hands_model.close()

    return keypoints_sequence

def render_person(frame, pose, left_hand, right_hand):
    h, w = frame.shape[:2]
    
    # Define MediaPipe Pose keypoint indices
    # Face
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    
    # Body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Define hand keypoint indices for MediaPipe Hands
    # Thumb: 0-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    # Define finger connections
    finger_connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # Enhanced friendly color palette
    skin_color = (173, 216, 230)  # Light brown bear skin
    outline_color = (40, 40, 40)  # Softer outline
    shirt_color = (205, 170, 125) # Light blue tuxedo jacket

    # shirt_color = (205, 170, 125)
    # skin_color = (173, 216, 230)


    pants_color = (135, 206, 235) # Slightly darker light blue tuxedo pants
    bow_tie_color = (255, 255, 255)  # White bow tie
    eye_color = (255, 255, 255)   # White eyes
    pupil_color = (0, 0, 0)       # Black pupils
    
    # Draw body parts as filled shapes
    
    # 1. Head (face) with enhanced friendly styling
    if len(pose) > max(LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR):
        # Calculate head center and size
        head_center_x = pose[NOSE][0] * w
        head_center_y = pose[NOSE][1] * h
        
        # Estimate head size based on face keypoints
        if pose[LEFT_EYE][0] > 0 and pose[RIGHT_EYE][0] > 0:
            eye_distance = abs(pose[LEFT_EYE][0] - pose[RIGHT_EYE][0]) * w
            head_radius = eye_distance * 1.8  # Larger head for friendlier look
        else:
            head_radius = 35
        
        # Draw bear ears first (behind the head)
        ear_radius = int(head_radius * 0.4)
        # Left ear
        left_ear_x = int(head_center_x - head_radius * 0.6)
        left_ear_y = int(head_center_y - head_radius * 0.8)
        cv2.circle(frame, (left_ear_x, left_ear_y), ear_radius, skin_color, -1)
        cv2.circle(frame, (left_ear_x, left_ear_y), ear_radius, outline_color, 2)
        # Inner ear detail
        cv2.circle(frame, (left_ear_x, left_ear_y), int(ear_radius * 0.6), (120, 160, 180), -1)
        
        # Right ear
        right_ear_x = int(head_center_x + head_radius * 0.6)
        right_ear_y = int(head_center_y - head_radius * 0.8)
        cv2.circle(frame, (right_ear_x, right_ear_y), ear_radius, skin_color, -1)
        cv2.circle(frame, (right_ear_x, right_ear_y), ear_radius, outline_color, 2)
        # Inner ear detail
        cv2.circle(frame, (right_ear_x, right_ear_y), int(ear_radius * 0.6), (120, 160, 180), -1)
        
        # Draw head with skin color
        cv2.circle(frame, (int(head_center_x), int(head_center_y)), int(head_radius), skin_color, -1)
        cv2.circle(frame, (int(head_center_x), int(head_center_y)), int(head_radius), outline_color, 2)
        
        # Draw larger, cuter bear eyes
        if pose[LEFT_EYE][0] > 0 and pose[LEFT_EYE][1] > 0:
            eye_x, eye_y = int(pose[LEFT_EYE][0] * w), int(pose[LEFT_EYE][1] * h)
            # Larger white eye
            cv2.circle(frame, (eye_x, eye_y), 10, eye_color, -1)
            # Larger pupil
            cv2.circle(frame, (eye_x, eye_y), 6, pupil_color, -1)
            # Eye outline
            cv2.circle(frame, (eye_x, eye_y), 10, outline_color, 1)
            # Eye shine
            cv2.circle(frame, (eye_x-3, eye_y-3), 3, (255, 255, 255), -1)
            
        if pose[RIGHT_EYE][0] > 0 and pose[RIGHT_EYE][1] > 0:
            eye_x, eye_y = int(pose[RIGHT_EYE][0] * w), int(pose[RIGHT_EYE][1] * h)
            # Larger white eye
            cv2.circle(frame, (eye_x, eye_y), 10, eye_color, -1)
            # Larger pupil
            cv2.circle(frame, (eye_x, eye_y), 6, pupil_color, -1)
            # Eye outline
            cv2.circle(frame, (eye_x, eye_y), 10, outline_color, 1)
            # Eye shine
            cv2.circle(frame, (eye_x-3, eye_y-3), 3, (255, 255, 255), -1)
        
        # Draw cute bear nose
        nose_x = int(head_center_x)
        nose_y = int(head_center_y + head_radius * 0.1)
        # Draw a cute round nose
        cv2.circle(frame, (nose_x, nose_y), 6, (80, 40, 20), -1)  # Dark brown nose
        cv2.circle(frame, (nose_x, nose_y), 6, outline_color, 1)
        
        # Draw friendly smile
        smile_center_x = int(head_center_x)
        smile_center_y = int(head_center_y + head_radius * 0.3)
        smile_radius = int(head_radius * 0.6)
        # Draw smile arc
        cv2.ellipse(frame, (smile_center_x, smile_center_y), (smile_radius, smile_radius//2), 
                   0, 0, 180, outline_color, 3)
    
    # 2. Torso with nice shirt
    if len(pose) > max(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        # Calculate torso points
        left_shoulder = (int(pose[LEFT_SHOULDER][0] * w), int(pose[LEFT_SHOULDER][1] * h))
        right_shoulder = (int(pose[RIGHT_SHOULDER][0] * w), int(pose[RIGHT_SHOULDER][1] * h))
        left_hip = (int(pose[LEFT_HIP][0] * w), int(pose[LEFT_HIP][1] * h))
        right_hip = (int(pose[RIGHT_HIP][0] * w), int(pose[RIGHT_HIP][1] * h))
        
        # Draw torso as a filled polygon with nice shirt color
        torso_points = np.array([left_shoulder, right_shoulder, right_hip, left_hip], np.int32)
        cv2.fillPoly(frame, [torso_points], shirt_color)
        cv2.polylines(frame, [torso_points], True, outline_color, 2)
    
    # 3. Arms with better proportions (non-stick)
    # Left arm
    if len(pose) > max(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST):
        if pose[LEFT_SHOULDER][0] > 0 and pose[LEFT_ELBOW][0] > 0:
            # Upper arm - 3x thicker and more natural
            cv2.line(frame, 
                     (int(pose[LEFT_SHOULDER][0] * w), int(pose[LEFT_SHOULDER][1] * h)),
                     (int(pose[LEFT_ELBOW][0] * w), int(pose[LEFT_ELBOW][1] * h)),
                     skin_color, 36)
            cv2.line(frame, 
                     (int(pose[LEFT_SHOULDER][0] * w), int(pose[LEFT_SHOULDER][1] * h)),
                     (int(pose[LEFT_ELBOW][0] * w), int(pose[LEFT_ELBOW][1] * h)),
                     outline_color, 2)
            
            # Lower arm
            if pose[LEFT_WRIST][0] > 0:
                cv2.line(frame, 
                         (int(pose[LEFT_ELBOW][0] * w), int(pose[LEFT_ELBOW][1] * h)),
                         (int(pose[LEFT_WRIST][0] * w), int(pose[LEFT_WRIST][1] * h)),
                         skin_color, 30)
                cv2.line(frame, 
                         (int(pose[LEFT_ELBOW][0] * w), int(pose[LEFT_ELBOW][1] * h)),
                         (int(pose[LEFT_WRIST][0] * w), int(pose[LEFT_WRIST][1] * h)),
                         outline_color, 2)
    
    # Right arm
    if len(pose) > max(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST):
        if pose[RIGHT_SHOULDER][0] > 0 and pose[RIGHT_ELBOW][0] > 0:
            # Upper arm - 3x thicker and more natural
            cv2.line(frame, 
                     (int(pose[RIGHT_SHOULDER][0] * w), int(pose[RIGHT_SHOULDER][1] * h)),
                     (int(pose[RIGHT_ELBOW][0] * w), int(pose[RIGHT_ELBOW][1] * h)),
                     skin_color, 36)
            cv2.line(frame, 
                     (int(pose[RIGHT_SHOULDER][0] * w), int(pose[RIGHT_SHOULDER][1] * h)),
                     (int(pose[RIGHT_ELBOW][0] * w), int(pose[RIGHT_ELBOW][1] * h)),
                     outline_color, 2)
            
            # Lower arm
            if pose[RIGHT_WRIST][0] > 0:
                cv2.line(frame, 
                         (int(pose[RIGHT_ELBOW][0] * w), int(pose[RIGHT_ELBOW][1] * h)),
                         (int(pose[RIGHT_WRIST][0] * w), int(pose[RIGHT_WRIST][1] * h)),
                         skin_color, 30)
                cv2.line(frame, 
                         (int(pose[RIGHT_ELBOW][0] * w), int(pose[RIGHT_ELBOW][1] * h)),
                         (int(pose[RIGHT_WRIST][0] * w), int(pose[RIGHT_WRIST][1] * h)),
                         outline_color, 2)
    
    # 4. Legs with nice pants
    # Left leg
    if len(pose) > max(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE):
        if pose[LEFT_HIP][0] > 0 and pose[LEFT_KNEE][0] > 0:
            # Upper leg
            cv2.line(frame, 
                     (int(pose[LEFT_HIP][0] * w), int(pose[LEFT_HIP][1] * h)),
                     (int(pose[LEFT_KNEE][0] * w), int(pose[LEFT_KNEE][1] * h)),
                     pants_color, 14)
            cv2.line(frame, 
                     (int(pose[LEFT_HIP][0] * w), int(pose[LEFT_HIP][1] * h)),
                     (int(pose[LEFT_KNEE][0] * w), int(pose[LEFT_KNEE][1] * h)),
                     outline_color, 2)
            
            # Lower leg
            if pose[LEFT_ANKLE][0] > 0:
                cv2.line(frame, 
                         (int(pose[LEFT_KNEE][0] * w), int(pose[LEFT_KNEE][1] * h)),
                         (int(pose[LEFT_ANKLE][0] * w), int(pose[LEFT_ANKLE][1] * h)),
                         pants_color, 12)
                cv2.line(frame, 
                         (int(pose[LEFT_KNEE][0] * w), int(pose[LEFT_KNEE][1] * h)),
                         (int(pose[LEFT_ANKLE][0] * w), int(pose[LEFT_ANKLE][1] * h)),
                         outline_color, 2)
    
    # Right leg
    if len(pose) > max(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE):
        if pose[RIGHT_HIP][0] > 0 and pose[RIGHT_KNEE][0] > 0:
            # Upper leg
            cv2.line(frame, 
                     (int(pose[RIGHT_HIP][0] * w), int(pose[RIGHT_HIP][1] * h)),
                     (int(pose[RIGHT_KNEE][0] * w), int(pose[RIGHT_KNEE][1] * h)),
                     pants_color, 14)
            cv2.line(frame, 
                     (int(pose[RIGHT_HIP][0] * w), int(pose[RIGHT_HIP][1] * h)),
                     (int(pose[RIGHT_KNEE][0] * w), int(pose[RIGHT_KNEE][1] * h)),
                     outline_color, 2)
            
            # Lower leg
            if pose[RIGHT_ANKLE][0] > 0:
                cv2.line(frame, 
                         (int(pose[RIGHT_KNEE][0] * w), int(pose[RIGHT_KNEE][1] * h)),
                         (int(pose[RIGHT_ANKLE][0] * w), int(pose[RIGHT_ANKLE][1] * h)),
                         pants_color, 12)
                cv2.line(frame, 
                         (int(pose[RIGHT_KNEE][0] * w), int(pose[RIGHT_KNEE][1] * h)),
                         (int(pose[RIGHT_ANKLE][0] * w), int(pose[RIGHT_ANKLE][1] * h)),
                         outline_color, 2)
    
    # 1.5. Neck connecting head to shoulders
    if len(pose) > max(LEFT_SHOULDER, RIGHT_SHOULDER):
        if pose[LEFT_SHOULDER][0] > 0 and pose[RIGHT_SHOULDER][0] > 0:
            # Calculate neck position and size
            neck_center_x = (pose[LEFT_SHOULDER][0] + pose[RIGHT_SHOULDER][0]) / 2 * w
            neck_center_y = (pose[LEFT_SHOULDER][1] + pose[RIGHT_SHOULDER][1]) / 2 * h
            
            # Position neck slightly above shoulders
            neck_y = neck_center_y - 15
            
            # Calculate neck width based on shoulder distance
            shoulder_distance = abs(pose[LEFT_SHOULDER][0] - pose[RIGHT_SHOULDER][0]) * w
            neck_width = shoulder_distance * 0.3  # Neck is about 30% of shoulder width
            neck_height = 25
            
            # Draw neck as a filled rectangle with rounded corners effect
            neck_left = int(neck_center_x - neck_width / 2)
            neck_right = int(neck_center_x + neck_width / 2)
            neck_top = int(neck_y - neck_height / 2)
            neck_bottom = int(neck_y + neck_height / 2)
            
            # Draw neck with skin color
            cv2.rectangle(frame, (neck_left, neck_top), (neck_right, neck_bottom), skin_color, -1)
            cv2.rectangle(frame, (neck_left, neck_top), (neck_right, neck_bottom), outline_color, 2)
            
            # Draw bow tie
            bow_center_x = int(neck_center_x)
            bow_center_y = int(neck_y + neck_height / 2 + 5)
            bow_width = 20
            bow_height = 12
            
            # Draw left side of bow tie
            left_bow_points = np.array([
                [bow_center_x - bow_width//2, bow_center_y - bow_height//2],
                [bow_center_x - bow_width//2 - 8, bow_center_y],
                [bow_center_x - bow_width//2, bow_center_y + bow_height//2],
                [bow_center_x - 2, bow_center_y + bow_height//2],
                [bow_center_x - 2, bow_center_y - bow_height//2]
            ], np.int32)
            cv2.fillPoly(frame, [left_bow_points], bow_tie_color)
            cv2.polylines(frame, [left_bow_points], True, outline_color, 1)
            
            # Draw right side of bow tie
            right_bow_points = np.array([
                [bow_center_x + bow_width//2, bow_center_y - bow_height//2],
                [bow_center_x + bow_width//2 + 8, bow_center_y],
                [bow_center_x + bow_width//2, bow_center_y + bow_height//2],
                [bow_center_x + 2, bow_center_y + bow_height//2],
                [bow_center_x + 2, bow_center_y - bow_height//2]
            ], np.int32)
            cv2.fillPoly(frame, [right_bow_points], bow_tie_color)
            cv2.polylines(frame, [right_bow_points], True, outline_color, 1)
            
            # Draw center knot of bow tie
            knot_points = np.array([
                [bow_center_x - 2, bow_center_y - 3],
                [bow_center_x + 2, bow_center_y - 3],
                [bow_center_x + 2, bow_center_y + 3],
                [bow_center_x - 2, bow_center_y + 3]
            ], np.int32)
            cv2.fillPoly(frame, [knot_points], bow_tie_color)
            cv2.polylines(frame, [knot_points], True, outline_color, 1)
    
    # 5. Enhanced Hands with clear finger definition (drawn last to ensure they're always in front)
    for hand, hand_color in [(left_hand, (255, 0, 0)), (right_hand, (0, 0, 255))]:
        if np.any(hand != 0):  # Only draw if hand is detected
            # Draw hand palm as a filled shape
            palm_points = []
            # Use wrist and base of fingers for palm
            palm_indices = [0, 5, 9, 13, 17]  # Wrist and base of each finger
            for idx in palm_indices:
                if idx < len(hand) and hand[idx][0] > 0 and hand[idx][1] > 0:
                    palm_points.append([int(hand[idx][0] * w), int(hand[idx][1] * h)])
            
            if len(palm_points) > 3:
                palm_points = np.array(palm_points, np.int32)
                hull = cv2.convexHull(palm_points)
                cv2.fillPoly(frame, [hull], (255, 182, 193))  # Light pink color for palm
                cv2.polylines(frame, [hull], True, outline_color, 2)
            
            # Draw individual fingers with clear connections
            for connection in finger_connections:
                start_idx, end_idx = connection
                if (start_idx < len(hand) and end_idx < len(hand) and 
                    hand[start_idx][0] > 0 and hand[start_idx][1] > 0 and
                    hand[end_idx][0] > 0 and hand[end_idx][1] > 0):
                    
                    start_point = (int(hand[start_idx][0] * w), int(hand[start_idx][1] * h))
                    end_point = (int(hand[end_idx][0] * w), int(hand[end_idx][1] * h))
                    
                    # Draw finger bone
                    cv2.line(frame, start_point, end_point, (255, 182, 193), 9)  # Light pink color for finger bones
                    cv2.line(frame, start_point, end_point, outline_color, 1)
            
            # Draw finger tips with emphasis
            finger_tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            for tip_idx in finger_tips:
                if tip_idx < len(hand) and hand[tip_idx][0] > 0 and hand[tip_idx][1] > 0:
                    tip_x, tip_y = int(hand[tip_idx][0] * w), int(hand[tip_idx][1] * h)
                    # Draw larger, more visible finger tips
                    cv2.circle(frame, (tip_x, tip_y), 4, (255, 182, 193), -1)  # Light pink color for finger tips
                    cv2.circle(frame, (tip_x, tip_y), 4, outline_color, 2)
                    # Add a small highlight
                    cv2.circle(frame, (tip_x-1, tip_y-1), 1, (255, 255, 255), -1)
            
            # Draw all hand keypoints for clarity (keeping original red/blue colors for dots)
            for i, (x, y) in enumerate(hand):
                if x > 0 and y > 0:
                    point_x, point_y = int(x * w), int(y * h)
                    # Different colors for different parts of the hand
                    if i in finger_tips:
                        cv2.circle(frame, (point_x, point_y), 2, hand_color, -1)
                    else:
                        cv2.circle(frame, (point_x, point_y), 1, hand_color, -1)
    
    return frame

def interpolate_keypoints(kptsA, kptsB, steps):
    poseA, leftA, rightA = kptsA
    poseB, leftB, rightB = kptsB

    frames = []
    for t in range(1, steps + 1):
        alpha = t / (steps + 1)
        interp_pose = (1 - alpha) * poseA + alpha * poseB
        
        # Check if hands are detected (non-zero coordinates)
        leftA_detected = np.any(leftA != 0)
        rightA_detected = np.any(rightA != 0)
        leftB_detected = np.any(leftB != 0)
        rightB_detected = np.any(rightB != 0)
        
        # Interpolate left hand only if both frames have detected hands
        if leftA_detected and leftB_detected:
            print("leftA_detected and leftB_detected")
            interp_left = (1 - alpha) * leftA + alpha * leftB
        elif leftA_detected:
            interp_left = leftA  # Keep the last known position
        elif leftB_detected:
            interp_left = leftB  # Use the new position
        else:
            interp_left = np.zeros((21, 2))  # No hands detected
        
        # Interpolate right hand only if both frames have detected hands
        if rightA_detected and rightB_detected:
            print("rightA_detected and rightB_detected")
            interp_right = (1 - alpha) * rightA + alpha * rightB
        elif rightA_detected:
            interp_right = rightA  # Keep the last known position
        elif rightB_detected:
            interp_right = rightB  # Use the new position
        else:
            interp_right = np.zeros((21, 2))  # No hands detected
        
        frames.append((interp_pose, interp_left, interp_right))
    return frames

def get_video_writer(output_path, fps=30.0, width=1280, height=720):
    """
    Create a video writer with H.264 codec for better browser compatibility.
    Falls back to other codecs if H.264 is not available.
    """
    # Try H.264 codec first (best for browser compatibility)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print("Using H.264 (avc1) codec for video encoding")
            return out
        else:
            out.release()
    except Exception as e:
        print(f"H.264 codec not available: {e}")
    
    # Fallback to MPEG-4
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print("Using MPEG-4 (mp4v) codec for video encoding")
            return out
        else:
            out.release()
    except Exception as e:
        print(f"MPEG-4 codec not available: {e}")
    
    # Final fallback to XVID
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print("Using XVID codec for video encoding")
            return out
        else:
            out.release()
    except Exception as e:
        print(f"XVID codec not available: {e}")
    
    raise RuntimeError("No suitable video codec found")

def create_stitched_video(videoA_path, videoB_path, output_path="stitched_output.mp4"):
    # Extract keypoints from both videos
    videoA_keypoints = extract_keypoints_from_video(videoA_path)
    videoB_keypoints = extract_keypoints_from_video(videoB_path)
    
    # Create video writer with H.264 codec for better browser compatibility
    out = get_video_writer(output_path, 30.0, 1280, 720)

    # Show original A
    for pose, l, r in videoA_keypoints:
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        out.write(render_person(frame, pose, l, r))

    # Interpolation
    interp = interpolate_keypoints(videoA_keypoints[-1], videoB_keypoints[0], steps=15)
    for pose, l, r in interp:
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        out.write(render_person(frame, pose, l, r))

    # Show original B
    for pose, l, r in videoB_keypoints:
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        out.write(render_person(frame, pose, l, r))

    out.release()
    print(f"Video saved to {output_path}")

def create_multi_stitched_video(video_paths, output_path="multi_stitched_output.mp4", transition_steps=15):
    """
    Create a stitched video from multiple video files.
    
    Args:
        video_paths (list): List of paths to MP4 video files
        output_path (str): Output path for the final video
        transition_steps (int): Number of frames for transitions between videos
    """
    if len(video_paths) < 2:
        print("Need at least 2 videos to stitch together!")
        return
    
    print(f"Processing {len(video_paths)} videos...")
    
    # Extract keypoints from all videos
    all_keypoints = []
    for i, video_path in enumerate(video_paths):
        print(f"Extracting keypoints from video {i+1}/{len(video_paths)}: {video_path}")
        keypoints = extract_keypoints_from_video(video_path)
        all_keypoints.append(keypoints)
        print(f"  - Extracted {len(keypoints)} frames")
    
    # Create video writer with H.264 codec for better browser compatibility
    out = get_video_writer(output_path, 30.0, 1280, 720)
    
    total_frames = 0
    
    # Process each video
    for i, keypoints in enumerate(all_keypoints):
        print(f"Rendering video {i+1}/{len(all_keypoints)}...")
        
        # Render all frames from current video
        for pose, l, r in keypoints:
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
            out.write(render_person(frame, pose, l, r))
            total_frames += 1
        
        # Add transition to next video (except for the last video)
        if i < len(all_keypoints) - 1:
            print(f"  Adding transition to next video...")
            next_keypoints = all_keypoints[i + 1]
            
            # Interpolate between last frame of current video and first frame of next video
            interp = interpolate_keypoints(keypoints[-1], next_keypoints[0], steps=transition_steps)
            for pose, l, r in interp:
                frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
                out.write(render_person(frame, pose, l, r))
                total_frames += 1
    
    out.release()
    print(f"Multi-stitched video saved to {output_path}")
    print(f"Total frames rendered: {total_frames}")
    print(f"Video duration: {total_frames/30:.2f} seconds")

if __name__ == "__main__":
    # Example usage for multiple videos
    video_list = [
        "/Users/ethantam/desktop/35304.mp4",
        "/Users/ethantam/desktop/23978.mp4",
        "/Users/ethantam/desktop/23106.mp4",
        # Add more video paths here as needed
    ]
    
    # Create multi-stitched video
    create_multi_stitched_video(video_list, "multi_stitched_output_1.mp4")
    
    # Or use the original 2-video function
    # create_stitched_video("/Users/ethantam/desktop/35304.mp4", "/Users/ethantam/desktop/23978.mp4") 