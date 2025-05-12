import cv2
from ultralytics import YOLO
import time
import os
import numpy as np

# def detect_motion_and_check_for_cyclist_colour_focus(lower_color, upper_color, motion_threshold=120, min_motion_area=1000, capture_path="captured_images"):
#     """
#     Detects motion using a USB camera, captures a photo upon detection,
#     and then checks if a cyclist is present in the photo using YOLOv8.
#
#     Args:
#         lower_color (np.ndarray): Lower HSV color for the bike lane.
#         upper_color (np.ndarray): Upper HSV color for the bike lane.
#         motion_threshold (int): Threshold for pixel difference to consider motion.
#         min_motion_area (int): Minimum area of motion contour to trigger capture.
#         capture_path (str): Directory to save captured images.
#     """
#
#     # Create the capture directory if it doesn't exist
#     os.makedirs(capture_path, exist_ok=True)
#
#     # Open the default camera
#     cap = cv2.VideoCapture(0)
#     counted_cyclists = set()
#     if not cap.isOpened():
#         print("Cannot open camera")
#         return
#
#     # Load a pre-trained YOLOv8 model (you can use your custom trained one)
#     model = YOLO('cyclist_detect.pt')  # Or replace with 'path/to/your/best.pt'
#
#     # Define the Region of Interest (ROI) for the bike lane
#     # These coordinates should roughly outline the bike lane in your camera view
#     roi_points = np.array([[100, 400],
#                            [150, 300],
#                            [850, 350],
#                            [900, 450]], np.int32)
#     roi_points = roi_points.reshape((-1, 1, 2))
#     roi_color = (255, 0, 0)  # Blue color for the ROI
#     roi_thickness = 2
#
#     # Read the first frame for background differencing
#     ret, background_frame = cap.read()
#     if not ret:
#         print("Cannot read first frame")
#         cap.release()
#         return
#     background_hsv = cv2.cvtColor(background_frame, cv2.COLOR_BGR2HSV)
#     background_mask = cv2.inRange(background_hsv, lower_color, upper_color)
#     background_gray_masked = cv2.cvtColor(cv2.bitwise_and(background_frame, background_frame, mask=background_mask),
#                                           cv2.COLOR_BGR2GRAY)
#     background_gray_masked = cv2.GaussianBlur(background_gray_masked, (11, 11), 0)
#
#     motion_detected = False
#     detected_cyclist_id = set()
#     next_cyclist_id = 1
#     cyclist_count = 0
#     print("Motion detection and cyclist check started. Press 'q' to quit.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#
#         # Draw the ROI rectangle/polygon
#         cv2.polylines(frame, [roi_points], True, roi_color, roi_thickness)
#
#         hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         color_mask = cv2.inRange(hsv_frame, lower_color, upper_color)
#         masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
#         gray_frame_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
#         gray_frame_masked = cv2.GaussianBlur(gray_frame_masked, (11, 11), 0)
#
#         cv2.imshow("Motion Detection", frame) # Show the frame with the ROI
#         cv2.imshow("Color Mask", color_mask)
#
#         frame_diff = cv2.absdiff(background_gray_masked, gray_frame_masked)
#         thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         motion_detected = False
#         for contour in contours:
#             if cv2.contourArea(contour) > min_motion_area:
#                 # Check if the motion is within the ROI (optional but recommended)
#                 mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
#                 cv2.drawContours(mask_roi, [roi_points], -1, 255, -1)
#                 motion_centroid = np.mean(cv2.moments(contour)['m01'] / cv2.moments(contour)['m00']) if cv2.moments(contour)['m00'] != 0 else None
#                 if motion_centroid is not None:
#                     # You'd need to check if this centroid is within the ROI mask
#                     pass # Implementation needed
#
#                 motion_detected = True
#                 break
#
#         if motion_detected:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             image_name = os.path.join(capture_path, f"motion_{timestamp}.jpg")
#             cv2.imwrite(image_name, frame)
#             print(f"Motion detected in ROI! Photo saved as: {image_name}")
#
#             # Perform cyclist detection on the captured image
#             results = model(image_name)
#             cyclists_in_photo = False
#             current_photo_cyclist_ids = set()
#
#             for result in results:
#                 for box in result.boxes:
#                     class_id = int(box.cls[0])
#                     class_name = result.names[class_id]
#                     if class_name == 'bicycle' or class_name == 'person':  # Adjust based on your model's classes
#                         print("Cyclist detected in the photo!")
#                         cyclists_in_photo = True
#                         new_id = f"cyclist_{next_cyclist_id}_{timestamp}"
#                         detected_cyclist_id.add(new_id)
#                         current_photo_cyclist_ids.add(new_id)
#                         next_cyclist_id += 1
#                         cyclist_count += 1
#                         print(cyclist_count)
#                         # You could add further actions here, like saving cyclist detection info
#
#             # Reset motion_detected to avoid capturing multiple photos for the same motion
#             motion_detected = False
#             time.sleep(5)  # Add a small delay before looking for motion again
#
#
#         # Update background periodically (optional)
#         if cv2.waitKey(1) & 0xFF == ord('u'):
#             ret, background_frame = cap.read()
#             if ret:
#                 background_hsv = cv2.cvtColor(background_frame, cv2.COLOR_BGR2HSV)
#                 background_mask = cv2.inRange(background_hsv, lower_color, upper_color)
#                 background_gray_masked = cv2.cvtColor(
#                     cv2.bitwise_and(background_frame, background_frame, mask=background_mask), cv2.COLOR_BGR2GRAY)
#                 background_gray_masked = cv2.GaussianBlur(background_gray_masked, (11, 11), 0)
#                 print("Background updated (color focused).")
#
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print('There were {} cyclists detected'.format(cyclist_count))
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     # Define the HSV color range for your red/brown bike lane
#     lower_red_brown = np.array([44, 59, 0])  # Example: Broader lower bound for brown
#     upper_red_brown = np.array([179, 88, 191])  # Example: Broader upper bound for
#
#     detect_motion_and_check_for_cyclist_colour_focus(lower_red_brown, upper_red_brown)


def detect_motion_and_check_for_cyclist_colour_focus(lower_color, upper_color, motion_threshold=120, min_motion_area=1000, capture_path="captured_images"):
    """
    Detects motion within a defined ROI using a USB camera, captures a photo upon detection,
    and then checks if a cyclist is present in the photo using YOLOv8.

    Args:
        lower_color (np.ndarray): Lower HSV color for the bike lane (currently not directly used for motion).
        upper_color (np.ndarray): Upper HSV color for the bike lane (currently not directly used for motion).
        motion_threshold (int): Threshold for pixel difference to consider motion.
        min_motion_area (int): Minimum area of motion contour to trigger capture.
        capture_path (str): Directory to save captured images.
    """

    # Create the capture directory if it doesn't exist
    os.makedirs(capture_path, exist_ok=True)

    # Open the default camera
    cap = cv2.VideoCapture(0)
    counted_cyclists = set()
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Load a pre-trained YOLOv8 model
    model = YOLO('cyclist_detect.pt')

    # Define the Region of Interest (ROI) for the bike lane
    roi_points = np.array([[100, 400], [150, 300], [850, 350], [900, 450]], np.int32)
    roi_points = roi_points.reshape((-1, 1, 2))
    roi_color = (255, 0, 0)  # Blue color for the ROI
    roi_thickness = 2

    # Create a mask for the ROI
    mask_roi = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)
    cv2.drawContours(mask_roi, [roi_points], -1, 255, -1)

    # Read the first frame for background differencing *within the ROI*
    ret, background_frame = cap.read()
    if not ret:
        print("Cannot read first frame")
        cap.release()
        return
    background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    background_gray_masked = cv2.bitwise_and(background_gray, background_gray, mask=mask_roi)
    background_gray_masked = cv2.GaussianBlur(background_gray_masked, (11, 11), 0)

    motion_detected = False
    detected_cyclist_id = set()
    next_cyclist_id = 1
    cyclist_count = 0
    print("Motion detection within ROI and cyclist check started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Draw the ROI
        cv2.polylines(frame, [roi_points], True, roi_color, roi_thickness)

        # Convert current frame to grayscale and mask with ROI
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_masked = cv2.bitwise_and(gray_frame, gray_frame, mask=mask_roi)
        gray_frame_masked = cv2.GaussianBlur(gray_frame_masked, (11, 11), 0)

        frame_diff = cv2.absdiff(background_gray_masked, gray_frame_masked)
        thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > min_motion_area:
                # Check if the centroid of the motion is within the ROI
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if mask_roi[cY, cX] == 255:  # Check if centroid is within the white ROI mask
                        motion_detected = True
                        break

        if motion_detected:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = os.path.join(capture_path, f"motion_roi_{timestamp}.jpg")
            cv2.imwrite(image_name, frame)
            print(f"Motion detected within ROI! Photo saved as: {image_name}")

            # Perform cyclist detection on the captured image
            results = model(image_name)
            cyclists_in_photo = False
            current_photo_cyclist_ids = set()

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    if class_name == 'bicycle' or class_name == 'person':  # Adjust based on your model's classes
                        print("Cyclist detected in the photo!")
                        cyclists_in_photo = True
                        new_id = f"cyclist_{next_cyclist_id}_{timestamp}"
                        detected_cyclist_id.add(new_id)
                        current_photo_cyclist_ids.add(new_id)
                        next_cyclist_id += 1
                        cyclist_count += 1
                        print(cyclist_count)

            motion_detected = False
            time.sleep(5)

        cv2.imshow("Motion Detection", frame)
        cv2.imshow("Motion Threshold", thresh) # To visualize the motion

        if cv2.waitKey(1) & 0xFF == ord('u'):
            ret, background_frame = cap.read()
            if ret:
                background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
                background_gray_masked = cv2.bitwise_and(background_gray, background_gray, mask=mask_roi)
                background_gray_masked = cv2.GaussianBlur(background_gray_masked, (11, 11), 0)
                print("Background updated (within ROI).")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f'There were {cyclist_count} cyclists detected.')
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lower_red_brown = np.array([44, 59, 0])
    upper_red_brown = np.array([179, 88, 191])

    detect_motion_and_check_for_cyclist_colour_focus(lower_red_brown, upper_red_brown)
