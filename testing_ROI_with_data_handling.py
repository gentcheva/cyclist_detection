# import cv2
# from ultralytics import YOLO
# import time
# import os
# import numpy as np
# import sqlite3
# import datetime
#
# # Define the database file name
# DATABASE_FILE = 'cyclist_data.db'
#
#
# # Define the database file name
# DATABASE_FILE = 'cyclist_data.db'
#
# def connect_db():
#     """Connects to the SQLite database."""
#     conn = sqlite3.connect(DATABASE_FILE)
#     cursor = conn.cursor()
#     return conn, cursor
#
# def create_table():
#     """Creates the cyclist_counts table if it doesn't exist."""
#     conn, cursor = connect_db()
#     cursor.execute("\n"
#                    "        CREATE TABLE IF NOT EXISTS cyclist_counts (\n"
#                    "            timestamp TEXT PRIMARY KEY,\n"
#                    "            count INTEGER NOT NULL\n"
#                    "        )\n"
#                    "    ")
#     conn.commit()
#     conn.close()
#
# def insert_count(timestamp_str, count):
#     """Inserts a specific timestamp and cyclist count into the database."""
#     conn, cursor = connect_db()
#     try:
#         cursor.execute("INSERT INTO cyclist_counts (timestamp, count) VALUES (?, ?)", (timestamp_str, count))
#         conn.commit()
#         print(f"Dummy data: {count} cyclists recorded at {timestamp_str}")
#     except sqlite3.IntegrityError:
#         print(f"Dummy data: Count already recorded for timestamp: {timestamp_str}")
#     finally:
#         conn.close()
#
# def get_all_counts():
#     """Retrieves all recorded counts from the database."""
#     conn, cursor = connect_db()
#     cursor.execute("SELECT * FROM cyclist_counts")
#     rows = cursor.fetchall()
#     conn.close()
#     return rows
#
# def get_counts_by_date(date_str):
#     """Retrieves counts for a specific date (YYYY-MM-DD)."""
#     conn, cursor = connect_db()
#     cursor.execute("SELECT * FROM cyclist_counts WHERE strftime('%Y-%m-%d', timestamp) = ?", (date_str,))
#     rows = cursor.fetchall()
#     conn.close()
#     return rows
#
#
#
#
# def detect_motion_and_check_for_cyclist_colour_focus(lower_color, upper_color, motion_threshold=120, min_motion_area=1000, capture_path="captured_images"):
#     """
#     Detects motion within a defined ROI using a USB camera, captures a photo upon detection,
#     and then checks if a cyclist is present in the photo using YOLOv8.
#
#     Args:
#         lower_color (np.ndarray): Lower HSV color for the bike lane (currently not directly used for motion).
#         upper_color (np.ndarray): Upper HSV color for the bike lane (currently not directly used for motion).
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
#     # Load a pre-trained YOLOv8 model
#     model = YOLO('cyclist_detect.pt')
#
#     # Define the Region of Interest (ROI) for the bike lane
#     roi_points = np.array([[100, 400], [150, 300], [850, 350], [900, 450]], np.int32)
#     roi_points = roi_points.reshape((-1, 1, 2))
#     roi_color = (255, 0, 0)  # Blue color for the ROI
#     roi_thickness = 2
#
#     # Create a mask for the ROI
#     mask_roi = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)
#     cv2.drawContours(mask_roi, [roi_points], -1, 255, -1)
#
#     # Read the first frame for background differencing *within the ROI*
#     ret, background_frame = cap.read()
#     if not ret:
#         print("Cannot read first frame")
#         cap.release()
#         return
#     background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
#     background_gray_masked = cv2.bitwise_and(background_gray, background_gray, mask=mask_roi)
#     background_gray_masked = cv2.GaussianBlur(background_gray_masked, (11, 11), 0)
#
#     motion_detected = False
#     detected_cyclist_id = set()
#     next_cyclist_id = 1
#     cyclist_count = 0
#     print("Motion detection within ROI and cyclist check started. Press 'q' to quit.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#
#         # Draw the ROI
#         cv2.polylines(frame, [roi_points], True, roi_color, roi_thickness)
#
#         # Convert current frame to grayscale and mask with ROI
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray_frame_masked = cv2.bitwise_and(gray_frame, gray_frame, mask=mask_roi)
#         gray_frame_masked = cv2.GaussianBlur(gray_frame_masked, (11, 11), 0)
#
#         frame_diff = cv2.absdiff(background_gray_masked, gray_frame_masked)
#         thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         motion_detected = False
#         for contour in contours:
#             if cv2.contourArea(contour) > min_motion_area:
#                 # Check if the centroid of the motion is within the ROI
#                 M = cv2.moments(contour)
#                 if M["m00"] != 0:
#                     cX = int(M["m10"] / M["m00"])
#                     cY = int(M["m01"] / M["m00"])
#                     if mask_roi[cY, cX] == 255:  # Check if centroid is within the white ROI mask
#                         motion_detected = True
#                         break
#
#         if motion_detected:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             image_name = os.path.join(capture_path, f"motion_roi_{timestamp}.jpg")
#             cv2.imwrite(image_name, frame)
#             print(f"Motion detected within ROI! Photo saved as: {image_name}")
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
#
#             motion_detected = False
#             time.sleep(5)
#
#         cv2.imshow("Motion Detection", frame)
#         cv2.imshow("Motion Threshold", thresh) # To visualize the motion
#
#         if cv2.waitKey(1) & 0xFF == ord('u'):
#             ret, background_frame = cap.read()
#             if ret:
#                 background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
#                 background_gray_masked = cv2.bitwise_and(background_gray, background_gray, mask=mask_roi)
#                 background_gray_masked = cv2.GaussianBlur(background_gray_masked, (11, 11), 0)
#                 print("Background updated (within ROI).")
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print(f'There were {cyclist_count} cyclists detected.')
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     create_table() # Create the table if it's the first run
#
#     # Example of inserting a count (you would call this from your cyclist detection logic)
#     # Let's say your cyclist counting variable is 'num_cyclists'
#     num_cyclists = 5
#     insert_count(num_cyclists)
#     time.sleep(10) # Simulate another detection after some time
#     num_cyclists = 2
#     insert_count(num_cyclists)
#
#     # Example of retrieving all counts
#     all_data = get_all_counts()
#     print("\nAll recorded data:")
#     for row in all_data:
#         print(row)
#
#     # Example of retrieving counts for today's date
#     today = datetime.date.today().strftime('%Y-%m-%d')
#     today_data = get_counts_by_date(today)
#     print(f"\nCounts for {today}:")
#     for row in today_data:
#         print(row)
#
#
# if __name__ == "__main__":
#     lower_red_brown = np.array([44, 59, 0])
#     upper_red_brown = np.array([179, 88, 191])
#
#     detect_motion_and_check_for_cyclist_colour_focus(lower_red_brown, upper_red_brown)

import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import sqlite3
import datetime

# Define the database file name
DATABASE_FILE = 'cyclist_data.db'


def connect_db():
    """Connects to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    return conn, cursor


def create_table():
    """Creates the cyclist_counts table if it doesn't exist."""
    conn, cursor = connect_db()
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS cyclist_counts
                   (
                       timestamp
                       TEXT
                       PRIMARY
                       KEY,
                       count
                       INTEGER
                       NOT
                       NULL
                   )
                   """)
    conn.commit()
    conn.close()


def insert_count(count):
    """Inserts the current timestamp and cyclist count into the database."""
    conn, cursor = connect_db()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute("INSERT INTO cyclist_counts (timestamp, count) VALUES (?, ?)", (now, count))
        conn.commit()
        print(f"Cyclist count of {count} recorded at {now}")
    except sqlite3.IntegrityError:
        print(f"Count already recorded for timestamp: {now}")  # Handle duplicate entries if needed
    finally:
        conn.close()


def get_all_counts():
    """Retrieves all recorded counts from the database."""
    conn, cursor = connect_db()
    cursor.execute("SELECT * FROM cyclist_counts")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_counts_by_date(date_str):
    """Retrieves counts for a specific date (YYYY-MM-DD)."""
    conn, cursor = connect_db()
    cursor.execute("SELECT * FROM cyclist_counts WHERE strftime('%Y-%m-%d', timestamp) = ?", (date_str,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def detect_motion_and_check_for_cyclist_colour_focus(lower_color, upper_color, motion_threshold=120,
                                                     min_motion_area=1000, capture_path="captured_images"):
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
    mask_roi = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                        dtype=np.uint8)
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
    print("Motion detection within ROI and cyclist check started. Press 'q' to quit.")

    create_table()  # Initialize the database table

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
            results = model(image_name, verbose=False)  # Suppress YOLOv8 output
            cyclist_count = 0  # Reset count for each detection event

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    if class_name == 'bicycle' or class_name == 'person':  # Adjust based on your model's classes
                        cyclist_count += 1

            if cyclist_count > 0:
                print(f"Number of cyclists detected: {cyclist_count}")
                insert_count(cyclist_count)  # Save the actual count to the database

            motion_detected = False
            time.sleep(5)

        cv2.imshow("Motion Detection", frame)
        cv2.imshow("Motion Threshold", thresh)  # To visualize the motion

        if cv2.waitKey(1) & 0xFF == ord('u'):
            ret, background_frame = cap.read()
            if ret:
                background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
                background_gray_masked = cv2.bitwise_and(background_gray, background_gray, mask=mask_roi)
                background_gray_masked = cv2.GaussianBlur(background_gray_masked, (11, 11), 0)
                print("Background updated (within ROI).")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f'Detection stopped.')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    lower_red_brown = np.array([44, 59, 0])
    upper_red_brown = np.array([179, 88, 191])

    detect_motion_and_check_for_cyclist_colour_focus(lower_red_brown, upper_red_brown)

