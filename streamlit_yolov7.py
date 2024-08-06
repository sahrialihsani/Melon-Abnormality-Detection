import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import cv2
from datetime import datetime, timezone
import tempfile
from google.cloud import firestore
import uuid

db = firestore.Client.from_service_account_json("firestore_credentials.json")

class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    Streamlit app that uses YOLOv7
    '''
    def __init__(self):
        self.detection_results = []
        self.logging_main = logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    def new_yolo_model(self, img_size, path_yolov7_weights, path_img_i, device_i='cpu'):
        super().__init__(img_size, path_yolov7_weights, path_img_i, device_i=device_i)

    def main(self):
        st.title('MeloAnalytics')
        st.subheader("""Input: image, video, webcam then run YOLOv7 on it.\n""")
        st.markdown(
            """
            <style>
            .reportview-container .markdown-text-container {
                font-family: monospace;
            }
            .sidebar .sidebar-content {
                color: white;
            }
            .Widget>label {
                color: green;
                font-family: monospace;
            }
            [class^="st-b"]  {
                color: green;
                font-family: monospace;
            }
            .st-bb {
                background-color: black;
            },
            .st-eb {
                color: black;
            },
            .st-at {
                background-color: green;
            }
            footer {
                font-family: monospace;
            }
            .reportview-container .main footer, .reportview-container .main footer a {
                color: white;
            }
            header .decoration {
                background-image: None);
            },
            </style>
            """,
            unsafe_allow_html=True,
        )
        self.conf_selection = st.selectbox('Confidence Threshold', tuple([0.5, 0.75, 0.95]))
        self.iou_selection = st.selectbox('IoU Threshold', tuple([0.5, 0.75, 0.95]))

        input_type = st.sidebar.selectbox('Input Type', ('Image', 'Video', 'Webcam'))

        if input_type == 'Image':
            self.response = requests.get(self.path_img_i)
            response.raise_for_status()
            self.img_screen = Image.open(BytesIO(self.response.content))
            st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            st.markdown('YOLOv7 on Streamlit. Demo of object detection with YOLOv7 with a web application.')
            self.im0 = np.array(self.img_screen.convert('RGB'))
            self.load_image_st()

            predictions = st.button('Predict on the image?')
            if predictions:
                self.predict()
                predictions = False

        elif input_type == 'Video':
            self.response = requests.get(self.path_img_i)
            response.raise_for_status()
            self.img_screen = Image.open(BytesIO(self.response.content))
            st.markdown('YOLOv7 on Streamlit. Demo of object detection with YOLOv7 with a web application.')
            self.video_frames = np.array(self.img_screen.convert('RGB'))
            self.load_video_st()
            predictions = st.button('Predict on the video?')
            if predictions:
                self.predict_on_video()
                predictions = False

        elif input_type == 'Webcam':
            self.conf_thres = self.conf_selection
            self.iou_thres = self.iou_selection
            self.load_webcam()

    def load_webcam(self):
        all_frame_results = []
        df_all_frames = []
    
        pipeline = (
            'v4l2src device=/dev/video0 ! '
            'video/x-raw, width=640, height=480, format=(string)YUY2 ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! '
            'appsink drop=1'
        )
    
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.write("Error: Unable to open video source.")
            return
    
        frame_placeholder = st.empty()
        stop_button_pressed = st.button("Stop")
    
        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Unable to read frame from video source.")
                break
    
            self.load_cv2mat(frame)
            self.inference()
    
            current_frame_results = []
    
            if len(self.predicted_bboxes_PascalVOC) > 0:
                for item in self.predicted_bboxes_PascalVOC:
                    name = str(item[0])
                    conf = str(round(100 * item[-1], 2))
                    current_frame_results.append({'name': name, 'confidence': float(conf)})
    
            all_frame_results.extend(current_frame_results)
    
            ret, buffer = cv2.imencode('.jpg', self.image)
            if not ret:
                st.write("Error: Unable to encode frame to JPEG.")
                break
    
            frame_bytes = buffer.tobytes()
            frame_placeholder.image(frame_bytes, channels="BGR")
    
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break
    
        df_all_frames = pd.DataFrame(all_frame_results)
        st.subheader("Consolidated Detection Results for All Frames")
        st.table(df_all_frames)
        for index in df_all_frames.index:
            random_id = str(uuid.uuid4())
            current_datetime = self.get_current_datetime()
            unique_identifier = f"{current_datetime}"
            doc_ref = db.collection("results").document(random_id)
            doc_ref.set({
                "id": random_id,
                "waktu": unique_identifier,
                "kelas": df_all_frames.loc[index, 'name'],
                "akurasi": df_all_frames.loc[index, 'confidence']
            })
    
        cap.release()
        cv2.destroyAllWindows()
        st.write("Webcam stream has been stopped.")
    
        if cap.isOpened():
            st.write("Error: Unable to properly release the video capture device.")

    def load_video_st(self):
        uploaded_video = st.file_uploader(label='Upload a video', type=["mp4", "avi", "mkv"])
        if uploaded_video is not None:
            video_data = uploaded_video.read()
            st.video(video_data)
            self.video_frames = self.load_video_frames(video_data)
            return self.video_frames
        elif hasattr(self, 'video_frames'):
            return self.video_frames
        else:
            return None

    def load_video_frames(self, video_data):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_data)
        temp_file_path = temp_file.name
        temp_file.close()

        cap = cv2.VideoCapture(temp_file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        os.remove(temp_file_path)
        return frames

    def load_image_st(self):
        uploaded_img = st.file_uploader(label='Upload an image')
        if uploaded_img is not None:
            self.img_data = uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0 = Image.open(BytesIO(self.img_data))
            self.im0 = np.array(self.im0)
            return self.im0
        elif hasattr(self, 'im0'):
            return self.im0
        else:
            return None

    def get_current_datetime(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def predict(self):
        self.conf_thres = self.conf_selection
        self.iou_thres = self.iou_selection
        st.write('Loading image')
        self.load_cv2mat()
        st.write('Making inference')
        self.inference()

        self.img_screen = Image.fromarray(self.image).convert('RGB')
        self.capt = 'DETECTED:'
        current_image_results = []

        if len(self.predicted_bboxes_PascalVOC) > 0:
            for item in self.predicted_bboxes_PascalVOC:
                name = str(item[0])
                conf = str(round(100 * item[-1], 2))
                self.capt = self.capt + ' name=' + name + ' confidence=' + conf + '%, '
                current_image_results.append({'name': name, 'confidence': float(conf)})

        self.detection_results.append(current_image_results)
        st.image(
            self.img_screen,
            width=None,
            use_column_width=None,
            clamp=False,
            channels="RGB",
            output_format="auto",
        )
        json_array = self.detection_results[0]
        df = pd.DataFrame(json_array)
        count_normal = len(df[df['name'] == "normal"])
        count_abnormal = len(df[df['name'] == "abnormal"])
        self.image = None
        st.subheader("""Detection Result""")
        st.table(df)
        # Checking the condition and displaying the appropriate message
        
        st.subheader("""Status""")
        if count_normal >= count_abnormal:
            st.write("Tanaman Normal: Jumlah daun normal lebih besar atau sama dengan jumlah daun yang abnormal.")
        else:
            st.write("Tanaman Terdapat Penyakit / Defisiensi Unsur Hara: Jumlah daun abnormal lebih besar dari jumlah daun normal.")

        for index in df.index:
            random_id = str(uuid.uuid4())
            current_datetime = self.get_current_datetime()
            unique_identifier = f"{current_datetime}"
            doc_ref = db.collection("results").document(random_id)
            doc_ref.set({"id": random_id, "waktu": unique_identifier, "kelas": df.loc[index, 'name'], "akurasi": df.loc[index, 'confidence']})

    def predict_on_video(self):
        try:
            self.conf_thres = self.conf_selection
            self.iou_thres = self.iou_selection
            all_frame_results = []

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1980, 1080))

            st.write('Making inference')
            for frame in self.video_frames:
                self.load_cv2mat(frame)
                self.inference()

                current_frame_results = []

                if len(self.predicted_bboxes_PascalVOC) > 0:
                    for item in self.predicted_bboxes_PascalVOC:
                        name = str(item[0])
                        conf = str(round(100 * item[-1], 2))
                        current_frame_results.append({'name': name, 'confidence': float(conf)})

                all_frame_results.extend(current_frame_results)
                out.write(self.image)

            out.release()

            df_all_frames = pd.DataFrame(all_frame_results)

            st.subheader("""Consolidated Detection Results for All Frames""")
            st.table(df_all_frames)

            # Dynamically determine normal and abnormal detections
            normal_detections = df_all_frames[df_all_frames['name'].str.contains('normal', case=False)]['name'].unique()
            abnormal_detections = df_all_frames[df_all_frames['name'].str.contains('abnormal', case=False)]['name'].unique()

            # Count normal and abnormal detections
            count_normal = df_all_frames['name'].isin(normal_detections).sum()
            count_abnormal = df_all_frames['name'].isin(abnormal_detections).sum()

            # Display the appropriate message based on the counts
            if count_normal >= count_abnormal:
                st.write("Tanaman Normal: Jumlah daun normal lebih besar atau sama dengan jumlah daun yang abnormal.")
            else:
                st.write("Tanaman Terdapat Penyakit / Defisiensi Unsur Hara: Jumlah daun abnormal lebih besar dari jumlah daun normal.")

            # Storing results in Firestore
            for index in df_all_frames.index:
                random_id = str(uuid.uuid4())
                current_datetime = self.get_current_datetime()
                unique_identifier = f"{current_datetime}"
                doc_ref = db.collection("results").document()
                doc_ref.set({
                    "id": random_id,
                    "waktu": unique_identifier,
                    "kelas": df_all_frames.loc[index, 'name'],
                    "akurasi": df_all_frames.loc[index, 'confidence']
                })

            self.image = None

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app = Streamlit_YOLOV7()


    # INPUTS for YOLOV7
    img_size = 640
    path_yolov7_weights = ["weights/maximum_epochs/best_fold4.pt"]
    path_img_i = "https://github.com/sahrialihsani/Melon-Abnormality-Detection/blob/main/test_images/test.jpg"
    # INPUTS for webapp
    app.capt = "Initial Image"
    app.new_yolo_model(img_size, path_yolov7_weights, path_img_i)
    app.conf_thres=0.5
    app.iou_thres=0.5
    app.load_model() # Load the YOLOv7 model
    app.main()
