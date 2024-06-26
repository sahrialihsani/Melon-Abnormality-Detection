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
import ntplib
from google.cloud import firestore

db = firestore.Client.from_service_account_json("firestore_credentials.json")
class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    streamlit app that uses yolov7
    '''
    def __init__(self,):
        self.detection_results = []
        self.logging_main=logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    def new_yolo_model(self,img_size,path_yolov7_weights,path_img_i,device_i='cpu'):
        '''
        SimpleInference_YOLOV7
        created by Steven Smiley 2022/11/24

        INPUTS:
        VARIABLES                    TYPE    DESCRIPTION
        1. img_size,                    #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
        2. path_yolov7_weights,         #str#   #this is the path to your yolov7 weights 
        3. path_img_i,                  #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())

        OUTPUT:
        VARIABLES                    TYPE    DESCRIPTION
        1. predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)

        CREDIT
        Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
        @article{wang2022yolov7,
            title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
            author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
            journal={arXiv preprint arXiv:2207.02696},
            year={2022}
            }
        
        '''
        super().__init__(img_size,path_yolov7_weights,path_img_i,device_i=device_i)
    
    def main(self):
        st.title('MeloAnalytics')
        st.subheader(""" Upload image and video then run YoloV7 on it.  
        Use multiple models to gain accuracy of melon abnormality on leaves more accurate.\n""")
        st.markdown(
            """
        <style>
        .reportview-container .markdown-text-container {
            font-family: monospace;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
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
        st.markdown(
            """
            <style>
            .reportview-container {
                background-color: green
            }
        .sidebar .sidebar-content {
                background-color: green
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        self.conf_selection=st.selectbox('Confidence Threshold',tuple([0.5,0.75,0.95]))
        self.iou_selection=st.selectbox('IoU Threshold',tuple([0.5,0.75,0.95]))

        input_type = st.sidebar.selectbox(
            'Input Type', ('Image', 'Video')
        )

        if input_type == 'Image':
            self.response=requests.get(self.path_img_i)
    
            self.img_screen=Image.open(BytesIO(self.response.content))
    
            st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            st.markdown('YoloV7 on streamlit.  Demo of object detection with YoloV7 with a web application.')
            self.im0=np.array(self.img_screen.convert('RGB'))
            self.load_image_st()
            predictions = st.button('Predict on the image?')
            if predictions:
                self.predict()
                predictions=False
        elif input_type == 'Video':
                self.response = requests.get(self.path_img_i)
                self.img_screen = Image.open(BytesIO(self.response.content))
                st.markdown('YoloV7 on streamlit. Demo of object detection with YoloV7 with a web application.')
                self.video_frames = np.array(self.img_screen.convert('RGB'))
                self.load_video_st()
                predictions = st.button('Predict on the video?')
                if predictions:
                    self.predict_on_video()
                    predictions = False
    def load_video_st(self):
        uploaded_video = st.file_uploader(label='Upload a video', type=["mp4", "avi", "mkv"])
        if type(uploaded_video) != type(None):
            video_data = uploaded_video.read()
            st.video(video_data)
            self.video_frames = self.load_video_frames(video_data)
            return self.video_frames
        elif hasattr(self, 'video_frames'):
            return self.video_frames
        else:
            return None
    
    # Helper method to load video frames
    def load_video_frames(self, video_data):
        # Save video data to a temporary file
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
    
        # Delete the temporary file after use
        os.remove(temp_file_path)
    
        return frames
        
    def load_image_st(self):
        uploaded_img=st.file_uploader(label='Upload an image')
        if type(uploaded_img) != type(None):
            self.img_data=uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0=Image.open(BytesIO(self.img_data))#.convert('RGB')
            self.im0=np.array(self.im0)
            return self.im0
        elif type(self.im0) !=type(None):
            return self.im0
        else:
            return None
    def get_current_datetime(self):
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')
        return datetime.fromtimestamp(response.tx_time, tz=timezone.utc).strftime('%Y-%m-%d %H-%M-%S')
        
    def predict(self):
        self.conf_thres = self.conf_selection
        self.iou_thres = self.iou_selection
        st.write('Loading image')
        self.load_cv2mat(self.im0)
        st.write('Making inference')
        self.inference()
        
        self.img_screen = Image.fromarray(self.image).convert('RGB')
        
        self.capt = 'DETECTED:'
        current_image_results = []  # Store results for the current image
        
        if len(self.predicted_bboxes_PascalVOC) > 0:
            for item in self.predicted_bboxes_PascalVOC:
                name = str(item[0])
                conf = str(round(100 * item[-1], 2))
                self.capt = self.capt + ' name=' + name + ' confidence=' + conf + '%, '
                current_image_results.append({'name': name, 'confidence': float(conf)})

        # Save the detection results for the current image
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
        # Convert the list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(json_array)
        self.image = None
        st.subheader("""Detection Result""")
        st.table(df)
        for index in df.index:
            # Generate a unique document ID based on the date and time
            current_datetime = self.get_current_datetime()
            unique_identifier = f"{current_datetime}"  # You can customize this format as needed
            doc_ref = db.collection("results").document()
            doc_ref.set({"waktu": unique_identifier, "kelas": df.loc[index, 'name'], "akurasi": df.loc[index, 'confidence']})

    def predict_on_video(self):
        self.conf_thres = self.conf_selection
        self.iou_thres = self.iou_selection
        for frame in self.video_frames:
            st.write('Loading video frame')
            self.load_cv2mat(frame)
            st.write('Making inference')
            self.inference()
    
            self.img_screen = Image.fromarray(self.image).convert('RGB')
    
            self.capt = 'DETECTED:'
            current_frame_results = []  # Store results for the current video frame
    
            if len(self.predicted_bboxes_PascalVOC) > 0:
                for item in self.predicted_bboxes_PascalVOC:
                    name = str(item[0])
                    conf = str(round(100 * item[-1], 2))
                    self.capt = self.capt + ' name=' + name + ' confidence=' + conf + '%, '
                    current_frame_results.append({'name': name, 'confidence': float(conf)})
            st.image(
                self.img_screen,
                width=None,
                use_column_width=None,
                clamp=False,
                channels="RGB",
                output_format="auto",
            )
            json_array = current_frame_results # Use results for the last frame
            df = pd.DataFrame(json_array)
            st.subheader("""Detection Result""")
            st.table(df)
            for index in df.index:
                # Generate a unique document ID based on the date and time
                current_datetime = self.get_current_datetime()
                unique_identifier = f"{current_datetime}"  # You can customize this format as needed
                doc_ref = db.collection("results").document()
                doc_ref.set({"waktu": unique_identifier, "kelas": df.loc[index, 'name'], "akurasi": df.loc[index, 'confidence']})
            self.image = None
            
if __name__=='__main__':
    app=Streamlit_YOLOV7()

    #INPUTS for YOLOV7
    img_size=640
    path_yolov7_weights= ["weights/best_1.pt","weights/best_2.pt","weights/best_3.pt","weights/best_4.pt"]
    path_img_i="https://raw.githubusercontent.com/sahrialihsani/Melon-Abnormality-Detection/main/test_images/2020-02-18-22-54-20_jpg.rf.fb182d3ca77327e644c61382fe0c9ffe.jpg"
    #INPUTS for webapp
    app.capt="Initial Image"
    app.new_yolo_model(img_size,path_yolov7_weights,path_img_i)
    app.conf_thres=0.5
    app.iou_thres=0.75
    app.load_model() #Load the yolov7 model
    
    app.main()
