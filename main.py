
### Health Management APP
from dotenv import load_dotenv

load_dotenv() ## load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Google Gemini Pro Vision API And get response

def get_gemini_repsonse(input,image,prompt):
    model=genai.GenerativeModel('gemini-1.5-pro')
    response=model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
##initialize our streamlit app

st.set_page_config(page_title="nutritional information")

st.header(" nutritional information ")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.")


submit=st.button(" Predict")

input_prompt="""


            rompt:

You are a smart AI assistant specialized in extracting nutritional information from FMCG product packaging. Your task is to identify the product name and its nutritional information from the given input. If the text is blurred or hard to read, attempt to re-analyze the image or text carefully. Structure the output in the following JSON format:

json
Copy code
{
  "Product Name": "<Extracted Product Name>",
  "Nutritional Information": {
    "Calories": "<Extracted Calories>",
    "Protein": "<Extracted Protein>",
    "Carbohydrates": "<Extracted Carbohydrates>",
    "Fats": "<Extracted Fats>"
  }
}
 
 not  only  Calories protein carbohydrates fats exact what ever present in the image  sugar and sodium 

 
Steps to follow:

Extract the product name.
Identify and extract the nutritional values (e.g., calories, protein, carbohydrates, fats, and others if available).
If any values are missing, mention "Not Available" for that field in the JSON output.
Example Input: A product packaging image of "Choco Delight."

and check the expire date of the product if the proudct is expired tell name of the proudct is expired or it is not expire 

not  only  Calories protein carbohydrates fats exact what ever present in the image  sugar and sodium 

Expected Output:

json
Copy code
{
  "Product Name": "Choco Delight",
  "Nutritional Information": {
    "Calories": "250 kcal",
    "Protein": "6g",
    "Carbohydrates": "35g",
    "Fats": "8g"

  }
}

 " choco Delight is  not expire " written this in big words 



  
"""

## If submit button is clicked

if submit:
    image_data=input_image_setup(uploaded_file)
    response=get_gemini_repsonse(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from datetime import datetime
import os

# Dictionary of prices for different items
prices = {
    'fivestar': 10,  # Added Five Star chocolate
    'large_item': 100,  # Added large item
    'apple': 20,
    'banana': 15,
    'milk': 50,
    'bread': 25
}

class SmartTrolley:
    def __init__(self):
        self.tracked_items = {}  # {track_id: {'item': name, 'price': price, 'last_seen': frame_count, 'position': (x,y)}}
        self.item_quantities = {}  # {item_name: quantity}
        self.total_bill = 0
        self.frame_count = 0
        self.confidence_threshold = 0.6
        self.removal_threshold = 30  # Frames before considering item removed
        self.position_threshold = 50  # Pixel distance threshold for considering same item

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def is_duplicate_detection(self, item_name, position):
        """Check if an item detection is a duplicate based on position"""
        for data in self.tracked_items.values():
            if (data['item'] == item_name and
                self.calculate_distance(data['position'], position) < self.position_threshold and
                self.frame_count - data['last_seen'] < self.removal_threshold):
                return True
        return False

    def update_item_quantity(self, item_name, change):
        """Update quantity of an item"""
        if item_name not in self.item_quantities:
            self.item_quantities[item_name] = 0
        self.item_quantities[item_name] += change

    def remove_old_items(self):
        """Remove items that haven't been seen for a while"""
        items_to_remove = []
        for track_id, data in self.tracked_items.items():
            if self.frame_count - data['last_seen'] > self.removal_threshold:
                items_to_remove.append(track_id)
                self.total_bill -= data['price']
                self.update_item_quantity(data['item'], -1)
                print(f"Removed {data['item']} (ID: {track_id}) - Price: ${data['price']}")

        for track_id in items_to_remove:
            del self.tracked_items[track_id]

# Initialize the system
trolley = SmartTrolley()

# Load the YOLO model
model = YOLO("yolo11s.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    trolley.frame_count += 1
    frame = cv2.resize(frame, (1020, 600))

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True)

    # Process detection results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        current_frame_detections = set()  # Track detections in current frame

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            if conf < trolley.confidence_threshold:
                continue

            item_name = model.names[class_id]
            x1, y1, x2, y2 = box
            center_position = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Check for duplicates based on position
            if not trolley.is_duplicate_detection(item_name, center_position):
                if track_id not in trolley.tracked_items and item_name in prices:
                    # New item added to trolley
                    trolley.tracked_items[track_id] = {
                        'item': item_name,
                        'price': prices[item_name],
                        'last_seen': trolley.frame_count,
                        'position': center_position
                    }
                    trolley.total_bill += prices[item_name]
                    trolley.update_item_quantity(item_name, 1)
                    print(f"Added {item_name} (ID: {track_id}) - Price: ${prices[item_name]}")

            # Update tracking info for existing items
            if track_id in trolley.tracked_items:
                trolley.tracked_items[track_id]['last_seen'] = trolley.frame_count
                trolley.tracked_items[track_id]['position'] = center_position

            # Draw bounding box and label
            color = (0, 255, 0) if item_name == 'fivestar' else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add item label with tracking ID
            label = f"{item_name} (ID: {track_id})"
            cvzone.putTextRect(frame, label, (x1, y1-20), 1, 1)

            current_frame_detections.add(track_id)

    # Display total bill and quantities
    y_pos = 30
    cvzone.putTextRect(frame, f"Total Bill: ${trolley.total_bill:.2f}", (10, y_pos), 2, 2)

    y_pos += 40
    cvzone.putTextRect(frame, "Items in Trolley:", (10, y_pos), 2, 2)

    # Display quantities with special highlight for Five Star chocolates
    for item_name, quantity in trolley.item_quantities.items():
        if quantity > 0:
            y_pos += 30
            text_color = (0, 255, 0) if item_name == 'fivestar' else (255, 255, 255)
            cvzone.putTextRect(frame, f"{item_name}: x{quantity}", (10, y_pos), 1, 1,
                             colorR=text_color if item_name == 'fivestar' else None)

    # Show the frame
    cv2.imshow("Smart Trolley System", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        trolley = SmartTrolley()
        print("System reset!")

cap.release()
cv2.destroyAllWindows()