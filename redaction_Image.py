import streamlit as st
import cv2
import numpy as np
import pytesseract
import spacy
from PIL import Image
import io
import json
from datetime import datetime
import string
from PIL import Image
import io
import json
from datetime import datetime
import string
import pandas as pd

# Load spaCy model
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

def load_image(image_file):
    """Load and convert uploaded image to numpy array"""
    img = Image.open(image_file)
    return np.array(img)

def perform_ocr(image):
    """Perform OCR on the image and return annotated boxes and text"""
    custom_config = r'--oem 3 --psm 6'  # Customize OCR parameters if needed
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
    
    draw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 2 else image.copy()
    
    annotations = []
    n_boxes = len(ocr_data['text'])
    for i in range(n_boxes):
        try:
            if int(ocr_data['conf'][i]) > 30:  # Only consider confident OCR outputs
                text = ocr_data['text'][i].strip()
                if text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    cv2.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    annotations.append({
                        'id': len(annotations),
                        'text': text,
                        'confidence': int(ocr_data['conf'][i]),
                        'coordinates': {
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        }
                    })
        except KeyError:
            continue  # Skip invalid data
        except ValueError:
            continue  # Skip if confidence or dimensions are invalid
    
    return draw_image, annotations


def should_merge_boxes(box1, box2, max_horizontal_gap=50, max_vertical_gap=10):
    """Determine if two boxes should be merged based on proximity"""
    x1, y1, w1, h1 = box1['coordinates'].values()
    x2, y2, w2, h2 = box2['coordinates'].values()
    
    horizontal_gap = x2 - (x1 + w1) if x2 > x1 else x1 - (x2 + w2)
    vertical_gap = abs(y1 - y2)
    height_ratio = max(h1, h2) / min(h1, h2)
    
    return (horizontal_gap < max_horizontal_gap and 
            vertical_gap < max_vertical_gap and 
            height_ratio < 1.5)

def merge_boxes(box1, box2):
    """Merge two bounding boxes into one"""
    x1 = min(box1['coordinates']['x'], box2['coordinates']['x'])
    y1 = min(box1['coordinates']['y'], box2['coordinates']['y'])
    x2 = max(box1['coordinates']['x'] + box1['coordinates']['width'],
             box2['coordinates']['x'] + box2['coordinates']['width'])
    y2 = max(box1['coordinates']['y'] + box1['coordinates']['height'],
             box2['coordinates']['y'] + box2['coordinates']['height'])
    
    return {
        'id': f"{box1['id']}-{box2['id']}",
        'text': f"{box1['text']} {box2['text']}",
        'confidence': min(box1['confidence'], box2['confidence']),
        'coordinates': {
            'x': int(x1),
            'y': int(y1),
            'width': int(x2 - x1),
            'height': int(y2 - y1)
        }
    }

def clean_text(text):
    """Clean text by removing unnecessary punctuation and retaining certain special characters."""
    allowed_punctuation = "‘’“”'\"-."
    cleaned_text = ''.join([char if char in string.ascii_letters + string.digits + allowed_punctuation else ' ' for char in text])
    return cleaned_text.strip()

def perform_ner(annotations, nlp, entity_type=None):
    """Perform NER on annotations and group boxes belonging to the same entity"""
    entities = []
    processed_groups = []
    
    # Sort annotations by position
    sorted_annotations = sorted(annotations, key=lambda x: (x['coordinates']['y'], x['coordinates']['x']))
    
    # Group potentially related boxes
    i = 0
    while i < len(sorted_annotations):
        current_group = [sorted_annotations[i]]
        j = i + 1
        while j < len(sorted_annotations):
            if should_merge_boxes(current_group[-1], sorted_annotations[j]):
                current_group.append(sorted_annotations[j])
            else:
                break
            j += 1
        
        if len(current_group) > 1:
            # Merge the boxes of this group
            merged_box = current_group[0]
            for box in current_group[1:]:
                merged_box = merge_boxes(merged_box, box)
            processed_groups.append(merged_box)
        else:
            processed_groups.append(current_group[0])
        
        i = j if j > i + 1 else i + 1
    
    # Process each group for NER
    for group in processed_groups:
        clean_group_text = clean_text(group['text'])
        doc = nlp(clean_group_text)
        
        for ent in doc.ents:
            if entity_type is None or ent.label_ == entity_type:
                # Ensure group['id'] is treated as a string
                box_ids = str(group['id']).split('-')
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'box_ids': box_ids,
                    'original_text': group['text']
                })
    
    return entities

def redact_entities(image, entities, annotations):
    """
    Redact specific entities on the image by drawing black rectangles over just the entity text.
    """
    image_np = image.copy()
    
    for entity in entities:
        # Get all relevant boxes for this entity
        box_ids = entity['box_ids']
        entity_annotations = []
        
        for box_id in box_ids:
            matching_annotations = [ann for ann in annotations if str(ann['id']) == box_id]
            entity_annotations.extend(matching_annotations)
        
        if entity_annotations:
            # Sort annotations by x coordinate
            entity_annotations.sort(key=lambda x: x['coordinates']['x'])
            
            # Find the start and end points for redaction
            min_x = min(ann['coordinates']['x'] for ann in entity_annotations)
            max_x = max(ann['coordinates']['x'] + ann['coordinates']['width'] 
                       for ann in entity_annotations)
            min_y = min(ann['coordinates']['y'] for ann in entity_annotations)
            max_y = max(ann['coordinates']['y'] + ann['coordinates']['height'] 
                       for ann in entity_annotations)
            
            # For each line of text containing the entity
            current_y = min_y
            while current_y <= max_y:
                # Find annotations on this line
                line_annotations = [
                    ann for ann in entity_annotations 
                    if abs(ann['coordinates']['y'] - current_y) < ann['coordinates']['height']
                ]
                
                if line_annotations:
                    # Get coordinates for this line
                    line_min_x = min(ann['coordinates']['x'] for ann in line_annotations)
                    line_max_x = max(ann['coordinates']['x'] + ann['coordinates']['width'] 
                                   for ann in line_annotations)
                    line_height = max(ann['coordinates']['height'] for ann in line_annotations)
                    
                    # Draw redaction rectangle for this line
                    cv2.rectangle(image_np,
                                (int(line_min_x), int(current_y)),
                                (int(line_max_x), int(current_y + line_height)),
                                (0, 0, 0),
                                -1)
                
                # Move to next possible line
                current_y += max(ann['coordinates']['height'] for ann in entity_annotations)
    
    return image_np

def highlight_boxes(image, annotations, box_ids):
    """
    Debug function to highlight boxes being processed
    """
    draw_image = image.copy()
    
    for box_id in box_ids:
        matching_annotations = [ann for ann in annotations if str(ann['id']) == box_id]
        for ann in matching_annotations:
            x = ann['coordinates']['x']
            y = ann['coordinates']['y']
            w = ann['coordinates']['width']
            h = ann['coordinates']['height']
            cv2.rectangle(draw_image, 
                         (int(x), int(y)), 
                         (int(x + w), int(y + h)), 
                         (0, 0, 255), 
                         2)
    
    return draw_image





def main():
    st.set_page_config(page_title="OCR and NER System", layout="wide")
    st.title("OCR and Named Entity Recognition with Redaction")
    
    # Initialize session state
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = None
    if 'entities' not in st.session_state:
        st.session_state.entities = None
    
    nlp = load_nlp_model()
    if not nlp:
        return
    
    # Sidebar controls
    st.sidebar.title("Settings")
    entity_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'MONEY']
    selected_entity = st.sidebar.selectbox("Filter by entity type", ['All'] + entity_types)
    
    # OCR confidence threshold
    confidence_threshold = st.sidebar.slider("OCR Confidence Threshold", 0, 100, 30)
    
    # Image upload
    image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if image_file is not None:
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        image = load_image(image_file)
        if image is not None:
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Process Image"):
                try:
                    with st.spinner('Processing image...'):
                        # Perform OCR
                        annotated_image, annotations = perform_ocr(image)
                        
                        # Store annotations in session state
                        st.session_state.annotations = annotations
                        
                        # Filter annotations by confidence threshold
                        annotations = [ann for ann in annotations if ann['confidence'] >= confidence_threshold]
                        
                        # Perform NER
                        entity_type = None if selected_entity == 'All' else selected_entity
                        entities = perform_ner(annotations, nlp, entity_type)
                        
                        with col2:
                            st.image(annotated_image, caption="Detected Text", use_column_width=True)
                        
                        # Display results in an expander
                        with st.expander("Detection Results", expanded=True):
                            # Display entities in a more organized way
                            if entities:
                                entity_df = pd.DataFrame(entities)
                                entity_df = entity_df[['text', 'type', 'original_text']]
                                st.dataframe(entity_df)
                                
                                # Save results
                                st.session_state.entities = entities
                                st.session_state.annotated_image = annotated_image
                                
                                # Add download button for results
                                results = {
                                    'entities': entities,
                                    'annotations': annotations,
                                    'timestamp': datetime.now().isoformat()
                                }
                                st.download_button(
                                    label="Download Results",
                                    data=json.dumps(results, indent=2),
                                    file_name="ocr_results.json",
                                    mime="application/json"
                                )
                            else:
                                st.warning("No entities found.")
                                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            
            # Redaction controls
            if (st.session_state.entities is not None and 
                st.session_state.annotations is not None):
                if st.button("Redact Entities"):
                    with st.spinner('Redacting entities...'):
                        redacted_image = redact_entities(
                            image, 
                            st.session_state.entities, 
                            st.session_state.annotations
                        )
                        st.image(redacted_image, caption="Redacted Image", use_column_width=True)
                        
                        # Add download button for redacted image
                        redacted_pil = Image.fromarray(redacted_image)
                        img_byte_arr = io.BytesIO()
                        redacted_pil.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        st.download_button(
                            label="Download Redacted Image",
                            data=img_byte_arr,
                            file_name="redacted_image.png",
                            mime="image/png"
                        )

if __name__ == "__main__":
    main()