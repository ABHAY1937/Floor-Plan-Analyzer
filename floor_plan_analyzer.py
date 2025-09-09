import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
from paddleocr import PaddleOCR
import re
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FloorPlanAnalyzer:
    def __init__(self, yolo_model_path: str):
        """
        Initialize the floor plan analyzer
        
        Args:
            yolo_model_path: Path to the trained YOLOv8 model
        """
        self.yolo_model = YOLO(yolo_model_path)
        # Initialize PaddleOCR with English language
        self.ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply different preprocessing techniques
        # 1. Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 2. Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. Adaptive thresholding for better text extraction
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh
        
    def detect_door_symbols(self, image: np.ndarray, confidence_threshold: float = 0.5):
        """
        Detect door symbols using YOLOv8 model
        
        Args:
            image: Input floor plan image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            results: YOLOv8 detection results
            door_count: Number of door symbols detected
        """
        results = self.yolo_model(image, conf=confidence_threshold)
        
        # Count door symbols (assuming class 0 is door symbol)
        door_count = 0
        if len(results) > 0 and results[0].boxes is not None:
            door_count = len(results[0].boxes)
            
        return results, door_count
    
    def extract_text_labels(self, image: np.ndarray) -> List[Dict]:
        """
        Extract all text labels from the floor plan using PaddleOCR
        
        Args:
            image: Input floor plan image
            
        Returns:
            List of dictionaries containing text and bounding box coordinates
        """
        text_labels = []
        
        try:
            # Preprocess image for better OCR
            processed_img = self.preprocess_image_for_ocr(image)
            
            # Use PaddleOCR on original image
            ocr_results_original = self.ocr_reader.ocr(image, cls=True)
            
            # Use PaddleOCR on preprocessed image
            ocr_results_processed = self.ocr_reader.ocr(processed_img, cls=True)
            
            # Combine results from both images
            all_results = []
            if ocr_results_original and ocr_results_original[0]:
                all_results.extend(ocr_results_original[0])
            if ocr_results_processed and ocr_results_processed[0]:
                all_results.extend(ocr_results_processed[0])
            
            # Remove duplicates and process results
            seen_texts = set()
            for result in all_results:
                if result is None:
                    continue
                    
                bbox, (text, confidence) = result
                
                # Filter by confidence and avoid duplicates
                if confidence > 0.3 and text.strip():
                    text_clean = text.strip().upper()
                    
                    # Create a simple hash for duplicate detection
                    bbox_center = ((bbox[0][0] + bbox[2][0])/2, (bbox[0][1] + bbox[2][1])/2)
                    text_hash = f"{text_clean}_{int(bbox_center[0]/10)}_{int(bbox_center[1]/10)}"
                    
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        
                        # Convert bbox to standard format (x1, y1, x2, y2)
                        bbox_array = np.array(bbox)
                        x1, y1 = bbox_array.min(axis=0).astype(int)
                        x2, y2 = bbox_array.max(axis=0).astype(int)
                        
                        # Ensure bounding box is valid
                        if x2 > x1 and y2 > y1:
                            text_labels.append({
                                'text': text_clean,
                                'original_text': text.strip(),
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence
                            })
                            
        except Exception as e:
            st.warning(f"OCR processing error: {str(e)}")
            # Fallback: try with basic OCR settings
            try:
                basic_results = self.ocr_reader.ocr(image, cls=False)
                if basic_results and basic_results[0]:
                    for result in basic_results[0]:
                        if result is None:
                            continue
                        bbox, (text, confidence) = result
                        if confidence > 0.3 and text.strip():
                            bbox_array = np.array(bbox)
                            x1, y1 = bbox_array.min(axis=0).astype(int)
                            x2, y2 = bbox_array.max(axis=0).astype(int)
                            
                            if x2 > x1 and y2 > y1:
                                text_labels.append({
                                    'text': text.strip().upper(),
                                    'original_text': text.strip(),
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': confidence
                                })
            except Exception as e2:
                st.error(f"Fallback OCR also failed: {str(e2)}")
        
        return text_labels
    
    def find_matching_labels(self, text_labels: List[Dict], target_label: str) -> List[Dict]:
        """
        Find text labels that match the target label with improved matching
        
        Args:
            text_labels: List of all detected text labels
            target_label: The label to search for (e.g., "STR")
            
        Returns:
            List of matching text labels
        """
        matching_labels = []
        target_upper = target_label.upper().strip()
        
        for label in text_labels:
            text_upper = label['text'].upper().strip()
            original_upper = label['original_text'].upper().strip()
            
            # Multiple matching strategies
            match_found = False
            
            # 1. Exact match
            if text_upper == target_upper or original_upper == target_upper:
                match_found = True
            
            # 2. Contains match (target in text or text in target)
            elif (target_upper in text_upper or text_upper in target_upper or
                  target_upper in original_upper or original_upper in target_upper):
                match_found = True
            
            # 3. Fuzzy matching for common OCR errors
            elif self.fuzzy_text_match(text_upper, target_upper) or \
                 self.fuzzy_text_match(original_upper, target_upper):
                match_found = True
            
            # 4. Remove common OCR noise and try again
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text_upper)
            cleaned_target = re.sub(r'[^A-Z0-9]', '', target_upper)
            if cleaned_text == cleaned_target and len(cleaned_text) > 0:
                match_found = True
            
            if match_found:
                matching_labels.append(label)
        
        return matching_labels
    
    def fuzzy_text_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Simple fuzzy matching for OCR errors
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold
            
        Returns:
            True if texts are similar enough
        """
        if len(text1) == 0 or len(text2) == 0:
            return False
        
        # Simple character-based similarity
        if abs(len(text1) - len(text2)) > 2:
            return False
        
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        similarity = matches / max(len(text1), len(text2))
        
        return similarity >= threshold
    
    def visualize_results(self, image: np.ndarray, door_results, matching_labels: List[Dict], 
                         target_label: str, all_text_labels: List[Dict] = None) -> np.ndarray:
        """
        Visualize detection results on the image with improved annotations
        
        Args:
            image: Original floor plan image
            door_results: YOLOv8 detection results for doors
            matching_labels: List of matching text labels
            target_label: The target label being searched
            all_text_labels: All detected text labels for debugging
            
        Returns:
            Annotated image
        """
        # Convert to PIL Image for better drawing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            except:
                font_large = None
                font_small = None
        
        # Draw door symbol detections (blue boxes)
        if len(door_results) > 0 and door_results[0].boxes is not None:
            boxes = door_results[0].boxes.xyxy.cpu().numpy()
            confidences = door_results[0].boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw thick blue bounding box for door symbol
                draw.rectangle([x1, y1, x2, y2], outline='blue', width=3)
                
                # Add door label with background
                label_text = f'Door {i+1}'
                if font_large:
                    bbox = draw.textbbox((0, 0), label_text, font=font_large)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width, text_height = 60, 15
                
                # Background rectangle for text
                draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], 
                             fill='blue', outline='blue')
                draw.text((x1+5, y1-text_height-2), label_text, 
                         fill='white', font=font_large)
        
        # Draw matching text labels (red boxes)
        for i, label in enumerate(matching_labels):
            x1, y1, x2, y2 = label['bbox']
            
            # Draw thick red bounding box for matching text
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Add match label with background
            match_text = f"{target_label} Match {i+1}"
            conf_text = f"'{label['original_text']}' ({label['confidence']:.2f})"
            
            if font_large:
                bbox = draw.textbbox((0, 0), match_text, font=font_large)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = 80, 15
            
            # Background rectangle for text
            draw.rectangle([x1, y1-text_height*2-10, x1+max(text_width, 100)+10, y1], 
                         fill='red', outline='red')
            draw.text((x1+5, y1-text_height*2-5), match_text, 
                     fill='white', font=font_large)
            draw.text((x1+5, y1-text_height-2), conf_text, 
                     fill='white', font=font_small)
        
        # Optionally draw all text labels in light green for debugging
        if all_text_labels and st.session_state.get('show_all_text', False):
            for label in all_text_labels:
                if label not in matching_labels:  # Don't duplicate matched labels
                    x1, y1, x2, y2 = label['bbox']
                    draw.rectangle([x1, y1, x2, y2], outline='lightgreen', width=1)
                    draw.text((x1, y2+2), f"'{label['original_text']}'", 
                             fill='green', font=font_small)
        
        # Convert back to numpy array
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image
    
    def analyze_floor_plan(self, image: np.ndarray, target_label: str, 
                          confidence_threshold: float = 0.5) -> Dict:
        """
        Complete analysis of the floor plan
        
        Args:
            image: Input floor plan image
            target_label: Text label to search for
            confidence_threshold: Confidence threshold for door detection
            
        Returns:
            Dictionary containing all analysis results
        """
        # Detect door symbols
        door_results, door_count = self.detect_door_symbols(image, confidence_threshold)
        
        # Extract all text labels
        all_text_labels = self.extract_text_labels(image)
        
        # Find matching labels
        matching_labels = self.find_matching_labels(all_text_labels, target_label)
        
        # Create visualization
        annotated_image = self.visualize_results(image, door_results, matching_labels, 
                                               target_label, all_text_labels)
        
        # Prepare results
        results = {
            'door_count': door_count,
            'matching_label_count': len(matching_labels),
            'matching_labels': matching_labels,
            'all_text_labels': all_text_labels,
            'annotated_image': annotated_image,
            'door_detections': door_results
        }
        
        return results

def main():
    st.set_page_config(
        page_title="Floor Plan Analyzer", 
        page_icon="🏠", 
        layout="wide"
    )
    
    st.title("🏠 Floor Plan Analysis System with PaddleOCR")
    st.markdown("Upload a floor plan image and analyze architectural elements")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model path input
        model_path = st.text_input(
            "YOLOv8 Model Path", 
            value="best.pt",
            help="Path to your trained YOLOv8 model"
        )
        
        # Confidence threshold
        confidence = st.slider(
            "Detection Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1
        )
        
        # Target label input
        target_label = st.text_input(
            "Target Label to Find", 
            value="STR",
            help="Enter the text label you want to detect (e.g., STR, BR, LR)"
        )
        
        # Debug options
        st.header("Debug Options")
        show_all_text = st.checkbox(
            "Show All Detected Text", 
            value=False,
            help="Display all text found by OCR in light green boxes"
        )
        st.session_state['show_all_text'] = show_all_text
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Floor Plan")
        uploaded_file = st.file_uploader(
            "Choose a floor plan image", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file is not None:
            # Read and display original image
            image_bytes = uploaded_file.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Floor Plan", type="primary"):
                try:
                    # Initialize analyzer
                    with st.spinner("Loading model and initializing PaddleOCR..."):
                        analyzer = FloorPlanAnalyzer(model_path)
                    
                    # Perform analysis
                    with st.spinner("Analyzing floor plan..."):
                        results = analyzer.analyze_floor_plan(
                            image, target_label, confidence
                        )
                    
                    # Store results in session state
                    st.session_state['analysis_results'] = results
                    st.success("Analysis completed!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Please check your model path and ensure the YOLOv8 model file exists.")
                    st.info("Make sure PaddleOCR is properly installed: pip install paddlepaddle paddleocr")
    
    with col2:
        st.header("Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            # Display key metrics
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("🚪 Door Symbols Found", results['door_count'])
            with col2_2:
                st.metric(f"📝 '{target_label}' Labels Found", results['matching_label_count'])
            
            # Display annotated image
            st.subheader("Annotated Image")
            st.image(
                cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB),
                use_column_width=True,
                caption="Red boxes: Target labels, Blue boxes: Door symbols, Green boxes: All text (if enabled)"
            )
            
            # Detailed results
            with st.expander("📊 Detailed Results"):
                st.subheader(f"Matching '{target_label}' Labels:")
                if results['matching_labels']:
                    label_df = pd.DataFrame([
                        {
                            'Original Text': label['original_text'],
                            'Processed Text': label['text'],
                            'Confidence': f"{label['confidence']:.3f}",
                            'Position (x1,y1,x2,y2)': str(label['bbox'])
                        }
                        for label in results['matching_labels']
                    ])
                    st.dataframe(label_df, use_container_width=True)
                else:
                    st.info(f"No '{target_label}' labels found in the image")
                
                st.subheader("All Detected Text Labels:")
                if results['all_text_labels']:
                    all_labels_df = pd.DataFrame([
                        {
                            'Original Text': label['original_text'],
                            'Processed Text': label['text'],
                            'Confidence': f"{label['confidence']:.3f}",
                            'Position': str(label['bbox'])
                        }
                        for label in results['all_text_labels']
                    ])
                    st.dataframe(all_labels_df, use_container_width=True)
                else:
                    st.info("No text labels detected")
            
            # Download results
            st.subheader("📥 Download Results")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Create summary report
                summary_text = f"""Floor Plan Analysis Report
=====================================

Analysis Parameters:
- Target Label: {target_label}
- Confidence Threshold: {confidence}
- OCR Engine: PaddleOCR

Results:
- Door Symbols Detected: {results['door_count']}
- Target Labels Found: {results['matching_label_count']}
- Total Text Labels: {len(results['all_text_labels'])}

Matching Labels:
"""
                for i, label in enumerate(results['matching_labels'], 1):
                    summary_text += f"{i}. '{label['original_text']}' -> '{label['text']}' at {label['bbox']} (conf: {label['confidence']:.3f})\n"
                
                summary_text += f"\nAll Detected Text:\n"
                for i, label in enumerate(results['all_text_labels'], 1):
                    summary_text += f"{i}. '{label['original_text']}' (conf: {label['confidence']:.3f})\n"
                
                st.download_button(
                    "📄 Download Summary Report",
                    summary_text,
                    file_name=f"floor_plan_analysis_{target_label}.txt",
                    mime="text/plain"
                )
            
            with col_d2:
                # Convert annotated image to bytes for download
                _, buffer = cv2.imencode('.png', results['annotated_image'])
                img_bytes = buffer.tobytes()
                
                st.download_button(
                    "🖼️ Download Annotated Image",
                    img_bytes,
                    file_name=f"annotated_floor_plan_{target_label}.png",
                    mime="image/png"
                )
        else:
            st.info("Upload an image and click 'Analyze Floor Plan' to see results here.")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Instructions:
        1. **Install Dependencies**: 
           ```bash
           # For CPU version (recommended for most users)
           pip install paddlepaddle paddleocr ultralytics streamlit opencv-python pillow pandas matplotlib numpy
           
           # Or for GPU version (if you have CUDA)
           pip install paddlepaddle-gpu paddleocr ultralytics streamlit opencv-python pillow pandas matplotlib numpy
           ```
           
        2. **Troubleshooting PaddleOCR**:
           - If you get "Unknown argument" errors, update PaddleOCR: `pip install --upgrade paddleocr`
           - If installation fails, try: `pip install paddlepaddle==2.4.2 paddleocr==2.6.1.3`
           - For M1/M2 Macs: Use `pip install paddlepaddle` (CPU version only)
           
        3. **Model Setup**: Ensure your trained YOLOv8 model file is accessible and enter the correct path
        3. **Upload Image**: Choose a floor plan image in supported formats (PNG, JPG, etc.)
        4. **Set Parameters**: 
           - Adjust detection confidence threshold (0.1-1.0)
           - Enter the target label you want to find (e.g., "STR", "BR", "LR")
           - Enable "Show All Detected Text" for debugging
        5. **Analyze**: Click the "Analyze Floor Plan" button to start processing
        6. **Review Results**: 
           - View metrics for door symbols and target labels found
           - Examine the annotated image with highlighted detections
           - Download detailed reports and annotated images
        
        ### Color Coding:
        - **🔴 Red boxes**: Target text labels (e.g., "STR") with original text and confidence
        - **🔵 Blue boxes**: Door symbols detected by YOLO
        - **🟢 Light Green boxes**: All detected text (debug mode)
        
        ### Improvements in this version:
        - **PaddleOCR Integration**: More accurate text detection
        - **Image Preprocessing**: Enhanced contrast and noise reduction
        - **Better Text Matching**: Fuzzy matching for OCR errors
        - **Improved Visualization**: Text labels show original text and confidence
        - **Debug Mode**: Option to see all detected text
        - **Duplicate Removal**: Prevents duplicate text detections
        """)

if __name__ == "__main__":
    # For running as a script (non-Streamlit)
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze floor plan images")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 model")
    parser.add_argument("--image", required=True, help="Path to floor plan image")
    parser.add_argument("--label", default="STR", help="Target label to find")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", help="Output path for annotated image")
    
    # Check if running in Streamlit
    try:
        import streamlit.web.cli as stcli
        # If we get here, we're likely running via Streamlit
        main()
    except ImportError:
        # Running as regular script
        args = parser.parse_args()
        
        # Load image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image from {args.image}")
            exit(1)
        
        # Initialize analyzer and run analysis
        analyzer = FloorPlanAnalyzer(args.model)
        results = analyzer.analyze_floor_plan(image, args.label, args.confidence)
        
        # Print results
        print(f"Analysis Results for '{args.label}':")
        print(f"Door Symbols Found: {results['door_count']}")
        print(f"Target Labels Found: {results['matching_label_count']}")
        print(f"Total Text Labels: {len(results['all_text_labels'])}")
        
        # Print all detected text for debugging
        print("\nAll detected text:")
        for i, label in enumerate(results['all_text_labels'], 1):
            print(f"{i}. '{label['original_text']}' (conf: {label['confidence']:.3f})")
        
        # Save annotated image
        output_path = args.output or f"annotated_{args.label}_{args.image}"
        cv2.imwrite(output_path, results['annotated_image'])
        print(f"Annotated image saved to: {output_path}")