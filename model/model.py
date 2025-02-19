import torch
import base64
import io
import os
import time
import easyocr
from PIL import Image
from util.utils import check_ocr_box, get_som_labeled_img
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

class Model:
    def __init__(self, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._data_dir = kwargs["data_dir"]

    def load(self):
        # Load SOM model
        som_model_path = os.path.join(self._data_dir, "omniparser", "weights", "icon_detect", "model.pt")
        self.som_model = YOLO(som_model_path)

        # Load caption model        
        caption_model_path = os.path.join(self._data_dir, "omniparser", "weights", "icon_caption_florence")
        torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        caption_model = AutoModelForCausalLM.from_pretrained(caption_model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(self.device)
        
        # Load caption model processor
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        self.caption_model_processor = {'model': caption_model, 'processor': processor}

        # Load easyocr reader
        easyocr_dir = os.path.join(self._data_dir, "easyocr", "easyocr", "model")
        self.easyocr_reader = easyocr.Reader(['en'], gpu=self.device == 'cuda', model_storage_directory=easyocr_dir, download_enabled=False)
    
    def preprocess(self, request: dict):
        image_base64 = request.get("image")
        image_bytes = base64.b64decode(image_base64)
        self._image = Image.open(io.BytesIO(image_bytes))
        return request

    def predict(self, request: dict):
        # Get parameters from request
        box_threshold = request.get('box_threshold', 0.1)
        iou_threshold = request.get('iou_threshold', 0.7)
        
        # Calculate box overlay ratio
        box_overlay_ratio = max(self._image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        start_time2 = time.time()
        # Get OCR text and bounding boxes
        (text, ocr_bbox), _ = check_ocr_box(self._image, self.easyocr_reader, easyocr_args={'paragraph': False, 'text_threshold': 0.8}, display_img=False, output_bb_format='xyxy')
        print(f"OCR time: {time.time() - start_time2}")

        start_time3 = time.time()
        # Get SOM labeled image
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(self._image, self.som_model, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text, iou_threshold=iou_threshold)
        print(f"SOM time: {time.time() - start_time3}")
        
        start_time4 = time.time()
        # Convert bounding box coordinates to absolute values
        width, height = self._image.size
        for item in parsed_content_list:
            bbox = item['bbox']
            item['bbox'] = [
                int(bbox[0] * width), 
                int(bbox[1] * height), 
                int(bbox[2] * width),  
                int(bbox[3] * height)  
            ]
        
        # Create parsed content dictionary
        parsed_content_dict = {str(i): item for i, item in enumerate(parsed_content_list)}
        print(f"Bounding box conversion time: {time.time() - start_time4}")

        return { "image": dino_labled_img, "parsed_content_list": parsed_content_dict }