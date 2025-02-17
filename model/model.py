import torch
import base64
import io
from PIL import Image
from packages.util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img

class Model:
    def __init__(self, **kwargs):
        self._environment = kwargs["environment"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(self):
        self.som_model_path = self._environment.get("SOM_MODEL_PATH")
        self.caption_model_name = self._environment.get("CAPTION_MODEL_NAME")
        self.caption_model_path = self._environment.get("CAPTION_MODEL_PATH")

        self.som_model = get_yolo_model(model_path=self.som_model_path)
        self.caption_model_processor = get_caption_model_processor(model_name=self.caption_model_name, model_name_or_path=self.caption_model_path, device=self.device)

    def predict(self, request: dict):
        image_base64 = request.get("image")
        box_threshold = request.get('box_threshold', 0.1)
        iou_threshold = request.get('iou_threshold', 0.7)

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'paragraph': False, 'text_threshold': 0.8})
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text, iou_threshold=iou_threshold)
        
        width, height = image.size
        for item in parsed_content_list:
            bbox = item['bbox']
            item['bbox'] = [
                int(bbox[0] * width), 
                int(bbox[1] * height), 
                int(bbox[2] * width),  
                int(bbox[3] * height)  
            ]

        parsed_content_dict = {str(i): item for i, item in enumerate(parsed_content_list)}
        return { "image": dino_labled_img, "parsed_content_list": parsed_content_dict }