

import torchvision.models.detection as detection

def get_model(num_classes):
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.FastRCNNPredictor(in_features, num_classes)
    return model
