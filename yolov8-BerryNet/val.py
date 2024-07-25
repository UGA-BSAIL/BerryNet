import argparse, warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def transformer_opt(opt):
    opt = vars(opt)
    del opt['data']
    del opt['weight']
    return opt
    
def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/yolov8/runs/train/exp3/weights/best.pt', help='training model path')
    # parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/weights/model_blueberry_ASABE/yolov8x_1280.pt', help='training model path')

    # parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/training_result/segmentation/yolov8x_seg/exp/weights/best.pt', help='training model path')
    # parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/training_result/cluster_detection/yolov8n-C2f-faster_1280/exp/weights/best.pt', help='training model path')
    #parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/training_result/flower_detection_tifton/yolov8n_1280/exp/weights/best.pt', help='training model path')
    # parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/training_result/segmentation/yolov8n-p2-seg-640/exp/weights/best.pt', help='training model path')
    
    parser.add_argument('--weight', type=str, default='/blue/cli2/zhengkun.li/sim2real_project/training_result/lincoln/lincoln_50r-s/exp/weights/best.pt', help='training model path')

    # parser.add_argument('--weight', type=str, default='/blue/lift-phenomics/zhengkun.li/yolov8/runs/train/blueberry_dyhead/exp/weights/best.pt', help='training model path')
    # parser.add_argument('--data', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/data/flower_tifton_20240205/blueberry_flower_tifton-1/data.yaml', help='data yaml path')

    
    # parser.add_argument('--data', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/data/dataset/segmentation/berry-seg_test100_manual/data.yaml', help='data yaml path')
    parser.add_argument('--data', type=str, default='/blue/cli2/zhengkun.li/sim2real_project/dataset/lincolnbeet/data.yaml', help='data yaml path')

    # parser.add_argument('--data', type=str, default='/blue/lift-phenomics/zhengkun.li/blueberry_project/data/dataset/segmentation/1-seg-1/data.yaml', help='data yaml path')

    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--batch', type=int, default=1, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='dataset split to use for validation, i.e. val, test or train')
    parser.add_argument('--project', type=str, default='runs/val', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name (project/name)')
    parser.add_argument('--save_txt', action="store_true" , default=True, help='save results as .txt file')
    parser.add_argument('--save_json', action="store_true", help='save results to JSON file')
    parser.add_argument('--save_hybrid', action="store_true", help='save hybrid version of labels (labels + additional predictions)')
    # parser.add_argument('--conf', type=float, default=0.3, help='object confidence threshold for detection (0.001 in val)')
    parser.add_argument('--conf', type=float, default=0.3, help='object confidence threshold for detection (0.001 in val)')
    # parser.add_argument('--iou', type=float, default=0.5, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--iou', type=float, default=0.5, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=3000, help='maximum number of detections per image')
    parser.add_argument('--half', action="store_true", help='use half precision (FP16)')
    parser.add_argument('--dnn', action="store_true", help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', action="store_true", default=True, help='ave plots during train/val')
    parser.add_argument('--rect', action="store_true", help='rectangular val')
    
    return parser.parse_known_args()[0]

class YOLOV8(YOLO):
    '''
    weigth:model path
    '''
    def __init__(self, weight='', task=None) -> None:
        super().__init__(weight, task)
        
if __name__ == '__main__':
    opt = parse_opt()
    
    model = YOLOV8(weight=opt.weight)
    model.val(data=opt.data, **transformer_opt(opt))