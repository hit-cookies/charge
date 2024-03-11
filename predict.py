import cv2
from PIL import Image
import numpy as np
import os
import time
 
import torch
from torch import nn
# from torchvision.transforms import v2 as T
import torchvision.transforms as T
from main import get_args_parser as get_main_args_parser
from models import build_model
 
torch.set_grad_enabled(False)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))
 
# 图像数据处理
transform = T.Compose([
    T.Resize((576, 704), antialias=True),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3, 1, 1)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=False):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL =['circle','N/A']
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl] * 100, 2))
 
        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)
    # if imshow:
    #     cv2.imshow('detect', opencvImage)
        # cv2.waitKey(0)
 
    if imwrite:
        if not os.path.exists("./result/pred"):
            os.makedirs('./result/pred')
        cv2.imwrite('./result/pred/{}'.format(save_name), opencvImage)
 
# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
 
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b
 
def load_model(model_path , args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path) # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model
 
# 图像的推断
def detect(im, model, transform, prob_threshold=0.999):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
 
 
    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)
   
    # keep only predictions with 0.7+ confidence
    # print(outputs['pred_logits'].softmax(-1)[0, :, :-1])
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold


    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()
 
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    end = time.time()
    return probas[keep], bboxes_scaled, end - start

def get_transform(train):
    transforms = []
    transforms.append(T.Grayscale())
    transforms.append(T.Resize((576, 704), antialias=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
def predict(im,model):
    # img = transform(im).unsqueeze(0)
    #
    # # propagate through the model
    # img = img.to(device)
    # output = model(img)
    # output['pred_logits'] = torch.sigmoid(output['pred_logits'])
    # # h = 576
    # # w = 704
    #
    # _, h, w = img[0].shape
    #
    #
    # img = torch.permute(img[0], (1, 2, 0))
    #
    # img = img.cpu().detach().numpy().copy()
    #
    # print(img.shape)
    #
    # pred_logits = output['pred_logits']
    # pred_logits = torch.sigmoid(pred_logits)

    scores, boxes, waste_time, _ = detect(im, model, transform)
    x1, y1, x2, y2 = int(boxes[0,0]), int(boxes[0, 1]), int(boxes[0,2]), int(boxes[0,3])

    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0),3)


    cv2.imshow('sdf', img)
    cv2.waitKey(-1)

    # if pred_logits[0,0,0] > 0.5:
    #     pt = output['pred_boxes'].cpu().detach().numpy()
    #
    #     ptx = int(pt[0, 0, 0] * w)
    #     pty = int(pt[0, 0, 1] * h)
    #
    #     ptw = int(pt[0, 0, 2] * w)
    #     pth = int(pt[0, 0, 3] * h)
    #
    #     print(ptx, pty)
    #     cv2.rectangle(img, (int(ptx-ptw/2), int(pty-pth/2)), (int(ptx+ptw/2), int(pty+pth/2)), (255,0,0),3)
    #     cv2.circle(img, (ptx, pty), 3, (0, 255, 0), thickness=-1)
    #
    # cv2.imshow('sdf', img)
    # cv2.waitKey(-1)

if __name__ == "__main__":
    
    main_args = get_main_args_parser().parse_args()
    # 加载模型
    dfdetr = load_model('/home/bing/charge-coco/exps/r50_deformable_detr/checkpoint0005.pth',main_args) # <--修改为自己加载模型的路径
 
    files = os.listdir("data/coco/mytestdata") # <--修改为待预测图片所在文件夹路径

    cn = 0
    waste=0
    for file in files:
        img_path = os.path.join("/home/bing/charge-coco/data/coco/mytestdata/", file) # <--修改为待预测图片所在文件夹路径
        im = Image.open(img_path)

        # predict(im, dfdetr)
        scores, boxes, waste_time = detect(im, dfdetr, transform)
        plot_result(im, scores, boxes, save_name=file, imshow=True, imwrite=False)
        # print("{} [INFO] {} time: {} done!!!".format(cn,file, waste_time))
        #
        # cn+=1
        # waste+=waste_time
        # waste_avg = waste/cn
        # print(waste_avg)
