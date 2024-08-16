import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import Dataset, random_split

from tools.loss import compute_loss
from tools.plot import vizualize_imputation_batch


from tools.dataset import ClassificationDataset,ImputationDataset,Standardize
from tools.features_loader import PoseFeaturesDataset,ImputationDataset_IRV
from tools.helper import noise_mask,feat_noise_mask
from torchvision import transforms


def load_onnx_model(weight_path):
    if not os.path.exists(weight_path):
        assert False, "Model is not exist in {}".format(weight_path)

    session = ort.InferenceSession(
        weight_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    return session


def get_uncalibrated(pts, K):
    """
    Generate the uncalibrated 3D points.

    Args:
        pts : numpy.ndarray (12, 2)
            Points at the image plane.
        K : numpy.ndarray (3, 3)
            The indices of the base pose for each sample.

    Returns:
        numpy.ndarray (12, 3):
            Points at the camera coordinate system.
    """
    pts_homo = np.concatenate((pts, np.ones_like(pts[..., :1])), axis=-1)
    pts_homo = pts_homo.reshape(-1, 3)
    pts_cam = (np.linalg.inv(K) @ pts_homo.T).T
    pts_cam = pts_cam.reshape(-1, 3)

    return pts_cam[:, :2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         origin_size,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * origin_size
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def train_test_split(dataset: Dataset, train_frac=0.8):
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def ttv_split(dataset: Dataset, train_frac=0.8, val_frac=0.2):
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])
    val_size = int(val_frac * train_size)
    train_final_size = train_size - val_size
    val, train_final = random_split(train, [val_size, train_final_size])
    return train_final, val, test


def train_val_split(train_dataset: Dataset, val_frac=0.2):
    train_size = len(train_dataset)
    val_size = int(val_frac * train_size)
    train, val = random_split(train_dataset, [train_size - val_size, val_size])
    return train, val




def get_IRV_dataset(data_cfg,task="classification"):
    
    assert os.path.exists(data_cfg.train_data_path), \
        f"Data file path '{data_cfg.train_data_path}' does not exist."
    assert os.path.exists(data_cfg.test_data_path), \
        f"Data file path '{data_cfg.test_data_path}' does not exist."

    if task == "imputation":
        dataset_class = ImputationDataset_IRV
    elif task == "classification":
        dataset_class = PoseFeaturesDataset
    else:
        raise ValueError("Invalid task.")
    
    feats = 3 
    if data_cfg.include_2d:
        feats+=24
    if data_cfg.include_3d:
        feats +=36

    transform = transforms.Compose([
        # standardize values (mean center)
        Standardize(list(np.arange(feats))),
    ])
    
    train_dataset = dataset_class(
        data_fp=data_cfg.train_data_path,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.window_size,
        window_overlap=data_cfg.window_overlap,
        include_2d=data_cfg.include_2d,
        include_3d=data_cfg.include_3d,
        transform=transform)

    test_dataset = dataset_class(
        data_fp=data_cfg.test_data_path,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.window_size,
        window_overlap=data_cfg.window_overlap,
        include_2d=data_cfg.include_2d,
        include_3d=data_cfg.include_3d,
        transform=transform)



    return train_dataset, test_dataset
    

def get_UEA_dataset(data_cfg, task="classification"):
    '''
    load a dataset from the uea multivariate time series archive

    '''
    assert os.path.exists(data_cfg.train_data_path), \
        f"Data file path '{data_cfg.train_data_path}' does not exist."
    assert os.path.exists(data_cfg.test_data_path), \
        f"Data file path '{data_cfg.test_data_path}' does not exist."

    if task == "imputation":
        dataset_class = ImputationDataset
    elif task == "classification":
        dataset_class = ClassificationDataset
    else:
        raise ValueError("Invalid task.")

    transform = transforms.Compose([
        # standardize values (mean center)
        Standardize([0, 1, 2]),
    ])

    train_dataset = dataset_class(
        data_fp=data_cfg.train_data_path,
        num_classes=data_cfg.num_classes,
        num_features=data_cfg.num_features,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.window_size,
        transform=transform)

    test_dataset = dataset_class(
        data_fp=data_cfg.test_data_path,
        num_classes=data_cfg.num_classes,
        num_features=data_cfg.num_features,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.window_size,
        transform=transform)

    return train_dataset, test_dataset



def get_dataset(data_cfg, task="classification"):
    IRV = ["PoolData"]
    UEA = ["BasicMotions","Epilepsy","WalkingSittingStanding","ChestMntdAcl"]
    if data_cfg.name in IRV:
        train_dataset,test_dataset = get_IRV_dataset(data_cfg,task)
    elif data_cfg.name in UEA:
        train_dataset,test_dataset = get_UEA_dataset(data_cfg,task)
    else:
        raise Exception("unrecognized dataset.")

    return train_dataset,test_dataset


def get_preds(batch,model,device,task):
    if task == "classification":
        x,lbl = batch
        return model(x.to(device)),lbl.to(device),None
    elif task == "imputation":
        x,target,masks = batch
        return model(x.to(device)),target.to(device),masks.to(device)


def train_epoch(model,optimizer,dl,device,task = "classification"):
    model.train()
    epoch_loss = 0
    total_elems = 0

    for _, batch in enumerate(dl):
    
        pred,target,masks = get_preds(batch,model,device,task)
       
        loss = compute_loss(pred,target,masks) ##if task is classifcation, masks will be null
        
        total_loss = torch.sum(loss) / len(loss)

        epoch_loss += torch.sum(loss).item() 
        total_elems += len(loss)
        
        optimizer.zero_grad()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0) ##helps deal with exploding gradients

        optimizer.step()  

    ##return the average loss for the epoch, given that we mask certain elements
    return epoch_loss / total_elems


def evaluate_epoch(model,dl,device,task= "classification"):
    ##test or validate 
    model.eval() 
    
    epoch_val_loss = 0.0
    epoch_val_count = 0
    epoch_results = {'preds':[],'targets':[],'masks':[]}

    with torch.no_grad():
        for _,batch in enumerate(dl):
            preds,target,mask = get_preds(batch,model,device,task)
        
            loss = compute_loss(preds,target,mask)
            if task == "imputation":
                epoch_results['preds'].append(preds.detach().cpu().numpy())
                epoch_results['targets'].append(target.detach().cpu().numpy())
                epoch_results['masks'].append(mask.detach().cpu().numpy())

            epoch_val_loss += torch.sum(loss).item()
            epoch_val_count += len(loss)

    loss = epoch_val_loss/epoch_val_count
    
    return loss, epoch_results


def test_epoch(model,dl,device,task,batch_size=32,plots=True):
    acc =0
    count =0
    epoch_results = {'preds':[],'targets':[]}
    for _,batch in enumerate(dl):
        pred,target,mask = get_preds(batch,model,device,task)

        if task == "imputation": 
            masked_preds = torch.masked_select(pred,mask)
            masked_trues = torch.masked_select(target,mask)

            acc += (masked_preds == masked_trues).float().sum()
            count += len(masked_trues)

            if plots:
                masked_preds = masked_preds.reshape(-1) 
                masked_preds = masked_preds.detach().cpu().numpy()
                epoch_results['preds'] = epoch_results['preds'] + masked_preds.tolist()

                masked_trues = masked_trues.reshape(-1)
                masked_trues = masked_trues.detach().cpu().numpy()
                epoch_results['targets'] =  epoch_results['targets']  +masked_trues.tolist()
        elif task == "classification":
           
            acc += (torch.argmax(pred, 1) == target).float().sum()
            count += len(target)
            if plots:
                reshaped_preds = np.argmax(pred.detach().cpu().numpy(),1)
                
                if reshaped_preds.shape == (batch_size,):
                   epoch_results['preds'] = epoch_results['preds'] + reshaped_preds.tolist()
        
                epoch_results['targets']= epoch_results['targets']  + target.tolist()
    return acc, count, epoch_results


