import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import torch.nn.functional as F
import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matplotlib.use('Agg')


def vizualize_imputation_batch(preds,targets,masks,out_fp):
    '''
    input:
    - preds List[(batch_size,channels,seq_len)]:
        predicted values for masked datapoints
    - targets List[(batch_size,channels,seq_len)]:
        raw sequence
    - masks List[(batch_size,channels,seq_len)]:
        boolean array indicating which indices are masked
    '''
    if not isinstance(preds, list) or \
        not isinstance(targets, list) or \
            not isinstance(masks, list):
        raise ValueError("Input must be lists")

    if not isinstance(preds[0], np.ndarray) or \
        not isinstance(targets[0], np.ndarray) or \
            not isinstance(masks[0], np.ndarray):
        raise ValueError("Input must be list of numpy arrays")

    # for now we just pick the first 8 (at most) batch for each epoch
    batch_size, dim, seq_len = preds[0].shape
    row, col = min(batch_size, 8), dim
    fig, ax = plt.subplots(row, col, figsize=(15, 15))

    fig.suptitle('Imputation of Missing Values', fontsize=20)
    fig.subplots_adjust(top=0.9)
    fig.supxlabel('Dimension', fontsize=14)
    fig.supylabel('Sample', fontsize=14)
    for i in range(row):
        for j in range(col):
            M = masks[0][i,j,:]
            P = preds[0][i,j,:]
            T = targets[0][i,j,:]

            pred_y = P[M]
            pred_x = np.arange(0, seq_len)[M]

            target_y = T[M]  # same as input
            target_x = np.arange(0, seq_len)[M]

            ax[i][j].plot(T, color='r')
            ax[i][j].scatter(x=target_x, y=target_y, color='grey', s=10)
            ax[i][j].scatter(x=pred_x, y=pred_y, color='b', s=10)

            if i == 0:
                ax[i][j].set_title(f"Dim {j}")

    fig.legend(['True, input', 'True masked', 'Predicted Values'],
               loc='upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_fp)
    plt.close()


def _convert_to_image(fig):
    # Convert the plot to numpy array
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)

    return image_array


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def plot_pose_2d(im, bbox, kpts):
    im = cv2.rectangle(
        im,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    plot_skeleton_kpts(im, kpts)

    return im


def plot_skeleton_kpts(im, kpts, steps=3):
    palette = np.array([
        [255, 128, 0], [255, 153, 51], [255, 178, 102],
        [230, 230, 0], [255, 153, 255], [153, 204, 255],
        [255, 102, 255], [255, 51, 255], [102, 178, 255],
        [51, 153, 255], [255, 153, 153], [255, 102, 102],
        [255, 51, 51], [153, 255, 153], [102, 255, 102],
        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
        [255, 255, 255]
    ])

    skeleton = [
        [1, 2], [2, 3], [3, 6], [3, 4],
        [6, 7], [7, 8], [8, 9], [5, 7],
        [4, 5], [10, 11],
        [3, 10],
        [7, 11],
        [10, 12], [12, 14], [14, 16],
        [11, 13], [13, 15], [15, 17]
    ]

    pose_limb_color = palette[
        [9, 9, 9, 9, 7, 7, 7, 7, 0, 0, 10, 11, 16, 16, 16, 13, 13, 13]]
    pose_kpt_color = palette[
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]
    radius = 2

    num_kpts = len(kpts) // steps
    # plot keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        # Confidence of a keypoint has to be greater than 0.5
        if conf > 0.5:
            cv2.circle(
                im,
                (int(x_coord), int(y_coord)),
                radius, (int(r), int(g), int(b)), -1)
    # plot skeleton
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        # For a limb, both the keypoint confidence must be greater than 0.5
        if conf1 > 0.5 and conf2 > 0.5:
            cv2.line(
                im,
                pos1, pos2,
                (int(r), int(g), int(b)), thickness=1)


def plot_body(ax, points, color, label):
    # Define the connections between the joints
    connections = [
        (0, 2), (2, 4),  # left arm
        (1, 3), (3, 5),  # right arm
        (6, 8), (8, 10),  # left leg
        (7, 9), (9, 11),  # right leg
        (0, 6), (1, 7), (0, 1), (6, 7),  # body
    ]

    # Plot the skeleton joints
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, marker='o', s=2)

    for connection in connections:
        joint1 = points[connection[0]]
        joint2 = points[connection[1]]
        ax.plot([joint1[0], joint2[0]],
                [joint1[1], joint2[1]],
                [joint1[2], joint2[2]], c=color)

    ax.plot([], [], c=color, label=label)


def plot_axis(ax, c, x, y, z):
    ax.quiver(*c, *x, color='r', length=1000,
              arrow_length_ratio=0.1, label='X')
    ax.quiver(*c, *y, color='g', length=1000,
              arrow_length_ratio=0.1, label='Y')
    ax.quiver(*c, *z, color='b', length=1000,
              arrow_length_ratio=0.1, label='Z')


def plot_pose_3d(pose_3d, axis=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-2000, 2000])
    ax.set_zlim3d([-1000, 1000])

    rot = Rotation.from_euler('zyx', np.array([0, 0, -90]),
                              degrees=True).as_matrix()

    pose_3d = (rot @ pose_3d.T).T
    plot_body(ax, pose_3d, 'darkorange', "estimation")

    if axis:
        c, x, y, z = axis
        c = (rot @ c.T).T
        x = (rot @ x.T).T
        y = (rot @ y.T).T
        z = (rot @ z.T).T

        # Plot the axes
        plot_axis(ax, c, x, y, z)

    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Human Skeleton')
    ax.legend()

    fig.tight_layout()
    image = _convert_to_image(fig)
    plt.close()

    return image


def pca(image_feats_list, dim=3, fit_pca=None, max_samples=None):
    # borrowed from https://github.com/mhamilton723/FeatUp
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(
                tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        tensor = tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0)
        return tensor.detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of
    # samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(
            x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


def plot_feats(image, lr):
    assert len(image.shape) == len(lr.shape) == 3
    [lr_feats_pca], _ = pca([lr.unsqueeze(0)])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image.detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Original Features")
    remove_axes(ax)

    image = _convert_to_image(fig)
    plt.close()

    return image


def plot_series(acc_x, acc_y, acc_z, frame_num):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.72))

    accs = [acc_x, acc_y, acc_z]
    dirs = ["roll", "pitch", "yaw"]
    for ax, acc, dir in zip(axes, accs, dirs):
        ax.set_xticks(list(range(0, frame_num+1, 50)))
        ax.set_yticks(list(range(-50, 50, 10)))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlim(0, frame_num)
        ax.set_ylim(-50, 50)
        ax.plot(acc)
        if ax == axes[-1]:
            ax.set_xlabel('Time (0.1 s)', fontsize=8)
        ax.set_ylabel(f'{dir}', fontsize=8)

    fig.suptitle('Angular Acceleration Series Plot', size=10)
    fig.tight_layout()

    image = _convert_to_image(fig)
    plt.close()

    return image



def visualize_dataset(ds, out_dir):
    dims,timesteps = ds[0][0].shape
    feats = ["avg","med","range"]
    stats = {f"dim_{str(i)}_{feat}":[] for feat in feats for i in range(dims)}
    stats['lbls'] = []

    for input,lbl in ds:
       
        for i in range(dims):
     
            stats[f"dim_{str(i)}_avg"].append(np.mean(input[i,:]))
            stats[f"dim_{str(i)}_med"].append(np.median(input[i,:]))
            stats[f"dim_{str(i)}_range"].append(np.max(input[i,:])-np.min(input[i,:]))
        stats['lbls'].append(lbl.item())
    
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3) ##aggregate by number of statistics to reprot
    fig.tight_layout()
    axs[0].set_title('Average (by dimension)')
    axs[1].set_title('Median (by dimension)')
    axs[2].set_title('Range (by dimension)')
    
   
    for i in range(dims):
        axs[0].hist(stats[f"dim_{str(i)}_avg"],bins=50,label=f"dim_{i}")
        axs[1].hist(stats[f"dim_{str(i)}_med"],bins=50,label=f"dim_{i}")
        axs[2].hist(stats[f"dim_{str(i)}_range"],bins=50,label=f"dim_{i}")
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")
    plt.savefig(out_dir + "_classstats.png")
    plt.clf()

    fig, ax = plt.subplots()
    c = Counter(stats['lbls'])
   
    sizes = [v for k,v in c.items()]
    labels = [k for k,v in c.items()]
    ax.set_title("Class Distribution")
    ax.pie(sizes, labels=labels, autopct='%.2f%%' )
    plt.savefig(out_dir + "_classprop.png")
    plt.clf()

    plt.close('all')
    return 


def visualize_dataloader(dl, out_fp):
    batch_class_counts = {}
    ##"class:..count."

    print("VISUALIZING DATALOADER.")
    for i, batch in enumerate(tqdm.tqdm(dl)):

        class_ids, class_counts = np.unique(batch[1], return_counts=True)
        id2class ={k:v for k,v in zip(class_ids,class_counts)}
        for id in class_ids: 
            if id not in batch_class_counts:
                batch_class_counts[id] = [0]*len(dl)
            batch_class_counts[id][i] = id2class[id]
        
    ## class i : ...,batch n count,... 
    _, ax = plt.subplots(nrows=1,ncols=1)
    bottom = np.zeros(len(dl))
    ##batch 
    for class_lbl,batch_counts in batch_class_counts.items():
        inds = np.arange(len(dl))
        ax.bar(inds,batch_counts, label=str(class_lbl),bottom=bottom,width = 0.35)
        bottom += np.array(batch_counts)

        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of samples in batch", fontsize=12)

    ax.set_title("Distribution of Classes Per Batch")
    plt.legend()
    plt.savefig(out_fp)
    plt.clf()

    plt.close('all')
    return batch_class_counts


def plot_confusion_matrix(y_preds, y_trues, out_fp):
    """
    This function plots the confusion matrix.
    """
    overlap = min(len(y_preds), len(y_trues))
    yt = np.array(y_preds)[:overlap]
    yp = np.array(y_trues)[:overlap]
    cm = confusion_matrix(yp, yt)
    cm = ConfusionMatrixDisplay(cm)

    cm.plot()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(out_fp)
    plt.close()
