import sim
import os
import sys
import random
import spacy
import json
import torch
import time
import argparse
import numpy as np
import open3d as o3d
import scipy.io as scio
from PIL import Image
from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
spacy_eng = spacy.load("en_core_web_sm")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/Text_Guided_RGBD_grasp_Generation_simulation/dataset')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.1, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

class Vocabulary:
    def __init__(self, freq_threshold=1):
        # Initialize 2 dictionary: index to string and string to index
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # Threshold for add word to dictionary
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    def read_vocab(self, file_name):
        """
        Load created vocabulary file and replace the 'index to string' and 'string to index' dictionary
        """
        vocab_path = open(file_name, 'r')
        vocab = json.load(vocab_path)
        new_itos = {int(key): value for key, value in vocab['itos'].items()}

        self.itos = new_itos
        self.stoi = vocab['stoi']

    def create_vocab(self, file_name):
        # create json object from dictionary
        vocab = json.dumps({'itos': self.itos,
                            'stoi': self.stoi})

        # open file for writing, "w"
        f = open(file_name, "w")

        # write json object to file
        f.write(vocab)

        # close file
        f.close()

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def get_text_data(descript):
    data_path = os.path.join(cfgs.root, 'text_data/vocab.json')
    vocab = Vocabulary()
    vocab.read_vocab(data_path)
    numericalized_descript = [vocab.stoi["<SOS>"]]
    numericalized_descript += vocab.numericalize(descript)
    numericalized_descript.append(vocab.stoi["<EOS>"])

    padding = [vocab.stoi["<PAD>"] for _ in range(50 - len(numericalized_descript))]
    numericalized_descript.extend(padding)

    return torch.tensor(numericalized_descript)

def get_net(check_point):
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(check_point)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir, color, pred, numericalized_descript):
    color = Image.fromarray(np.uint8(color))
    resized_color_image = color.resize((1280, 720), Image.LANCZOS)
    resized_pred = Image.fromarray(pred).resize((1280, 720), Image.NEAREST)

    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

    meta = scio.loadmat(os.path.join(data_dir))
    intrinsic = meta['intrinsic_matrix']

    factor_depth = 2020

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    points = create_point_cloud_from_depth_image(np.array(resized_pred), camera, organized=False)

    cloud_masked = points * 20
    color_masked = colors
    
    idxs = np.load('indx.npy')
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled
    cloud_sampled = torch.from_numpy(np.concatenate((cloud_sampled.squeeze(0), color_sampled), axis=1))
    cloud_sampled = torch.tensor(cloud_sampled, dtype=torch.float32)
    end_points['text_input'] = torch.tensor(numericalized_descript).to(device).unsqueeze(0)
    end_points['point_clouds'] = cloud_sampled.to(device).unsqueeze(0)
    end_points['cloud_colors'] = color_sampled
    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        end_points = net(end_points)
        print("--- %s seconds ---" % (time.time() - start_time))
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(np.array(gg_array))
    return gg

def grasp_parameters(data_dir, checkpoint, color, depth, descript):
    net = get_net(checkpoint)
    end_points, cloud = get_and_process_data(data_dir, color, depth, descript)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    gg.nms()
    gg.sort_by_score()
    scores, widths, heights, depths, translations, rotation_matrices = gg.value_predict()
    return translations, rotation_matrices

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

if __name__ == '__main__':
    '''******IMPOTRANT******
    simRemoteApi.start(19999) -- input this line in Lua command of simulation firstly'''

    sim.simxFinish(-1)  # Just in case, close all opened connections
    sim_client = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if sim_client == -1:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()
    else:
        print('Connected to simulation.')
    # Synchronously Running the client and server
    sim.simxSynchronous(sim_client, True);  # Enable the synchronous mode (Blocking function call)
    sim.simxStartSimulation(sim_client, sim.simx_opmode_oneshot)
    sim.simxSynchronousTrigger(sim_client)  # trigger the simulation

    sim_ret, rgb_cam = sim.simxGetObjectHandle(sim_client, "kinect_rgb", sim.simx_opmode_blocking)
    sim_ret, depth_cam = sim.simxGetObjectHandle(sim_client, "kinect_depth", sim.simx_opmode_blocking)

    # Wait for Signal objectNumber
    err, objectNumber = sim.simxGetIntegerSignal(sim_client, 'objectNumber', sim.simx_opmode_streaming)
    while err != sim.simx_return_ok:
        err, objectNumber = sim.simxGetIntegerSignal(sim_client, 'objectNumber', sim.simx_opmode_buffer)
    sim.simxClearIntegerSignal(sim_client, 'objectNumber', sim.simx_opmode_oneshot)
    
    with open(file=os.path.join(cfgs.root, 'text_data/data')) as f:
        for i in f.readlines():
            int_ = 1140
            random.seed(int_)
            np.random.seed(int_)
            torch.manual_seed(int_)
            descript = get_text_data(i)
            err2, sendImages = sim.simxGetStringSignal(sim_client, 'sendImages', sim.simx_opmode_streaming)
            while err2 != sim.simx_return_ok:
                err2, sendImages = sim.simxGetStringSignal(sim_client, 'sendImages', sim.simx_opmode_buffer)
            print(r'sendImages: %s' % (sendImages))
            date_dir = '/dataset/meta.mat'
            check_point = f'/logs/checkpoint4.tar'
            sim.simxClearStringSignal(sim_client, 'sendImages', sim.simx_opmode_oneshot)
            # Acquire RGB Image
            
            sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(sim_client, rgb_cam, 0, sim.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float_)
            color_img[color_img < 0] += 255
            color_img = np.flipud(color_img)
            color_img = color_img.astype(np.uint8)

            sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(sim_client, depth_cam,
                                                                           sim.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.flipud(depth_img)
            depth_img = depth_img * 255
            t, r = grasp_parameters(date_dir, check_point, color_img, depth_img, descript)
            r = np.array([[r[0][1], r[0][0],  r[0][2]], [r[1][1], r[1][0], r[1][2]], [r[2][1], r[2][0], r[2][2]]])
            Y, X, Z = t[0], t[1], t[2]
            sim.simxPauseCommunication(sim_client, True)
            sim.simxSetFloatSignal(sim_client, 'graspCenterX', X, sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'graspCenterY', Y, sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'graspCenterZ', Z, sim.simx_opmode_oneshot)

            sim.simxSetFloatSignal(sim_client, 'rx0', r[0][1], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'ry0', r[0][0], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'rz0', r[0][2], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'rx1', r[1][1], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'ry1', r[1][0], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'rz1', r[1][2], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'rx2', r[2][1], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'ry2', r[2][0], sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(sim_client, 'rz2', r[2][2], sim.simx_opmode_oneshot)
            sim.simxSetStringSignal(sim_client, 'sendGrasps', 'start', sim.simx_opmode_oneshot)
            sim.simxPauseCommunication(sim_client, False)




