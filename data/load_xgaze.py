import os
import torch
import numpy as np
import imageio
import cv2
import pickle as pkl
import random


def image_projection_wo_dist_mat(XX, c2w, ref_pose_idx):
    XXc = np.matmul(XX, c2w[:3, :3]) -  np.matmul(c2w[:3, 3], c2w[:3, :3])
    pts = XXc[:, :2] / XXc[:, 2][:, np.newaxis]

    pts = pts * fc[ref_pose_idx] + cc
    return pts

def multi_world2cam_grid_sample_mat(XX, c2w, ref_pose_idx):
    pts = image_projection_wo_dist_mat(XX, c2w, ref_pose_idx)
    pts = np.array(pts) / np.array([nx//2, ny//2]) - 1
    return pts

def image_projection_wo_dist_mat_torch(XX, c2w, ref_pose_idx, device):
    XXc = torch.matmul(XX, c2w[:3,:3]) - torch.matmul(c2w[:3,3], c2w[:3,:3])
    pts = XXc[:,:2] / XXc[:,2].unsqueeze(-1)
    # print(f'pts.shape:{pts.shape}, \nfc[ref_pose_idx].shape:{fc[ref_pose_idx].shape}')
    pts = pts * torch.Tensor(fc[ref_pose_idx]).to(device) + torch.Tensor(cc).to(device)
    return pts

def multi_world2cam_grid_sample_mat_torch(XX, c2w, ref_pose_idx, device):
    pts = image_projection_wo_dist_mat_torch(XX, c2w, ref_pose_idx, device)
    pts = pts / torch.tensor([nx//2, ny//2]).to(device) - 1
    return pts

# def image_projection_wo_dist(XX, pose, ref_pose_idx):
#     Rc, _ = cv2.Rodrigues(np.array(pose_extrinsics['omc_{}'.format(int(pose))]))
#     Tc = np.array(pose_extrinsics['Tc_{}'.format(int(pose))])

#     XXc = np.matmul(XX, Rc.T) + Tc
#     pts = XXc[:, :2] / XXc[:, 2][:, np.newaxis]

#     pts = pts * fc[ref_pose_idx] + cc
#     # print(Rc,Tc, fc, cc)
#     return pts

# def multi_world2cam_grid_sample(XX, pose):
#     pts = image_projection_wo_dist(XX, pose)
#     pts = np.array(pts) / np.array([nx//2, ny//2]) - 1
#     return pts

def img_string(scan, pose):
    return f"{scan}/cam{str(pose).zfill(2)}.JPG"

def load_scan_data(scan, mode, num_views, cfg, specific_poses = None, fixed = True, shuffle = False):
    
    if mode == 'train':
        poses_idx = random.sample(cam_idx,num_views)
    elif specific_poses is None:
        if fixed:
            poses_idx = cam_idx[cfg.fixed_batch * num_views:(cfg.fixed_batch + 1) * num_views]
        else:
            poses_idx = random.sample(cam_idx, num_views)
    else:
        if not len(specific_poses) == num_views:
            raise ValueError('Poses are invalid.')
        poses_idx = cam_idx[specific_poses]

    if shuffle:
        random.shuffle(poses_idx)

    imgs = []
    poses = []
    for pose in poses_idx:
        # always use max lightning images - these have the minimal amount of pose dependent shadows
        img_name = img_string(scan, pose)
        fname = os.path.join(basedir, img_name)
        img = imageio.imread(fname)

        if cfg.half_res:
            img_half_res = np.zeros(( ny, nx, 3))
            img_half_res[:] = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            img = img_half_res
        imgs.append(img)
        poses.append(pose_extrinsics[f'c2w_{int(pose)}'])

    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)

    return imgs, poses, poses_idx

def setup_xgaze(mode, cfg):
    load_parameters()

    global basedir, split, val_samples, images, light_cond
    basedir = cfg.datadir
    split = pkl.load(open( basedir + f'/{cfg.split}', 'rb'))


    global ny, nx, fc, cc
    if cfg.half_res:
        nx = nx//cfg.res_rate
        ny = ny//cfg.res_rate
        fc = fc/cfg.res_rate
        cc = cc/cfg.res_rate


    return split[mode], ny, nx, fc, cc, 'x_down_y_down_z_cam_dir'



def load_parameters():
    global cam_idx 
    cam_idx = [i for i in range(0, 18)]
    global pose_extrinsics
    pose_extrinsics = {  # Image #1:
        # Image #1:
        'omc_0': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'Tc_0': [0.0, 0.0, 0.0],

        # Image #2:
        'omc_1': [[0.9698392735891082, -0.008270901325641296, -0.24360495806820662], [0.007482661394246074, 0.9999633476147745, -0.0041609140237082965], [0.24363044387475521, 0.0022126044190360875, 0.9698656150204907]],
        'Tc_1': [221.40055372232777, -1.5731657904339034, 31.980068426505735],

        # Image #3:
        'omc_2': [[0.7863902773688491, 0.02457490982746557, -0.6172409622398017], [-0.01106061557973079, 0.9996083334853352, 0.025706855304923826], [0.6176309532744791, -0.013388556070122345, 0.786354088260254]],
        'Tc_2': [582.9710529629532, -22.844393775038416, 320.43184257787124],

        # Image #4:
        'omc_3': [[0.7755602021276123, -0.03949806218966534, -0.6300367258811493], [0.17518091275469772, 0.9723082350232455, 0.1546878919386135], [0.606480024965545, -0.23034018144744633, 0.7610028778713976]],
        'Tc_3': [561.9597220800626, -157.58447391158398, 369.5534503923164],

        # Image #5:
        'omc_4': [[0.9825999161441723, 0.05390649250122117, -0.17773996416023452], [0.03129924419440496, 0.8952123742582904, 0.44453926968007423], [0.18307856812397422, -0.44236737565178846, 0.8779483714049618]],
        'Tc_4': [177.6096275688536, -481.94275227326216, 283.5903282160665],

        # Image #6:
        'omc_5': [[0.9989049242833686, -0.04326729607798294, 0.017801498041956118], [0.030673361090901165, 0.8929380913269138, 0.4491332875403863], [-0.03532841861483183, -0.4480954208062701, 0.8932874099026734]],
        'Tc_5': [-10.013619556629912, -456.9689085248577, 265.38605835611116],

        # Image #7:
        'omc_6': [[0.9019530772769576, 0.04517994128615104, 0.4294641071102524], [-0.13693851015385017, 0.9731083451128499, 0.18522416988226556], [-0.40954668943536865, -0.2258736850036109, 0.8838849402469867]],
        'Tc_6': [-420.0358569418966, -171.7923788742382, 259.90954288588944],

        # Image #8:
        'omc_7': [[0.9034549188351174, -0.015574653729707198, 0.42840009313004496], [0.011507558592957517, 0.9998607911297412, 0.012081988935503852], [-0.42852862883109816, -0.0059856931601699696, 0.9035083761368963]],
        'Tc_7': [-414.70095212370296, 0.18529135395040885, 218.40313014420755],

        # Image #9:
        'omc_8': [[0.9996565637870843, -0.011139348111773325, 0.023720653469998926], [0.02060862864318743, 0.8932867427089053, -0.4490145651601528], [-0.016187615724566565, 0.4493492074368872, 0.8932095223815043]],
        'Tc_8': [-30.07439012373554, 459.89480260041273, 186.44674124248863],

        # Image #10:
        'omc_9': [[0.9703608190091254, -0.05040214282229401, -0.23634615488909538], [-0.06367119597705116, 0.8901291606580766, -0.45123835846368376], [0.23312198466637835, 0.4529124654335196, 0.860537296646775]],
        'Tc_9': [208.8877234017151, 441.51126355535337, 212.89337892226843],

        # Image #11:
        'omc_10': [[0.7742489492229995, -0.0696664581102551, -0.629035093807534], [-0.2921085346500205, 0.8423840347001444, -0.45283743448054686], [0.5614367004495688, 0.5343554273109593, 0.6318647867159739]],
        'Tc_10': [569.501849456624, 424.81382471370455, 421.42088877272647],

        # Image #12:
        'omc_11': [[0.34157472148589485, -0.06295367652472443, -0.9377438585529919], [0.0658221510385165, 0.9969066022527823, -0.04294963116809793], [0.9375468810008895, -0.04705380958895134, 0.3446618414165842]],
        'Tc_11': [844.3138574366806, 48.2674946199551, 756.4920827467118],

        # Image #13:
        'omc_12': [[0.6250756419679933, -0.19226107344902638, -0.7565157773996052], [0.6293823011911042, 0.697403106290454, 0.3427926870337463], [0.46169076313663804, -0.6904089997766931, 0.5569354112117983]],
        'Tc_12': [716.041279620078, -338.4725933497391, 849.1304313456653],

        # Image #14:
        'omc_13': [[0.9983192122010462, 0.032440152283014556, -0.04802485887680596], [0.009636848253201056, 0.7242050571299102, 0.6895173430619479], [0.057147893277242325, -0.6888212190019396, 0.7226752012810835]],
        'Tc_13': [53.42483923140793, -646.4838434270105, 539.1087085714361],

        # Image #15:
        'omc_14': [[0.7809554621739873, 0.03288192944746085, 0.6237205662926517], [-0.441918154346149, 0.7347861487591912, 0.5145849399768019], [-0.4413806871170363, -0.6775013611111402, 0.5883663779760406]],
        'Tc_14': [-579.1991528272825, -507.0731519086221, 828.6182606673376],

        # Image #16:
        'omc_15': [[0.5576138145013221, 0.07101464249734504, 0.8270572860620111], [-0.05277587298219646, 0.9973511165258926, -0.05005454620234178], [-0.8284211133887476, -0.015737563837188336, 0.5598846202354716]],
        'Tc_15': [-780.0477351457396, 52.50352781272512, 560.8080186215651],

        # Image #17:
        'omc_16': [[0.8938667769839744, -0.043001897441954345, 0.4462656404213452], [0.278059686163717, 0.8339942499010018, -0.4765882940889343], [-0.35168877709534163, 0.5500949263067757, 0.7574368462893301]],
        'Tc_16': [-436.9051119350867, 452.7480705123046, 304.1090086518881],

        # Image #18:
        'omc_17':  [[0.9971800904395784, 0.006545799442714842, -0.07475974679310951], [-0.054460308495951223, 0.7485142988847291, -0.6608785207310253], [0.051632761202940886, 0.6630863419455706, 0.7467600425141947]],
        'Tc_17': [109.78472957158333, 616.3677403386008, 349.0875152926945],

    }
    for i in range(0, 18):
        pose = i
        w2c = np.array(pose_extrinsics['omc_{}'.format(pose)])
        c2w = w2c.T
        translation = np.array([pose_extrinsics['Tc_{}'.format(pose)]])
        c2w = np.hstack((c2w, np.matmul(c2w, - translation.T)))
        c2w = np.vstack((c2w, [0, 0, 0, 1]))
        pose_extrinsics[f'c2w_{pose}'] = c2w

    # Focal length:
    global fc
    fc = np.array([[13200.70131101025, 13192.49643871068],
                  [13303.297441332066, 13289.773010770674],
                  [13202.447504723803, 13197.8614274423],
                  [13073.634346792553, 13071.096087185497],
                  [13868.253087403722, 13856.275256250801],
                  [13955.654490144081, 13942.033759177113],
                  [14025.849893138095, 14010.600985919233],
                  [13220.644426377807, 13217.09834057157],
                  [13960.810303292888, 13936.87005052997],
                  [13976.934945971894, 13931.733326108262],
                  [13371.652735350852, 13361.160302847506],
                  [13532.388711357731, 13519.037208138094],
                  [14008.281277510245, 13981.885542885882],
                  [13552.813628151382, 13549.661562953488],
                  [14164.237353620741, 14163.287660930266],
                  [13310.748461063455, 13322.183084755576],
                  [12802.929140491959, 12806.864869977968],
                  [14048.261040714882, 14049.871888688402]])

    # Principal point:
    global cc
    cc = np.array([2000, 2000])

    # Image size:
    global nx, ny
    nx = 4000
    ny = 4000

def load_cam_path():
    global camera_path
    poses = [1,3,5,7,1]
    camera_path = []

    for i, pose in enumerate(poses[:-1]):
        frames = 14
        for interpol in np.linspace(0,1,frames)[:frames - 1]:
            pose = (1-interpol) * pose_extrinsics[f'c2w_{poses[i]}'] + interpol * pose_extrinsics[f'c2w_{poses[i+1]}']
            camera_path.append(pose)
    return camera_path


def load_pose(pose):
    return pose_extrinsics[f'c2w_{pose}']