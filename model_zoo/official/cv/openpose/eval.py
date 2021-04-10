# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import json
import os
import argparse
import warnings
import sys
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from pycocotools.coco import COCO as LoadAnn
from pycocotools.cocoeval import COCOeval as MapEval

from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import dtype as mstype

from src.config import params, JointType
from src.openposenet import OpenPoseNet
from src.dataset import valdata


warnings.filterwarnings("ignore")
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend", save_graphs=False, device_id=devid)
show_gt = 0

parser = argparse.ArgumentParser('mindspore openpose_net test')
parser.add_argument('--model_path', type=str, default='./0-33_170000.ckpt', help='path of testing model')
parser.add_argument('--imgpath_val', type=str, default='./dataset/coco/val2017', help='path of testing imgs')
parser.add_argument('--ann', type=str, default='./dataset/coco/annotations/person_keypoints_val2017.json',
                    help='path of annotations')
parser.add_argument('--output_path', type=str, default='./output_img', help='path of testing imgs')
# distributed related
parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
args, _ = parser.parse_known_args()

def evaluate_mAP(res_file, ann_file, ann_type='keypoints', silence=True):
    class NullWriter():
        def write(self, arg):
            pass
    if silence:
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite  # disable output

    Gt = LoadAnn(ann_file)
    Dt = Gt.loadRes(res_file)

    Eval = MapEval(Gt, Dt, ann_type)
    Eval.evaluate()
    Eval.accumulate()
    Eval.summarize()

    if silence:
        sys.stdout = oldstdout  # enable output

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = Eval.stats[ind]

    return info_str


def load_model(test_net, model_path):
    assert os.path.exists(model_path)
    param_dict = load_checkpoint(model_path)
    param_dict_new = {}
    for key, values in param_dict.items():

        if key.startswith('moment'):
            continue
        elif key.startswith('network'):
            param_dict_new[key[8:]] = values

    load_param_into_net(test_net, param_dict_new)

def preprocess(img):
    x_data = img.astype('f')
    x_data /= 255
    x_data -= 0.5
    x_data = x_data.transpose(2, 0, 1)[None]
    return x_data

def getImgsPath(img_dir_path):
    filepaths = []
    dirpaths = []
    pathName = img_dir_path

    for root, dirs, files in os.walk(pathName):
        for file in files:
            file_path = os.path.join(root, file)
            filepaths.append(file_path)
        for d in dirs:
            dir_path = os.path.join(root, d)
            dirpaths.append(dir_path)
    return filepaths

def compute_optimal_size(orig_img, img_size, stride=8):
    orig_img_h, orig_img_w, _ = orig_img.shape
    aspect = orig_img_h / orig_img_w
    if orig_img_h < orig_img_w:
        img_h = img_size
        img_w = np.round(img_size / aspect).astype(int)
        surplus = img_w % stride
        if surplus != 0:
            img_w += stride - surplus
    else:
        img_w = img_size
        img_h = np.round(img_size * aspect).astype(int)
        surplus = img_h % stride
        if surplus != 0:
            img_h += stride - surplus
    return (img_w, img_h)

def compute_peaks_from_heatmaps(heatmaps):

    heatmaps = heatmaps[:-1]

    all_peaks = []
    peak_counter = 0
    for i, heatmap in enumerate(heatmaps):
        heatmap = gaussian_filter(heatmap, sigma=params['gaussian_sigma'])

        map_left = np.zeros(heatmap.shape)
        map_right = np.zeros(heatmap.shape)
        map_top = np.zeros(heatmap.shape)
        map_bottom = np.zeros(heatmap.shape)

        map_left[1:, :] = heatmap[:-1, :]
        map_right[:-1, :] = heatmap[1:, :]
        map_top[:, 1:] = heatmap[:, :-1]
        map_bottom[:, :-1] = heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce((
            heatmap > params['heatmap_peak_thresh'],
            heatmap > map_left,
            heatmap > map_right,
            heatmap > map_top,
            heatmap > map_bottom,
        ))

        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])

        peaks_with_score = [(i,) + peak_pos + (heatmap[peak_pos[1], peak_pos[0]],) for peak_pos in peaks]

        peaks_id = range(peak_counter, peak_counter + len(peaks_with_score))
        peaks_with_score_and_id = [peaks_with_score[i] + (peaks_id[i],) for i in range(len(peaks_id))]

        peak_counter += len(peaks_with_score_and_id)
        all_peaks.append(peaks_with_score_and_id)
    all_peaks = np.array([peak for peaks_each_category in all_peaks for peak in peaks_each_category])

    return all_peaks

def compute_candidate_connections(paf, cand_a, cand_b, img_len, params_):
    candidate_connections = []
    for joint_a in cand_a:
        for joint_b in cand_b:
            vector = joint_b[:2] - joint_a[:2]
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            ys = np.linspace(joint_a[1], joint_b[1], num=params_['n_integ_points'])
            xs = np.linspace(joint_a[0], joint_b[0], num=params_['n_integ_points'])
            integ_points = np.stack([ys, xs]).T.round().astype('i')

            paf_in_edge = np.hstack([paf[0][np.hsplit(integ_points, 2)], paf[1][np.hsplit(integ_points, 2)]])
            unit_vector = vector / norm
            inner_products = np.dot(paf_in_edge, unit_vector)
            integ_value = inner_products.sum() / len(inner_products)
            integ_value_with_dist_prior = integ_value + min(params_['limb_length_ratio'] * img_len / norm -
                                                            params_['length_penalty_value'], 0)
            n_valid_points = sum(inner_products > params_['inner_product_thresh'])
            if n_valid_points > params_['n_integ_points_thresh'] and integ_value_with_dist_prior > 0:
                candidate_connections.append([int(joint_a[3]), int(joint_b[3]), integ_value_with_dist_prior])
    candidate_connections = sorted(candidate_connections, key=lambda x: x[2], reverse=True)
    return candidate_connections

def compute_connections(pafs, all_peaks, img_len, params_):
    all_connections = []
    for i in range(len(params_['limbs_point'])):
        paf_index = [i * 2, i * 2 + 1]
        paf = pafs[paf_index]  # shape: (2, 320, 320)
        limb_point = params_['limbs_point'][i]  # example: [<JointType.Neck: 1>, <JointType.RightWaist: 8>]
        cand_a = all_peaks[all_peaks[:, 0] == limb_point[0]][:, 1:]
        cand_b = all_peaks[all_peaks[:, 0] == limb_point[1]][:, 1:]

        if cand_a.shape[0] > 0 and cand_b.shape[0] > 0:
            candidate_connections = compute_candidate_connections(paf, cand_a, cand_b, img_len, params_)

            connections = np.zeros((0, 3))

            for index_a, index_b, score in candidate_connections:
                if index_a not in connections[:, 0] and index_b not in connections[:, 1]:
                    connections = np.vstack([connections, [index_a, index_b, score]])
                    if len(connections) >= min(len(cand_a), len(cand_b)):
                        break
            all_connections.append(connections)
        else:
            all_connections.append(np.zeros((0, 3)))
    return all_connections

def grouping_key_points(all_connections, candidate_peaks, params_):
    subsets = -1 * np.ones((0, 20))

    for l, connections in enumerate(all_connections):
        joint_a, joint_b = params_['limbs_point'][l]
        for ind_a, ind_b, score in connections[:, :3]:
            ind_a, ind_b = int(ind_a), int(ind_b)
            joint_found_cnt = 0
            joint_found_subset_index = [-1, -1]
            for subset_ind, subset in enumerate(subsets):

                if subset[joint_a] == ind_a or subset[joint_b] == ind_b:
                    joint_found_subset_index[joint_found_cnt] = subset_ind
                    joint_found_cnt += 1

            if joint_found_cnt == 1:

                found_subset = subsets[joint_found_subset_index[0]]
                if found_subset[joint_b] != ind_b:
                    found_subset[joint_b] = ind_b
                    found_subset[-1] += 1  # increment joint count
                    found_subset[-2] += candidate_peaks[ind_b, 3] + score


            elif joint_found_cnt == 2:

                found_subset_1 = subsets[joint_found_subset_index[0]]
                found_subset_2 = subsets[joint_found_subset_index[1]]

                membership = ((found_subset_1 >= 0).astype(int) + (found_subset_2 >= 0).astype(int))[:-2]
                if not np.any(membership == 2):  # merge two subsets when no duplication
                    found_subset_1[:-2] += found_subset_2[:-2] + 1  # default is -1
                    found_subset_1[-2:] += found_subset_2[-2:]
                    found_subset_1[-2] += score
                    subsets = np.delete(subsets, joint_found_subset_index[1], axis=0)
                else:
                    if found_subset_1[joint_a] == -1:
                        found_subset_1[joint_a] = ind_a
                        found_subset_1[-1] += 1
                        found_subset_1[-2] += candidate_peaks[ind_a, 3] + score
                    elif found_subset_1[joint_b] == -1:
                        found_subset_1[joint_b] = ind_b
                        found_subset_1[-1] += 1
                        found_subset_1[-2] += candidate_peaks[ind_b, 3] + score
                    if found_subset_2[joint_a] == -1:
                        found_subset_2[joint_a] = ind_a
                        found_subset_2[-1] += 1
                        found_subset_2[-2] += candidate_peaks[ind_a, 3] + score
                    elif found_subset_2[joint_b] == -1:
                        found_subset_2[joint_b] = ind_b
                        found_subset_2[-1] += 1
                        found_subset_2[-2] += candidate_peaks[ind_b, 3] + score

            elif joint_found_cnt == 0 and l != 9 and l != 13:
                row = -1 * np.ones(20)
                row[joint_a] = ind_a
                row[joint_b] = ind_b
                row[-1] = 2
                row[-2] = sum(candidate_peaks[[ind_a, ind_b], 3]) + score
                subsets = np.vstack([subsets, row])
            elif joint_found_cnt >= 3:
                pass

    # delete low score subsets
    keep = np.logical_and(subsets[:, -1] >= params_['n_subset_limbs_thresh'],
                          subsets[:, -2] / subsets[:, -1] >= params_['subset_score_thresh'])
    subsets = subsets[keep]
    return subsets

def subsets_to_pose_array(subsets, all_peaks):
    person_pose_array = []
    for subset in subsets:
        joints = []
        for joint_index in subset[:18].astype('i'):
            if joint_index >= 0:
                joint = all_peaks[joint_index][1:3].tolist()
                joint.append(2)
                joints.append(joint)
            else:
                joints.append([0, 0, 0])
        person_pose_array.append(np.array(joints))
    person_pose_array = np.array(person_pose_array)
    return person_pose_array

def detect(img, network):
    orig_img = img.copy()
    orig_img_h, orig_img_w, _ = orig_img.shape

    input_w, input_h = compute_optimal_size(orig_img, params['inference_img_size']) # 368
    map_w, map_h = compute_optimal_size(orig_img, params['inference_img_size'])

    resized_image = cv2.resize(orig_img, (input_w, input_h))
    x_data = preprocess(resized_image)
    x_data = Tensor(x_data, mstype.float32)
    x_data.requires_grad = False

    logit_pafs, logit_heatmap = network(x_data)

    logit_pafs = logit_pafs[-1].asnumpy()[0]
    logit_heatmap = logit_heatmap[-1].asnumpy()[0]

    pafs = np.zeros((logit_pafs.shape[0], map_h, map_w))
    for i in range(logit_pafs.shape[0]):
        pafs[i] = cv2.resize(logit_pafs[i], (map_w, map_h))
        if show_gt:
            save_path = "./test_output/" + str(i) + "pafs.png"
            cv2.imwrite(save_path, pafs[i]*255)

    heatmaps = np.zeros((logit_heatmap.shape[0], map_h, map_w))
    for i in range(logit_heatmap.shape[0]):
        heatmaps[i] = cv2.resize(logit_heatmap[i], (map_w, map_h))
        if show_gt:
            save_path = "./test_output/" + str(i) + "heatmap.png"
            cv2.imwrite(save_path, heatmaps[i]*255)

    all_peaks = compute_peaks_from_heatmaps(heatmaps)
    if all_peaks.shape[0] == 0:
        return np.empty((0, len(JointType), 3)), np.empty(0)
    all_connections = compute_connections(pafs, all_peaks, map_w, params)
    subsets = grouping_key_points(all_connections, all_peaks, params)
    all_peaks[:, 1] *= orig_img_w / map_w
    all_peaks[:, 2] *= orig_img_h / map_h
    poses = subsets_to_pose_array(subsets, all_peaks)
    scores = subsets[:, -2]

    return poses, scores

def draw_person_pose(orig_img, poses):
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    if poses.shape[0] == 0:
        return orig_img

    limb_colors = [
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
        [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
        [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
        [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
    ]

    joint_colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    canvas = orig_img.copy()

    # limbs
    for pose in poses.round().astype('i'):
        for i, (limb, color) in enumerate(zip(params['limbs_point'], limb_colors)):
            if i not in (9, 13):  # don't show ear-shoulder connection
                limb_ind = np.array(limb)
                if np.all(pose[limb_ind][:, 2] != 0):
                    joint1, joint2 = pose[limb_ind][:, :2]
                    cv2.line(canvas, tuple(joint1), tuple(joint2), color, 2)

    # joints
    for pose in poses.round().astype('i'):
        for i, ((x, y, v), color) in enumerate(zip(pose, joint_colors)):
            if v != 0:
                cv2.circle(canvas, (x, y), 3, color, -1)
    return canvas

def depreprocess(img):
    x_data = img[0]
    x_data += 0.5
    x_data *= 255
    x_data = x_data.astype('uint8')
    x_data = x_data.transpose(1, 2, 0)
    return x_data

def val():
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    network = OpenPoseNet(vgg_with_bn=params['vgg_with_bn'])
    network.set_train(False)
    load_model(network, args.model_path)

    print("load models right")
    dataset = valdata(args.ann, args.imgpath_val, args.rank, args.group_size, mode='val')
    dataset_size = dataset.get_dataset_size()
    de_dataset = dataset.create_tuple_iterator()

    print("eval dataset size: ", dataset_size)
    kpt_json = []
    for _, (img, img_id) in tqdm(enumerate(de_dataset), total=dataset_size):
        img = img.asnumpy()
        img_id = int((img_id.asnumpy())[0])
        poses, scores = detect(img, network)

        if poses.shape[0] > 0:
            for index, pose in enumerate(poses):
                data = dict()

                pose = pose[[0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10, 1], :].round().astype('i')

                keypoints = pose.reshape(-1).tolist()
                keypoints = keypoints[:-3]
                data['image_id'] = img_id
                data['score'] = scores[index]
                data['category_id'] = 1
                data['keypoints'] = keypoints
                kpt_json.append(data)
        else:
            print("Predict poses size is zero.", flush=True)
        img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)

        save_path = os.path.join(args.output_path, str(img_id)+".png")
        cv2.imwrite(save_path, img)

    result_json = 'eval_result.json'
    with open(os.path.join(args.output_path, result_json), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(args.output_path, result_json), ann_file=args.ann)
    print('result: ', res)

if __name__ == "__main__":
    val()
