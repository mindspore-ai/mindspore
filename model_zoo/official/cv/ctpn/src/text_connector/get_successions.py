# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
from src.config import config
from src.text_connector.utils import overlaps_v, size_similarity

def get_successions(text_proposals, scores, im_size):
    """
    Get successions text boxes.

    Args:
        text_proposals(numpy.array): Predict text proposals.
        scores(numpy.array): Bbox predicts scores.
        size(numpy.array): Image size.
    Returns:
        sub_graph(list): Proposals graph.
    """
    bboxes_table = [[] for _ in range(int(im_size[1]))]
    for index, box in enumerate(text_proposals):
        bboxes_table[int(box[0])].append(index)
    graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)
    for index, box in enumerate(text_proposals):
        successions_left = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + config.max_horizontal_gap + 1, im_size[1])):
            adj_box_indices = bboxes_table[left]
            for adj_box_index in adj_box_indices:
                if meet_v_iou(text_proposals, adj_box_index, index):
                    successions_left.append(adj_box_index)
            if successions_left:
                break
        if not successions_left:
            continue
        succession_index = successions_left[np.argmax(scores[successions_left])]
        box_right = text_proposals[succession_index]
        succession_right = []
        for right in range(int(box_right[0]) - 1, max(int(box_right[0] - config.max_horizontal_gap), 0) - 1, -1):
            adj_box_indices = bboxes_table[right]
            for adj_box_index in adj_box_indices:
                if meet_v_iou(text_proposals, adj_box_index, index):
                    succession_right.append(adj_box_index)
            if succession_right:
                break
        if scores[index] >= np.max(scores[succession_right]):
            graph[index, succession_index] = True
    sub_graph = get_sub_graph(graph)
    return sub_graph

def get_sub_graph(graph):
    """
    Get successions text boxes.

    Args:
        graph(numpy.array): proposal graph
    Returns:
        sub_graph(list): Proposals graph after connect.
    """
    sub_graphs = []
    for index in range(graph.shape[0]):
        if not graph[:, index].any() and graph[index, :].any():
            v = index
            sub_graphs.append([v])
            while graph[v, :].any():
                v = np.where(graph[v, :])[0][0]
                sub_graphs[-1].append(v)
    return sub_graphs

def meet_v_iou(text_proposals, index1, index2):
    """
    Calculate vertical iou.

    Args:
        text_proposals(numpy.array): tex proposals
        index1(int): text_proposal index
        tindex2(int): text proposal index
    Returns:
        sub_graph(list): Proposals graph after connect.
    """
    heights = text_proposals[:, 3] - text_proposals[:, 1] + 1
    return overlaps_v(text_proposals, index1, index2) >= config.min_v_overlaps and \
        size_similarity(heights, index1, index2) >= config.min_size_sim
