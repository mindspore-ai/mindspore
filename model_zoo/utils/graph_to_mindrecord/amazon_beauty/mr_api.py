# Copyright 2020 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
User-defined API for MindRecord GNN writer.
"""
import os
import csv

args = os.environ['graph_api_args'].split(':')
USER_FILE = args[0]
ITEM_FILE = args[1]
RATING_FILE = args[2]

node_profile = (0, [], [])
edge_profile = (0, [], [])


def yield_nodes(task_id=0):
    """
    Generate node data

    Yields:
        data (dict): data row which is dict.
    """
    print("Node task is {}".format(task_id))
    with open(USER_FILE) as user_file:
        user_reader = csv.reader(user_file, delimiter=',')
        line_count = 0
        for row in user_reader:
            node = {'id': int(row[1]), 'type': 0}
            yield node
            line_count += 1
        print('Processed {} lines for users.'.format(line_count))

    with open(ITEM_FILE) as item_file:
        item_reader = csv.reader(item_file, delimiter=',')
        line_count = 0
        for row in item_reader:
            node = {'id': int(row[1]), 'type': 1,}
            yield node
            line_count += 1
        print('Processed {} lines for items.'.format(line_count))


def yield_edges(task_id=0):
    """
    Generate edge data

    Yields:
        data (dict): data row which is dict.
    """
    print("Edge task is {}".format(task_id))
    with open(RATING_FILE) as rating_file:
        rating_reader = csv.reader(rating_file, delimiter=',')
        line_count = 0
        for row in rating_reader:
            if line_count == 0:
                line_count += 1
                continue
            edge = {'id': line_count - 1, 'src_id': int(row[0]), 'dst_id': int(row[1]), 'type': int(row[2])}
            yield edge
            line_count += 1
        print('Processed {} lines for edges.'.format(line_count))
