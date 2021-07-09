/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <vector>
#include <queue>

namespace {
enum Format { kFormatUnknown, kFormatA, kFormatB };
enum TransOpType { kTransAB, kTransBA };

struct Edge {
  size_t from;
  size_t to;
  size_t val;
  size_t next;
  Edge(size_t From, size_t To, size_t Val, size_t Next) {
    from = From;
    to = To;
    val = Val;
    next = Next;
  }
};

struct Node {
  int head;
  int cur;
  int depth;
  size_t pre;
  Format format;
  Node() {
    head = -1;
    cur = -1;
    depth = -1;
    pre = 0;
    format = kFormatB;
  }
};

constexpr size_t INF = static_cast<size_t>(1) << 30;

class MinCut {
 public:
  // Connect the source_node to the node_with_a_certain_formatA
  // or Connect the node_with_a_certain_formatB to the sink_node
  void Add_1(size_t from, size_t to) {
    edges_.emplace_back(from, to, INF, nodes_[from].head);
    nodes_[from].head = edges_count_++;
    edges_.emplace_back(to, from, 0, nodes_[to].head);
    nodes_[to].head = edges_count_++;
  }

  // Split one origin_node into two new_nodes and connect them
  void Add_2(size_t nodes_id) {
    edges_.emplace_back(nodes_id, nodes_id + origin_nodes_num_, 1, nodes_[nodes_id].head);
    nodes_[nodes_id].head = edges_count_++;
    edges_.emplace_back(nodes_id + origin_nodes_num_, nodes_id, 1, nodes_[nodes_id + origin_nodes_num_].head);
    nodes_[nodes_id + origin_nodes_num_].head = edges_count_++;
  }

  // After splitting the origin_nodes, construct the new_edges based on the original_edges
  void Add_3(size_t from, size_t to) {
    edges_.emplace_back(from + origin_nodes_num_, to, 1, nodes_[from + origin_nodes_num_].head);
    nodes_[from + origin_nodes_num_].head = edges_count_++;
    edges_.emplace_back(to, from + origin_nodes_num_, 1, nodes_[to].head);
    nodes_[to].head = edges_count_++;
  }

  void BFS() {
    std::queue<size_t> bfs_queue;
    nodes_[sink_id_].depth = 0;
    bfs_queue.push(sink_id_);
    while (!bfs_queue.empty()) {
      size_t temp_node = bfs_queue.front();
      bfs_queue.pop();
      depth_num_[nodes_[temp_node].depth]++;
      for (size_t i = nodes_[temp_node].head; ~i; i = edges_[i].next) {
        if (edges_[i ^ 1].val && nodes_[edges_[i].to].depth == -1) {
          nodes_[edges_[i].to].depth = nodes_[temp_node].depth + 1;
          bfs_queue.push(edges_[i].to);
        }
      }
    }
  }

  void EdgeValueUpdate() {
    size_t k = sink_id_, flow = INF;
    while (k != source_id_) {
      if (edges_[nodes_[k].pre].val < flow) {
        flow = edges_[nodes_[k].pre].val;
      }
      k = edges_[nodes_[k].pre].from;
    }
    k = sink_id_;
    while (k != source_id_) {
      edges_[nodes_[k].pre].val -= flow;
      edges_[nodes_[k].pre ^ 1].val += flow;
      k = edges_[nodes_[k].pre].from;
    }
  }

  void ISAP() {
    size_t node_id = source_id_;
    int maxdep = 2 * origin_nodes_num_ + 2;
    BFS();
    for (size_t i = source_id_; i <= sink_id_; ++i) {
      nodes_[i].cur = nodes_[i].head;
    }
    while (nodes_[source_id_].depth <= maxdep) {
      if (node_id == sink_id_) {
        EdgeValueUpdate();
        node_id = source_id_;
      }
      bool can_arrive = false;
      for (size_t i = nodes_[node_id].cur; ~i; i = edges_[i].next) {
        if (edges_[i].val && nodes_[edges_[i].to].depth + 1 == nodes_[node_id].depth) {
          can_arrive = true;
          nodes_[edges_[i].to].pre = i;
          nodes_[node_id].cur = i;
          node_id = edges_[i].to;
          break;
        }
      }
      if (!can_arrive) {
        int mindep = 2 * origin_nodes_num_ + 2;
        for (size_t i = nodes_[node_id].head; ~i; i = edges_[i].next) {
          if (nodes_[edges_[i].to].depth < mindep && edges_[i].val) {
            mindep = nodes_[edges_[i].to].depth;
          }
        }
        --depth_num_[nodes_[node_id].depth];
        if (!depth_num_[nodes_[node_id].depth]) {
          break;
        }
        nodes_[node_id].depth = mindep + 1;
        depth_num_[nodes_[node_id].depth]++;
        nodes_[node_id].cur = nodes_[node_id].head;
        if (node_id != source_id_) {
          node_id = edges_[nodes_[node_id].pre].from;
        }
      }
    }
  }

  void SetFormat(size_t node_id) {
    nodes_[node_id].format = kFormatA;
    for (size_t i = nodes_[node_id].head; ~i; i = edges_[i].next) {
      if (edges_[i].val && nodes_[edges_[i].to].format != kFormatA) {
        SetFormat(edges_[i].to);
      }
    }
  }

  MinCut(std::vector<std::pair<size_t, Format>> original_nodes, std::vector<std::pair<size_t, size_t>> original_edges)
      : origin_nodes_num_(original_nodes.size()),
        sink_id_(2 * origin_nodes_num_ + 1),
        depth_num_(std::vector<size_t>(2 * origin_nodes_num_ + 2, 0)),
        nodes_(std::vector<Node>(2 * origin_nodes_num_ + 2, Node())),
        original_edges_(std::move(original_edges)) {
    for (size_t i = 0; i < origin_nodes_num_; ++i) {
      if (original_nodes[i].second == kFormatA) {
        Add_1(source_id_, original_nodes[i].first);
      } else if (original_nodes[i].second == kFormatB) {
        Add_1(original_nodes[i].first, sink_id_);
      }
      Add_2(original_nodes[i].first);
    }
    for (auto i : original_edges_) {
      Add_3(i.first, i.second);
    }
    ISAP();
    SetFormat(source_id_);
  }

  std::vector<std::pair<size_t, TransOpType>> GetOneNodeOps() const {
    std::vector<std::pair<size_t, TransOpType>> one_node_ops;
    for (size_t i = 1; i <= origin_nodes_num_; ++i) {
      if (nodes_[i].format == kFormatA && nodes_[i + origin_nodes_num_].format != kFormatA) {
        one_node_ops.emplace_back(i, kTransAB);
      } else if (nodes_[i].format != kFormatA && nodes_[i + origin_nodes_num_].format == kFormatA) {
        one_node_ops.emplace_back(i, kTransBA);
      }
    }
    return one_node_ops;
  }

  std::vector<std::pair<std::pair<size_t, size_t>, TransOpType>> GetTwoNodeOps() const {
    std::vector<std::pair<std::pair<size_t, size_t>, TransOpType>> two_node_ops;
    for (auto i : original_edges_) {
      if (nodes_[i.first + origin_nodes_num_].format == kFormatA && nodes_[i.second].format != kFormatA) {
        two_node_ops.emplace_back(i, kTransAB);
      } else if (nodes_[i.first + origin_nodes_num_].format != kFormatA && nodes_[i.second].format == kFormatA) {
        two_node_ops.emplace_back(i, kTransBA);
      }
    }
    return two_node_ops;
  }

 private:
  size_t origin_nodes_num_;
  size_t source_id_{0};
  size_t sink_id_;
  int edges_count_{0};
  std::vector<size_t> depth_num_;
  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<std::pair<size_t, size_t>> original_edges_;
};

}  // namespace
