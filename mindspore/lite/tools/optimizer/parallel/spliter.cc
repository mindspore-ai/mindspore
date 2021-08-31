/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/parallel/spliter.h"
#include <queue>
#include "tools/optimizer/fisson/fisson_util.h"
#include "tools/optimizer/parallel/split_strategy.h"
namespace mindspore {
namespace opt {
Spliter *Spliter::GetInstance() {
  static Spliter spliter;
  return &spliter;
}

void Spliter::VisitNodesInputs(const FuncGraphPtr &func_graph) {
  // for every node init it's inputs
  MS_ASSERT(func_graph != nullptr);
  for (const auto &node : func_graph->GetOrderedCnodes()) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    for (const auto &input : node->inputs()) {
      if (!utils::isa<CNodePtr>(input)) {
        continue;
      }
      nodes_inputs_[node].insert(input);
    }
  }
}

void Spliter::VisitNodesOutputs(const FuncGraphPtr &func_graph) {
  // for every node init it's outputs
  for (const auto &node : func_graph->GetOrderedCnodes()) {
    for (const auto &output_item : nodes_inputs_) {
      if (output_item.first != node) {
        for (const auto &output : output_item.second) {
          if (node == output) {
            nodes_outputs_[node].insert(output_item.first);
          }
        }
      }
    }
  }
}

void Spliter::RecordGraphInfo(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }
  VisitNodesInputs(func_graph);
  VisitNodesOutputs(func_graph);
  for (const auto &node : func_graph->GetOrderedCnodes()) {
    if (!utils::isa<CNodePtr>(node)) {
      return;
    }
    if (nodes_outputs_[node].size() > kDefaultBatch) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(kAnfPrimitiveIndex));
    MS_ASSERT(prim != nullptr);
    auto device_type =
      prim->GetAttr(ops::kDeviceType) != nullptr ? GetValue<int>(prim->GetAttr(ops::kDeviceType)) : kDeviceTypeNone;
    // has been searched
    if (device_type != kDeviceTypeNone) {
      return;
    }
    // check conv && depthwise_conv
    if (match_visited_[node] || !IsConv2D(node)) {
      continue;
    }
    int match_num = 0;
    std::queue<AnfNodePtr> conv_nodes;
    conv_nodes.push(node);
    while (true) {
      if (conv_nodes.empty()) {
        break;
      }
      auto curr_node = conv_nodes.front();
      conv_nodes.pop();
      if (match_visited_[curr_node]) {
        continue;
      }
      auto curr_cnode = curr_node->cast<CNodePtr>();
      match_visited_[curr_node] = true;
      // visit input, default pre_input is 1, and check it's node type whether is conv2d
      for (const auto &pre_input_node : nodes_inputs_[curr_node]) {
        if (match_visited_[pre_input_node] || !IsConv2D(pre_input_node)) {
          break;
        }
        conv_nodes.push(pre_input_node);
      }
      // visit output
      if (nodes_outputs_[curr_cnode].size() > kDefaultBatch) {
        break;
      }
      for (const auto &post_output_node : nodes_outputs_[curr_node]) {
        if (match_visited_[post_output_node] || !IsConv2D(post_output_node)) {
          break;
        }
        conv_nodes.push(post_output_node);
      }
      match_num++;
    }
    if (match_num != 0) {
      match_numbers_.insert(match_num);
    }
  }
}

void Spliter::UpdateNodeOutputs(const std::string &input_node_name, const AnfNodePtr &candidate_output) {
  if (candidate_output == nullptr) {
    return;
  }
  if (graph_node_outputs_.find(input_node_name) != graph_node_outputs_.end()) {
    std::vector<AnfNodePtr>::iterator it;
    it =
      find(graph_node_outputs_[input_node_name].begin(), graph_node_outputs_[input_node_name].end(), candidate_output);
    if (it != graph_node_outputs_[input_node_name].end()) {
      return;
    }
  }
  graph_node_outputs_[input_node_name].push_back(candidate_output);
}

void Spliter::UpdateNodeInputShapes(const std::string &node_name, const std::vector<ShapeVector> &input_shapes) {
  graph_node_input_shapes_[node_name] = (input_shapes);
}

void Spliter::UpdateNodeOutputShapes(const std::string &node_name, const std::vector<ShapeVector> &output_shapes) {
  graph_node_output_shapes_[node_name] = (output_shapes);
}

}  // namespace opt
}  // namespace mindspore
