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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_SPLITER_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_SPLITER_H_
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include "schema/inner/model_generated.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/lite_types.h"
namespace mindspore {
namespace opt {
struct IntCompare {
  bool operator()(const int &lhs, const int &rhs) const { return lhs > rhs; }
};

class Spliter {
 public:
  static Spliter *GetInstance();
  Spliter(const Spliter &) = delete;
  Spliter &operator=(const Spliter &) = delete;

  // record the global numbers of matched multi_conv nodes
  void RecordGraphInfo(const FuncGraphPtr &func_graph);

  // update current input node's output. if candidate node has been recorded, we will be ignore it, otherwise record it.
  void UpdateNodeOutputs(const std::string &input_node_name, const AnfNodePtr &candidate_output);

  // update current node's input shapes.
  void UpdateNodeInputShapes(const std::string &node_name, const std::vector<ShapeVector> &input_shapes);

  // update current node's output shapes.
  void UpdateNodeOutputShapes(const std::string &node_name, const std::vector<ShapeVector> &output_shapes);

  std::unordered_map<std::string, std::vector<AnfNodePtr>> graph_node_outputs() const { return graph_node_outputs_; }

  std::unordered_map<std::string, std::vector<ShapeVector>> graph_node_output_shapes() const {
    return graph_node_output_shapes_;
  }

  std::unordered_map<std::string, std::vector<ShapeVector>> graph_node_input_shapes() const {
    return graph_node_input_shapes_;
  }

  std::set<int, IntCompare> graph_match_multi_numbers() const { return match_numbers_; }

  std::unordered_map<AnfNodePtr, std::set<AnfNodePtr>> nodes_inputs() const { return nodes_inputs_; }

  std::unordered_map<AnfNodePtr, std::set<AnfNodePtr>> nodes_outputs() const { return nodes_outputs_; }

  void VisitNodesInputs(const FuncGraphPtr &func_graph);

  void VisitNodesOutputs(const FuncGraphPtr &func_graph);

 private:
  Spliter() = default;
  virtual ~Spliter() = default;

 private:
  std::unordered_map<std::string, std::vector<AnfNodePtr>> graph_node_outputs_;
  std::unordered_map<std::string, std::vector<ShapeVector>> graph_node_output_shapes_;
  std::unordered_map<std::string, std::vector<ShapeVector>> graph_node_input_shapes_;
  std::unordered_map<AnfNodePtr, std::set<AnfNodePtr>> nodes_inputs_;
  std::unordered_map<AnfNodePtr, std::set<AnfNodePtr>> nodes_outputs_;
  std::unordered_map<AnfNodePtr, bool> match_visited_;
  std::set<int, IntCompare> match_numbers_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_SPLITER_H_
