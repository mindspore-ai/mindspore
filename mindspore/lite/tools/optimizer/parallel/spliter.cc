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
namespace mindspore {
namespace opt {
Spliter *Spliter::GetInstance() {
  static Spliter spliter;
  return &spliter;
}

void Spliter::UpdateNodeOutputs(const std::string &input_node_name, const AnfNodePtr &candidate_output) {
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
