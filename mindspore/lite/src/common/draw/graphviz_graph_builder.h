/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_DRAW_GRAPHVIZ_GRAPH_BUILDER_H_
#define MINDSPORE_LITE_SRC_COMMON_DRAW_GRAPHVIZ_GRAPH_BUILDER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "src/common/draw/graphviz_graph.h"
#include "src/common/draw/adapter_graph.h"
#include "src/tensor.h"
#include "include/errorcode.h"

namespace mindspore::lite {
class GVGraphBuilder {
 public:
  std::shared_ptr<GVGraph> Build(const std::shared_ptr<AdapterGraph> &graph);

  void AppendGraphInputNode(const lite::Tensor &tensor);
  void AppendWeightNode(const lite::Tensor &tensor, const std::string &id, const std::string &label);
  int AppendComputeNode(const AdapterNode &node);
  int AppendGraphOutputNode(const std::vector<lite::Tensor *> &out_tensors);

 protected:
  static GVNode *CreateComputeNode(const AdapterNode &node);
  int LinkNodes(const AdapterNode &node, const GVNode &gv_node);
  void AppendOutTensorMap(const lite::Tensor *tensor, lite::GVNode *node, size_t out_index);
  std::pair<lite::GVNode *, size_t> GetBelongingGVNode(const lite::Tensor *tensor) const;

  std::shared_ptr<GVGraph> gv_graph_{nullptr};
  std::string name_;
  std::unordered_map<const lite::Tensor *, std::pair<lite::GVNode *, size_t>> gv_node_out_tensor_map_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_COMMON_DRAW_GRAPHVIZ_GRAPH_BUILDER_H_
