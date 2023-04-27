/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_ONLINE_FUSION_ONLINE_FUSION_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_ONLINE_FUSION_ONLINE_FUSION_PASS_H_

#include <stack>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include "include/model.h"
#include "src/executor/kernel_exec.h"
#include "src/litert/lite_model.h"
#include "src/litert/inner_context.h"
#include "src/litert/sub_graph_split.h"
#include "src/litert/pass/online_fusion/online_fusion_pass_registry.h"
#include "src/common/prim_util.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/split_parameter.h"

namespace mindspore::lite {
class OnlineFusionPass {
 public:
  explicit OnlineFusionPass(SearchSubGraph *search_subgrap);
  ~OnlineFusionPass();

 public:
  void DoOnlineFusionPass();

 protected:
  virtual void DoOnlineFusion() {}
  int InitOnlineFusion();
  int CreateCustomNode(LiteGraph::Node *node, SearchSubGraph::Subgraph *subgraph, SplitParameter *split_param);
  OpParameter *GetNodeOpParameter(LiteGraph::Node *node);
  std::vector<std::vector<uint32_t>> GetFrontNodeIndex(LiteGraph::Node *cur_node);
  std::vector<std::vector<uint32_t>> GetNextNodeIndex(LiteGraph::Node *cur_node);
  flatbuffers::Offset<mindspore::schema::Attribute> SetDataToUint8Vector(void *src, size_t len,
                                                                         flatbuffers::FlatBufferBuilder *fbb,
                                                                         const char *attr_name);

 protected:
  const InnerContext *context_ = nullptr;
  LiteModel *model_ = nullptr;
  SearchSubGraph *search_subgrap_ = nullptr;
  std::vector<lite::Tensor *> *src_tensors_ = nullptr;
  std::vector<SearchSubGraph::Tensor> *tensors_ = nullptr;
  std::vector<SearchSubGraph::Subgraph> sub_graphs_;
  std::vector<LiteGraph::Node *> node_list_;
};  // namespace mindspore::lite
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_SPLIT_REDUCE_CONCAT_ONLINE_FUSION_PASS_H_
