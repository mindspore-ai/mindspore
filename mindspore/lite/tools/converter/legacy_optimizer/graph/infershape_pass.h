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

#ifndef MINDSPORE_PREDICT_INFERSHAPE_PASS_H
#define MINDSPORE_PREDICT_INFERSHAPE_PASS_H

#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "tools/common/graph_util.h"
#include "tools/converter/optimizer.h"
#include "tools/converter/converter_flags.h"

using mindspore::converter::kFmkTypeTf;
using mindspore::schema::TensorT;
namespace mindspore {
namespace lite {
const constexpr int kTensorDataSize = 8;
const constexpr int kSwitchTrueIndex = 1;
const constexpr int kSwitchFalseIndex = 2;
struct InferTensor {
  std::vector<uint32_t> next_nodes_;
  std::vector<uint32_t> prev_nodes_;
  bool is_inferred_;
};

class InferShapePass : public GraphPass {
 public:
  explicit InferShapePass(converter::FmkType fmk_type) : fmk_type_(fmk_type) {}
  ~InferShapePass() override = default;
  STATUS Run(MetaGraphT *graph) override;

 private:
  int InitSearchTensor(const int &subgraph_index, MetaGraphT *graph, std::vector<uint32_t> *infer_node_indexes);
  void AddNextInferShapeNode(MetaGraphT *graph, std::vector<uint32_t> *infer_node_indexes,
                             std::vector<uint32_t> next_nodes_indexes, size_t index);
  void AddOutputNodes(MetaGraphT *graph, std::vector<uint32_t> *infer_node_indexes, uint32_t infer_node_index);
  void ResetIncorrectTensorShape(MetaGraphT *graph);
  int InferPartialNode(const CNodeT *partial_node, MetaGraphT *graph);
  int InferSwitchNode(const std::unique_ptr<CNodeT> &switch_node, MetaGraphT *graph);
  int InferCallNode(const std::unique_ptr<CNodeT> &call_node, MetaGraphT *graph);
  int CopyPartialShapeToSubGraph(const CNodeT *partial_node, MetaGraphT *graph);
  int RestoreSubGraphInput(const CNodeT *partial_node, MetaGraphT *graph);
  void InitInferTensor(MetaGraphT *graph);
  int InferSubgraph(const int &subgraph_index, MetaGraphT *graph);

  converter::FmkType fmk_type_ = kFmkTypeTf;
  std::vector<InferTensor> tensors_ = {};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_INFERSHAPE_PASS_H
