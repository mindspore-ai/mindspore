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

#ifndef MINDSPORE_PREDICT_SWITCH_PASS_H
#define MINDSPORE_PREDICT_SWITCH_PASS_H
#include <unordered_map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tools/common/graph_util.h"
#include "tools/converter/optimizer.h"

using mindspore::schema::TensorT;
namespace mindspore {
namespace lite {
class SwitchPass : public GraphPass {
 public:
  SwitchPass() = default;
  ~SwitchPass() override = default;
  STATUS Run(schema::MetaGraphT *graph) override;
};

class SingleSwitchPass {
 public:
  SingleSwitchPass(schema::MetaGraphT *graph, const size_t &node_index)
      : graph_(graph), switch_node_index_(node_index) {}
  ~SingleSwitchPass() = default;
  STATUS Run();

 private:
  STATUS Init();
  size_t InitThisGraphIndex();
  STATUS DoubleSwitchOutput();
  STATUS UpdateSwitchUser();
  STATUS ConcatCondSubgraphInputAndOutput();
  STATUS ConcatBodySubgraphInputAndOutput();
  bool IsLoop();
  STATUS InsertMerge();
  std::unique_ptr<schema::CNodeT> MakeMergeNode(const std::string &name, const std::vector<size_t> &const_in);

  // function for if
  STATUS InsertPartialAndMergeAfterSwitch();
  int GetSubgraphInputTensorIndex(const std::unique_ptr<SubGraphT> &subgraph, const std::unique_ptr<TensorT> &tensor);
  int GetSubgraphOutputTensorIndex(const std::unique_ptr<SubGraphT> &subgraph, CNodeT *node);
  STATUS UpdateSubgraphInput(const size_t &subgraph_index, schema::CNodeT *partial_node,
                             const std::vector<schema::CNodeT *> &subgraph_nodes);
  STATUS UpdateSubgraphOutput(const size_t &subgraph_index, schema::CNodeT *partial_node,
                              const std::vector<schema::CNodeT *> &subgraph_nodes);
  std::unique_ptr<schema::TensorT> NewTensor(const std::unique_ptr<schema::TensorT> &in_tensor, bool with_data = false);
  void IsolateUselessNode(schema::CNodeT *partial_node, schema::MetaGraphT *graph);
  void UpdateSwitchOutputIndices(uint32_t *idx);
  STATUS BodyGraphVariableInput(std::vector<size_t> *variable_input);

  const size_t kSwitchFirstIndex = 0;
  const size_t kSwitchSecondIndex = 1;
  const size_t kSwitchMinInputSize = 2;

  schema::MetaGraphT *graph_ = nullptr;
  schema::CNodeT *switch_node_ = nullptr;
  schema::CNodeT *first_partial_node_ = nullptr;
  schema::CNodeT *second_partial_node_ = nullptr;
  schema::CNodeT *body_to_cond_partial_node_ = nullptr;
  std::vector<schema::CNodeT *> this_graph_nodes_;
  std::vector<schema::CNodeT *> second_graph_nodes_;
  std::vector<schema::CNodeT *> first_graph_nodes_;
  size_t switch_node_index_ = -1;
  int32_t this_subgraph_index_ = -1;
  int32_t first_subgraph_index_ = -1;
  int32_t second_subgraph_index_ = -1;
  std::vector<uint32_t> origin_switch_output_tensor_indices_;
};
}  // namespace lite
}  // namespace mindspore
#endif
