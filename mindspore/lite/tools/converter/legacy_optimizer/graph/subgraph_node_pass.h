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

#ifndef MINDSPORE_PREDICT_ISOLATED_SUBGRAPH_NODE_PASS_H
#define MINDSPORE_PREDICT_ISOLATED_SUBGRAPH_NODE_PASS_H

#include <vector>
#include <utility>
#include <set>
#include <memory>
#include "tools/converter/optimizer.h"

namespace mindspore {
namespace lite {
class SubgraphNodePass : public GraphPass {
 public:
  explicit SubgraphNodePass(std::vector<schema::CNodeT *> old_nodes) : old_nodes_(std::move(old_nodes)) {}

  ~SubgraphNodePass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  void DecreaseSubgraphNodeIndices(const size_t &node_idx, schema::MetaGraphT *graph);
  void IncreaseSubgraphNodeIndices(const size_t &node_idx, schema::MetaGraphT *graph);
  STATUS GetSubgraphAllTensorIndices(const std::unique_ptr<SubGraphT> &subgraph, schema::MetaGraphT *graph,
                                     std::set<uint32_t> *tensors_indices);
  bool IsNodeInputInSubgraph(const std::set<uint32_t> &tensors_indices, const std::unique_ptr<CNodeT> &node,
                             const std::unique_ptr<SubGraphT> &subgraph);
  bool IsNodeOutputInSubgraph(const std::set<uint32_t> &tensors_indices, const std::unique_ptr<CNodeT> &node,
                              const std::unique_ptr<SubGraphT> &subgraph);
  std::vector<schema::CNodeT *> old_nodes_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_ISOLATED_NODE_REMOVE_PASS_H
