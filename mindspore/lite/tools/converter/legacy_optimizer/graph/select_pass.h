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
#ifndef MINDSPORE_PREDICT_SELECT_PASS_H
#define MINDSPORE_PREDICT_SELECT_PASS_H
#include <unordered_map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include "tools/common/graph_util.h"
#include "tools/converter/optimizer.h"

using mindspore::schema::TensorT;
namespace mindspore {
namespace lite {
class SelectPass : public GraphPass {
 public:
  explicit SelectPass(schema::MetaGraphT *graph) : graph_(graph) {}
  ~SelectPass() override = default;
  STATUS Run(schema::MetaGraphT *graph) override;
  STATUS RemoveSelectNodes();

 private:
  std::vector<uint32_t> select_indices_;
  schema::MetaGraphT *graph_ = nullptr;
};

class SingleSelectPass {
 public:
  SingleSelectPass(schema::MetaGraphT *graph, const size_t &node_index)
      : graph_(graph), select_node_index_(node_index) {}
  ~SingleSelectPass() = default;
  STATUS Run();

 private:
  STATUS Init();
  size_t InitThisGraphIndex();
  STATUS ConvertSelectToSwitch();
  std::unique_ptr<schema::TensorT> NewTensor(const std::unique_ptr<schema::TensorT> &in_tensor);
  void RemoveUselessNode(schema::CNodeT *partial_node);

  schema::MetaGraphT *graph_ = nullptr;
  schema::CNodeT *select_node_ = nullptr;
  size_t select_node_index_ = -1;
  int32_t this_subgraph_index_ = -1;
  const size_t kSelectMinInputSize = 2;
  const size_t kSelectMinOutputSize = 2;
};
}  // namespace lite
}  // namespace mindspore
#endif
