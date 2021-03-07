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

#ifndef MINDSPORE_PREDICT_ISOLATED_SUBGRAPH_TENSOR_PASS_H
#define MINDSPORE_PREDICT_ISOLATED_SUBGRAPH_TENSOR_PASS_H

#include <vector>
#include <utility>
#include "tools/converter/optimizer.h"

namespace mindspore {
namespace lite {
class SubgraphTensorPass : public GraphPass {
 public:
  SubgraphTensorPass() = default;

  ~SubgraphTensorPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  STATUS RemoveUselessTensors(schema::MetaGraphT *graph);
  bool IsUsing(schema::MetaGraphT *graph, const uint32_t &tensor_idx);
  STATUS UpdateTensorIdx(schema::MetaGraphT *graph, const uint32_t &tensor_idx);
  STATUS SyncMainGraphInputAndOutput(schema::MetaGraphT *graph);

  template <typename T>
  void UpdateVec(std::vector<T> *vec, T element) {
    for (auto iter = vec->begin(); iter != vec->end(); iter++) {
      if (*iter > element) {
        (*iter)--;
      }
    }
  }
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_ISOLATED_NODE_REMOVE_PASS_H
