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

using mindspore::lite::converter::FmkType_TF;
using mindspore::schema::TensorT;
namespace mindspore {
namespace lite {

struct InferTensor {
  std::vector<uint32_t> in_nodes_; /* used current tensor as input */
  std::vector<uint32_t> out_nodes_;
  bool is_infer_;
};

class InferShapePass : public GraphPass {
 public:
  explicit InferShapePass(converter::FmkType fmk_type) : fmk_type_(fmk_type) {}
  ~InferShapePass() override = default;
  STATUS Run(MetaGraphT *graph) override;

 private:
  void InitSearchTensor(MetaGraphT *graph);
  void AddNextInferShapeNode(std::vector<uint32_t> output_tensor_node_indexes, size_t index);
  void AddOutputNode(uint32_t infer_node_index);

  lite::converter::FmkType fmk_type_ = FmkType_TF;
  MetaGraphT *graph_ = nullptr;
  std::vector<InferTensor> tensors_ = {};
  std::vector<uint32_t> infer_node_indexes_ = {};
  bool infer_interrupt_ = false;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_INFERSHAPE_PASS_H
