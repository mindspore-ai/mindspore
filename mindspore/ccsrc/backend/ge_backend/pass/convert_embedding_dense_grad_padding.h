/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CONVERT_EMBEDDING_DENSE_GRAD_PADDING_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CONVERT_EMBEDDING_DENSE_GRAD_PADDING_H_

#include <vector>
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class ConvertEmbeddingDenseGradPadding : public PatternProcessPass {
 public:
  explicit ConvertEmbeddingDenseGradPadding(bool multi_graph = true)
      : PatternProcessPass("convert_embedding_dense_grad_padding", multi_graph) {}
  ~ConvertEmbeddingDenseGradPadding() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CONVERT_EMBEDDING_DENSE_GRAD_PADDING_H_
