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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_GE_SPECIALIZED_PREPARE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_GE_SPECIALIZED_PREPARE_H_

#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class GeTensorArrayPrepare : public PatternProcessPass {
 public:
  explicit GeTensorArrayPrepare(bool multigraph = true) : PatternProcessPass("ge_tensor_array_prepare", multigraph) {}
  ~GeTensorArrayPrepare() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  // Add a const input with value `size` to TensorArray node
  static void TransformTASizeFromAttrToInput(const AnfNodePtr &node);
  static void InsertFlowOutputToTA(const AnfNodePtr &node);
};
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_GE_SPECIALIZED_PREPARE_H_
