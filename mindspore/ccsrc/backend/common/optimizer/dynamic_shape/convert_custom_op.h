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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_CONVERT_CUSTOM_OP_H
#define MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_CONVERT_CUSTOM_OP_H

#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::opt::dynamic_shape {
class ConvertCustomOp : public Pass {
 public:
  ConvertCustomOp() : Pass("convert_custom_op") {}
  ~ConvertCustomOp() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void ConvertCustomOpForNode(const AnfNodePtr &node) const;
};
}  // namespace mindspore::opt::dynamic_shape
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_CONVERT_CUSTOM_OP_H
