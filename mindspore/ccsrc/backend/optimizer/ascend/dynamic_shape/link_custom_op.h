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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_DYNAMIC_SHAPE_LINK_CUSTOM_OP_H
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_DYNAMIC_SHAPE_LINK_CUSTOM_OP_H

#include "ir/anf.h"
#include "backend/optimizer/common/optimizer.h"

namespace mindspore::opt::dynamic_shape {
class LinkCustomOp : public Pass {
 public:
  LinkCustomOp() : Pass("link_custom_op") {}
  ~LinkCustomOp() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::opt::dynamic_shape
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_DYNAMIC_SHAPE_CONVERT_GENERAL_OP_H
