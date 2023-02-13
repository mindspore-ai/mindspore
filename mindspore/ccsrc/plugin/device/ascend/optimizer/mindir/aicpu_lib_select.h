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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_AICPU_LIB_SELECT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_AICPU_LIB_SELECT_H_

#include "backend/common/optimizer/inplace_node_pass.h"

namespace mindspore {
namespace opt {
class AICpuLibSelectPass : public InplaceNodePass {
 public:
  AICpuLibSelectPass() : InplaceNodePass("env_op_attr_update") {}
  bool Process(const AnfNodePtr &node) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_AICPU_LIB_SELECT_H_
