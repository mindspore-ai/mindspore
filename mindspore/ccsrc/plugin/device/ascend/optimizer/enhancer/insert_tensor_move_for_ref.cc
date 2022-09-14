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

#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_ref.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kNopNodeRealInputIndex = 1;
}

bool InsertTensorMoveForGraphOutputRefNode::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto task_sink = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto opt_level = ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL);
  if (!task_sink && (opt_level == kOptimizeO0)) {
    // not use somas
    return false;
  }

  // Need to insert TensorMove if the output of RefOp is GraphOutput
  auto tensor_move_list = InsertRefTensorMoveForGraphOutput(graph);
  for (auto &tensor_move : tensor_move_list) {
    kernel_select_->SelectKernel(tensor_move->cast<CNodePtr>());
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
