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

#include "backend/common/pass/insert_tensor_move_for_ref.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kNopNodeRealInputIndex = 1;
}

bool InsertTensorMoveForGraphOutputRefNode::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) == kOptimizeO0) {
    // not use somas
    return true;
  }

  // Need to insert TensorMove if the output of RefOp is GraphOutput
  (void)InsertRefTensorMoveForGraphOutput(graph);
  return true;
}
}  // namespace opt
}  // namespace mindspore
