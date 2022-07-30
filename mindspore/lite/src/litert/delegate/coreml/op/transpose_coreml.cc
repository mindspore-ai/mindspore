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

#include "src/litert/delegate/coreml/op/transpose_coreml.h"
namespace mindspore::lite {
int TransposeCoreMLOp::IsSupport() {
  MS_CHECK_GE(in_tensors_.size(), kInputSize1, RET_NOT_SUPPORT);
  auto perm_tensor = in_tensors_.at(1);
  if (!perm_tensor.IsConst()) {
    MS_LOG(WARNING) << "CoreML transpose must get fixed axis values.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int TransposeCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  auto transpose_param = op_->mutable_transpose();
  for (auto perm : perm_) {
    transpose_param->add_axes(perm);
  }
  return RET_OK;
}
}  // namespace mindspore::lite
