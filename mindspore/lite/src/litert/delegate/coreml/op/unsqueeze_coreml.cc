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

#include "src/litert/delegate/coreml/op/unsqueeze_coreml.h"
namespace mindspore::lite {
int UnsqueezeCoreMLOp::InitParams() {
  unsqueeze_prim_ = op_primitive_->value_as_Unsqueeze();
  if (unsqueeze_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int UnsqueezeCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr && unsqueeze_prim_ != nullptr);
  auto expanddims_param = op_->mutable_expanddims();
  MS_CHECK_TRUE_MSG(unsqueeze_prim_->axis() != nullptr, RET_ERROR, "Unsqueeze axis is null!");
  auto axes = std::vector<int>(unsqueeze_prim_->axis()->begin(), unsqueeze_prim_->axis()->end());
  for (auto axis : axes) {
    expanddims_param->add_axes(axis);
  }
  return RET_OK;
}
}  // namespace mindspore::lite
