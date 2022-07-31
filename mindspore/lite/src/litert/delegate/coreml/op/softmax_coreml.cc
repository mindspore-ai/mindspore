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

#include "src/litert/delegate/coreml/op/softmax_coreml.h"
namespace mindspore::lite {
int SoftmaxCoreMLOp::InitParams() {
  softmax_prim_ = op_primitive_->value_as_Softmax();
  if (softmax_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(softmax_prim_->axis() != nullptr, RET_ERROR, "Softmax axis is null!");
  axis_ = static_cast<int>(*(softmax_prim_->axis()->begin()));
  return RET_OK;
}

int SoftmaxCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr && softmax_prim_ != nullptr);
  auto softmax_param = op_->mutable_softmaxnd();
  softmax_param->set_axis(axis_);
  return RET_OK;
}

int SoftmaxCoreMLOp::HandleAxis() {
  axis_ = NCHW2NHWC_PERM[axis_];
  return RET_OK;
}
}  // namespace mindspore::lite
