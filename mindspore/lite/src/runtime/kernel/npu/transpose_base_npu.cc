/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/npu/transpose_base_npu.h"

namespace mindspore::kernel {
TransposeBaseNPUKernel::~TransposeBaseNPUKernel() {
  if (pre_trans_ != nullptr) {
    delete pre_trans_;
    pre_trans_ = nullptr;
  }
  if (post_trans_ != nullptr) {
    delete post_trans_;
    post_trans_ = nullptr;
  }
}

int TransposeBaseNPUKernel::SetPreTranspose(const ge::Operator *input) {
  // input permute: NHWC -> NCHW
  pre_trans_ = new (std::nothrow) hiai::op::Permute(name_ + "_pre_transpose");
  if (pre_trans_ == nullptr) {
    MS_LOG(ERROR) << "New pre transpose npu operator (NHWC -> NCHW) for op " << name_ << " failed.";
    return RET_ERROR;
  }
  pre_trans_->set_input_x(*input);
  pre_trans_->set_attr_order(ge::AttrValue::LIST_INT({0, 3, 1, 2}));
  return RET_OK;
}

int TransposeBaseNPUKernel::SetPostTranspose(const ge::Operator *input) {
  // permute: NCHW -> NHWC
  post_trans_ = new (std::nothrow) hiai::op::Permute(name_ + "_post_transpose");
  if (post_trans_ == nullptr) {
    MS_LOG(ERROR) << "New post transpose operator (NCHW -> NHWC) for op " << name_ << " failed.";
    return RET_ERROR;
  }
  post_trans_->set_input_x(*input);
  post_trans_->set_attr_order(ge::AttrValue::LIST_INT({0, 2, 3, 1}));
  return RET_OK;
}
}  // namespace mindspore::kernel
