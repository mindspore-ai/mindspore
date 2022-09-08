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

#include "src/litert/delegate/coreml/op/split_coreml.h"
namespace mindspore::lite {
int SplitCoreMLOp::InitParams() {
  split_prim_ = op_primitive_->value_as_Split();
  if (split_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  axis_ = static_cast<int>(split_prim_->axis());
  split_num_ = static_cast<int>(split_prim_->output_num());
  return RET_OK;
}

int SplitCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr && split_prim_ != nullptr);
  auto split_param = op_->mutable_splitnd();
  split_param->set_numsplits(split_num_);
  split_param->set_axis(axis_);
  auto split_sizes = split_prim_->size_splits();
  if (split_sizes != nullptr) {
    for (int i = 0; i < split_num_; ++i) {
      split_param->add_splitsizes(split_sizes->Get(i));
    }
  }
  return RET_OK;
}

int SplitCoreMLOp::HandleAxis() {
  axis_ = NCHW2NHWC_PERM[axis_];
  return RET_OK;
}
}  // namespace mindspore::lite
