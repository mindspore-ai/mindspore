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

#include "src/litert/delegate/coreml/op/concat_coreml.h"
namespace mindspore::lite {
int ConcatCoreMLOp::IsSupport() {
  MS_CHECK_GE(in_tensors_.size(), kInputSize1, RET_NOT_SUPPORT);
  if (std::any_of(in_tensors_.begin(), in_tensors_.end(), [](mindspore::MSTensor &tensor) {
        return tensor.IsConst() && tensor.DataType() != DataType::kNumberTypeInt32 &&
               tensor.DataType() != DataType::kNumberTypeFloat32;
      })) {
    MS_LOG(ERROR) << "The datatype of CoreML Concat op's constant inputs must be int or float, op name: " << name_;
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ConcatCoreMLOp::InitParams() {
  concat_prim_ = op_primitive_->value_as_Concat();
  if (concat_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  axis_ = concat_prim_->axis();
  return RET_OK;
}

int ConcatCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  op_->mutable_concatnd()->set_axis(axis_);
  for (const auto &in_tensor : in_tensors_) {
    if (in_tensor.IsConst()) {
      auto ret = SetConstInput(in_tensor);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set const input failed for op: " << name_;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int ConcatCoreMLOp::HandleAxis() {
  axis_ = NCHW2NHWC_PERM[axis_];
  return RET_OK;
}

void ConcatCoreMLOp::SetMLOpInOut() {
  MS_ASSERT(op_ != nullptr);
  for (const auto &in_tensor : in_tensors_) {
    if (in_tensor.IsConst()) {
      // const op has not input
      const_ops_[in_tensor.Name()]->add_output(in_tensor.Name());
    }
    op_->add_input(in_tensor.Name());
  }
  op_->add_output(out_tensors_[0].Name());
}
}  // namespace mindspore::lite
