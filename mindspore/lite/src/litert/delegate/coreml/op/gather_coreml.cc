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

#include "src/litert/delegate/coreml/op/gather_coreml.h"
namespace mindspore::lite {
int GatherCoreMLOp::IsSupport() {
  MS_CHECK_GE(in_tensors_.size(), kInputSize2, RET_NOT_SUPPORT);
  return RET_OK;
}

int GatherCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  auto gather_params = op_->mutable_gather();
  CHECK_NULL_RETURN(in_tensors_[THIRD_INPUT].Data());
  auto axis_data = reinterpret_cast<const int *>(in_tensors_[THIRD_INPUT].Data().get());
  gather_params->set_axis(axis_data[0]);
  auto indices_tensor = in_tensors_[SECOND_INPUT];
  if (indices_tensor.IsConst()) {
    auto ret = SetConstInput(indices_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set const input failed for op: " << name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void GatherCoreMLOp::SetMLOpInOut() {
  MS_ASSERT(op_ != nullptr);
  op_->add_input(in_tensors_[FIRST_INPUT].Name());
  auto indices_tensor = in_tensors_[SECOND_INPUT];
  if (indices_tensor.IsConst()) {
    const_ops_[indices_tensor.Name()]->add_output(indices_tensor.Name());
  }
  op_->add_input(indices_tensor.Name());
  op_->add_output(out_tensors_[0].Name());
}
}  // namespace mindspore::lite
