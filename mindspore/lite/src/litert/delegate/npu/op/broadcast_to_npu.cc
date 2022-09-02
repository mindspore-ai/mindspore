/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/op/broadcast_to_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"
#include "src/litert/delegate/npu/npu_manager.h"

namespace mindspore::lite {
int BroadcastToNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                                const std::vector<mindspore::MSTensor> &out_tensors) {
  MS_CHECK_GE(in_tensors.size(), kInputSize1, RET_NOT_SUPPORT);
  if (!NPUManager::CheckDDKVerGreatEqual("100.500.010.010")) {
    MS_LOG(WARNING) << "BroadcastTo is not supported for HiAI rom version lower than 100.500.010.010.";
    return RET_NOT_SUPPORT;
  }
  if (!in_tensors[1].IsConst()) {
    MS_LOG(WARNING) << "Not support non-const shape tensor for NPU BroadcastTo op: " << name_;
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int BroadcastToNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  broadcast_to_ = new (std::nothrow) hiai::op::BroadcastTo(name_);
  if (broadcast_to_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

int BroadcastToNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                   const std::vector<mindspore::MSTensor> &out_tensors,
                                   const std::vector<ge::Operator *> &npu_inputs) {
  MS_CHECK_GE(npu_inputs.size(), kInputSize1, RET_NOT_SUPPORT);
  broadcast_to_->set_input_x(*npu_inputs[0]);
  broadcast_to_->set_input_shape(*npu_inputs[1]);
  return RET_OK;
}

ge::Operator *BroadcastToNPUOp::GetNPUOp() { return this->broadcast_to_; }

BroadcastToNPUOp::~BroadcastToNPUOp() {
  if (broadcast_to_ != nullptr) {
    delete broadcast_to_;
    broadcast_to_ = nullptr;
  }
}
}  // namespace mindspore::lite
