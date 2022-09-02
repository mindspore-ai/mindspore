/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/op/gather_npu.h"
#include "src/litert/delegate/npu/npu_manager.h"

namespace mindspore::lite {
constexpr int AXIS_INDEX = 2;
constexpr int GATHER_INPUT_SIZE = 3;

int GatherNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_LESS_RETURN(in_tensors.size(), GATHER_INPUT_SIZE);
  if (in_tensors[1].DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(WARNING) << "Gather indices only support Int32";
    return RET_NOT_SUPPORT;
  }
  // npu rom version lower than that specified below has compatibility problem. The version threshold is not absolute
  // and may need to be adjusted later.
  if (in_tensors[1].ElementNum() > 1 && !NPUManager::CheckDDKVerGreatEqual("100.320.012.043")) {
    MS_LOG(WARNING) << "Gather indices number larger than 1 is not supported for current NPU ddk version.";
    return RET_NOT_SUPPORT;
  }
  if (in_tensors.size() >= GATHER_INPUT_SIZE && in_tensors[AXIS_INDEX].ElementNum() == 1) {
    MS_ASSERT(in_tensors[AXIS_INDEX].Data());
    axis_ = static_cast<const int *>(in_tensors[AXIS_INDEX].Data().get())[0];
  } else {
    MS_LOG(WARNING) << "NPU axis is attribute.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int GatherNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  gather_ = new (std::nothrow) hiai::op::GatherV2D(name_);
  if (gather_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  gather_->set_attr_axis(axis_);
  return RET_OK;
}

int GatherNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors,
                              const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_LESS_RETURN(npu_inputs.size(), GATHER_INPUT_SIZE - 1);
  gather_->set_input_x(*npu_inputs[0]);
  gather_->set_input_indices(*npu_inputs[1]);
  return RET_OK;
}

ge::Operator *GatherNPUOp::GetNPUOp() { return this->gather_; }

GatherNPUOp::~GatherNPUOp() {
  if (gather_ != nullptr) {
    delete gather_;
    gather_ = nullptr;
  }
}
}  // namespace mindspore::lite
