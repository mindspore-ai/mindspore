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

#include "src/runtime/kernel/npu/crop_and_resize_npu.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_CropAndResize;

namespace mindspore::kernel {
int CropAndResizeNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  // support only 0 linear and 1 nearest
  if (param_->method_ != 0 && param_->method_ != 1) {
    MS_LOG(WARNING) << "NPU CropAndResize only support method bilinear 0 and nearest 1, got " << param_->method_;
    return RET_ERROR;
  }
  return RET_OK;
}

int CropAndResizeNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                         const std::vector<lite::Tensor *> &outputs,
                                         const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::CropAndResize(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  if (npu_inputs.size() < 4) {
    MS_LOG(ERROR) << "NPU CropAndResize got nput inputs size < 4";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_boxes(*npu_inputs[1]);
  op_->set_input_box_index(*npu_inputs[2]);
  op_->set_input_crop_size(*npu_inputs[3]);
  op_->set_attr_extrapolation_value(param_->extrapolation_value_);
  if (param_->method_ == 0) {
    op_->set_attr_method("bilinear");
  } else if (param_->method_ == 1) {
    op_->set_attr_method("nearest");
  } else {
    MS_LOG(ERROR) << "NPU CropAndResize only support method bilinear and nearest";
    return RET_ERROR;
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::CropAndResizeNPUKernel::GetNPUOp() { return this->op_; }

CropAndResizeNPUKernel::~CropAndResizeNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_CropAndResize, NPUKernelCreator<CropAndResizeNPUKernel>)
}  // namespace mindspore::kernel
