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

#include "src/runtime/kernel/npu/resize_npu.h"
#include <memory>
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::kernel {
int ResizeNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               OpParameter *opParameter) {
  if (resize_parameter_->method_ != schema::ResizeMethod_LINEAR &&
      resize_parameter_->method_ != schema::ResizeMethod_NEAREST) {
    MS_LOG(WARNING) << "Unsupported resize method type:" << resize_parameter_->method_;
    return RET_ERROR;
  }
  if (inputs[0]->Height() > outputs[0]->Height() || inputs[0]->Width() > outputs[0]->Width()) {
    MS_LOG(WARNING) << "Npu resize does not support reduction.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  ge::TensorDesc sizeTensorDesc(ge::Shape({2}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr sizeTensor = std::make_shared<hiai::Tensor>(sizeTensorDesc);
  vector<int32_t> dataValue = {static_cast<int32_t>(resize_parameter_->new_height_),
                               static_cast<int32_t>(resize_parameter_->new_width_)};
  sizeTensor->SetData(reinterpret_cast<uint8_t *>(dataValue.data()), 2 * sizeof(int32_t));
  out_size_ = new (std::nothrow) hiai::op::Const(name_ + "_size");
  out_size_->set_attr_value(sizeTensor);
  if (resize_parameter_->method_ == schema::ResizeMethod_LINEAR) {
    auto op = new (std::nothrow) hiai::op::ResizeBilinearV2(name_);
    if (op == nullptr) {
      MS_LOG(ERROR) << " op is nullptr.";
      return RET_ERROR;
    }
    op->set_attr_align_corners(resize_parameter_->coordinate_transform_mode_ ==
                               schema::CoordinateTransformMode_ALIGN_CORNERS);
    op->set_input_x(*npu_inputs[0]);
    op->set_input_size(*out_size_);
    op->set_attr_half_pixel_centers(resize_parameter_->preserve_aspect_ratio_);
    op_ = op;
  } else if (resize_parameter_->method_ == schema::ResizeMethod_NEAREST) {
    auto op = new (std::nothrow) hiai::op::ResizeNearestNeighborV2(name_);
    if (op == nullptr) {
      MS_LOG(ERROR) << " op is nullptr.";
      return RET_ERROR;
    }
    op->set_attr_align_corners(resize_parameter_->coordinate_transform_mode_ ==
                               schema::CoordinateTransformMode_ALIGN_CORNERS);
    op->set_input_x(*npu_inputs[0]);
    op->set_input_size(*out_size_);
    op_ = op;
  } else {
    MS_LOG(WARNING) << "Unsupported resize method type:" << resize_parameter_->method_;
    return RET_ERROR;
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::ResizeNPUKernel::GetNPUOp() { return this->op_; }

ResizeNPUKernel::~ResizeNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (out_size_ != nullptr) {
    delete out_size_;
    out_size_ = nullptr;
  }
}
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Resize, NPUKernelCreator<ResizeNPUKernel>)
}  // namespace mindspore::kernel
