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

#include "src/runtime/kernel/npu/strided_slice_npu.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::kernel {
int StridedSliceNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  // Only onnx StridedSlice has 5 inputs, of which the 4th input is axes and the 5th input is strides.
  if (inputs.size() == 5) {
    vector<int> axes;
    size_t size = inputs[3]->shape()[0];
    axes.resize(size);
    memcpy(axes.data(), inputs[3]->data_c(), sizeof(int) * size);
    for (int i = 0; i < axes.size(); ++i) {
      if (i != axes[i]) {
        MS_LOG(ERROR) << "Does not support setting axis, so the axis must be continuous.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int StridedSliceNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                        const std::vector<lite::Tensor *> &outputs,
                                        const std::vector<ge::Operator *> &npu_inputs) {
  // StridedSliceV2 supports setting axes, but it will cause an endless loop.
  op_ = new (std::nothrow) hiai::op::StridedSlice(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_begin(*npu_inputs[1]);
  op_->set_input_end(*npu_inputs[2]);

  // The strides position of onnx is the 5th, and the others are the 4th.
  if (npu_inputs.size() == 5) {
    op_->set_input_strides(*npu_inputs[4]);
  } else {
    op_->set_input_strides(*npu_inputs[3]);
  }
  op_->set_attr_begin_mask(param_->begins_mask_);
  op_->set_attr_ellipsis_mask(param_->ellipsisMask_);
  op_->set_attr_end_mask(param_->ends_mask_);
  op_->set_attr_shrink_axis_mask(param_->shrinkAxisMask_);
  op_->set_attr_new_axis_mask(param_->newAxisMask_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::StridedSliceNPUKernel::GetNPUOp() { return this->op_; }

StridedSliceNPUKernel::~StridedSliceNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, NPUKernelCreator<StridedSliceNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeInt32, PrimitiveType_StridedSlice, NPUKernelCreator<StridedSliceNPUKernel>)
}  // namespace mindspore::kernel
