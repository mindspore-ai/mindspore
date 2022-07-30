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

#include "src/litert/kernel/cpu/fp16_grad/strided_slice_fp16_grad.h"
#include <vector>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp16_grad/strided_slice_grad.h"
#include "src/common/ops/populate/strided_slice_populate.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_StridedSliceGrad;

namespace mindspore::kernel {
int StridedSliceGradCPUKernelFp16::Prepare() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  param_ = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param_);
  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(out_tensors_.at(0));
  if (input->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Not supported data type: " << input->data_type();
    return RET_ERROR;
  }
  param_->data_type = ::kNumberTypeFloat16;
  FillEmptyDims();
  FillOutputDim();
  return ReSize();
}

void StridedSliceGradCPUKernelFp16::FillEmptyDims() {
  int32_t begins[DIMENSION_7D];
  int32_t ends[DIMENSION_7D];
  int32_t strides[DIMENSION_7D];
  int32_t input_shape[DIMENSION_7D];
  int32_t i;
  for (i = 0; i < param_->num_axes_; ++i) {
    begins[i] = param_->begins_[i];
    ends[i] = MSMIN(param_->ends_[i], param_->in_shape_[i]);
    strides[i] = param_->strides_[i];
    input_shape[i] = param_->in_shape_[i];
  }
  for (i = param_->num_axes_; i < param_->in_shape_length_; ++i) {
    input_shape[i] = param_->in_shape_[i];
    begins[i] = 0;
    ends[i] = param_->in_shape_[i];
    strides[i] = 1;
  }

  int32_t real_index = param_->in_shape_length_ - 1;
  for (i = DIMENSION_7D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param_->begins_[i] = begins[real_index];
      param_->ends_[i] = ends[real_index];
      param_->strides_[i] = strides[real_index];
      param_->in_shape_[i] = input_shape[real_index--];
    } else {
      param_->begins_[i] = 0;
      param_->ends_[i] = 1;
      param_->strides_[i] = 1;
      param_->in_shape_[i] = 1;
    }
  }
  param_->num_axes_ = DIMENSION_7D;
  param_->in_shape_length_ = DIMENSION_7D;

  for (i = 0; i < DIMENSION_7D; ++i) {
    if (param_->begins_[i] < 0) {
      param_->begins_[i] += param_->in_shape_[i];
    }
    if (param_->ends_[i] < 0) {
      param_->ends_[i] += param_->in_shape_[i];
    }
  }
}

void StridedSliceGradCPUKernelFp16::FillOutputDim() {
  auto output = out_tensors_.at(0);
  size_t out_size = output->shape().size();
  for (size_t i = 0; i < DIMENSION_7D; i++) {
    if (i < out_size) {
      output_shape_.push_back(output->shape()[i]);
    } else {
      output_shape_.insert(output_shape_.begin(), 1);
    }
  }
}

int StridedSliceGradCPUKernelFp16::ReSize() { return RET_OK; }

int StridedSliceFp16GradImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto slice = reinterpret_cast<StridedSliceGradCPUKernelFp16 *>(cdata);
  auto error_code = slice->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "StridedSliceGrad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceGradCPUKernelFp16::Run() {
  int error_code = ParallelLaunch(this->ms_context_, StridedSliceFp16GradImpl, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Strided slice error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceGradCPUKernelFp16::DoExecute(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  int *po = output_shape_.data();
  auto dx = reinterpret_cast<float16_t *>(output->data());
  auto dy = reinterpret_cast<float16_t *>(input->data());
  CHECK_NULL_RETURN(po);
  CHECK_NULL_RETURN(dx);
  CHECK_NULL_RETURN(dy);

  std::fill(dx, dx + output->ElementsNum(), 0.f);
  auto ret = DoStridedSliceFp16Grad(dy, dx, po, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSliceGrad error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_StridedSliceGrad, LiteKernelCreator<StridedSliceGradCPUKernelFp16>)
}  // namespace mindspore::kernel
