
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

#include "src/litert/kernel/cpu/fp32_grad/strided_slice_grad.h"
#include <vector>
#include <algorithm>
#include <utility>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32_grad/strided_slice_grad.h"
#include "src/common/ops/populate/strided_slice_populate.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_StridedSliceGrad;

namespace mindspore::kernel {
int StridedSliceGradCPUKernel::Prepare() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  param_ = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param_);
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  if (input->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Not supported data type: " << input->data_type();
    return RET_ERROR;
  }
  param_->data_type = ::kNumberTypeFloat32;
  return ReSize();
}

void StridedSliceGradCPUKernel::FillEmptyDims() {
  int32_t begins[DIMENSION_8D];
  int32_t ends[DIMENSION_8D];
  int32_t strides[DIMENSION_8D];
  int32_t i;

  // invert the order of the dimension and fill defout outsize actual ranae
  for (i = 0; i < DIMENSION_8D; ++i) {
    begins[i] = param_->begins_[i];
    ends[i] = param_->ends_[i];
    strides[i] = param_->strides_[i];
  }

  int out_shape_length = in_tensors_.at(1)->shape().at(0);
  int32_t real_index = out_shape_length - 1;
  for (i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param_->begins_[i] = begins[real_index];
      param_->ends_[i] = ends[real_index];
      param_->strides_[i] = strides[real_index--];
    } else {
      param_->begins_[i] = 0;
      param_->ends_[i] = 1;
      param_->strides_[i] = 1;
    }
  }
  for (i = 0; i < DIMENSION_8D; ++i) {
    int ax = param_->ends_[i] - param_->begins_[i];
    if (ax < 0) {
      ax = 0;
    }
    param_->in_shape_[i] = ax;
  }
  param_->num_axes_ = DIMENSION_8D;
  param_->in_shape_length_ = DIMENSION_8D;
}

void StridedSliceGradCPUKernel::FillOutputDim() {
  auto output = out_tensors_.at(0);
  size_t out_size = output->shape().size();
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    if (i < out_size) {
      output_shape_.push_back(output->shape()[i]);
    } else {
      output_shape_.insert(output_shape_.begin(), 1);
    }
  }
}

int StridedSliceGradCPUKernel::ReSize() {
  FillEmptyDims();
  FillOutputDim();
  for (int32_t i = 0; i < DIMENSION_8D; ++i) {
    if (param_->ends_[i] == 0 && param_->begins_[i] < 0) {
      param_->ends_[i] += output_shape_[i];
    }
    if (param_->ends_[i] < 0) {
      param_->ends_[i] = (param_->ends_[i] + output_shape_[i]) < 0 ? 0 : param_->ends_[i] + output_shape_[i];
    }
    if (param_->ends_[i] > output_shape_[i]) {
      param_->ends_[i] = output_shape_[i];
    }
    if (param_->begins_[i] < 0) {
      auto k = param_->begins_[i] + output_shape_[i];
      param_->begins_[i] = k < 0 ? 0 : k;
    }
    if (param_->begins_[i] > output_shape_[i]) {
      param_->begins_[i] = output_shape_[i];
    }
  }
  return RET_OK;
}

int StridedSliceGradImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto slice = reinterpret_cast<StridedSliceGradCPUKernel *>(cdata);
  auto error_code = slice->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "StridedSliceGrad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, StridedSliceGradImpl, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Strided slice error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceGradCPUKernel::DoExecute(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);

  auto *dx = reinterpret_cast<float *>(output->MutableData());
  auto *dy = reinterpret_cast<float *>(input->MutableData());
  CHECK_NULL_RETURN(dx);
  CHECK_NULL_RETURN(dy);
  return CalStridedSliceGrad(dy, dx);
}

int StridedSliceGradCPUKernel::CalStridedSliceGrad(float *input, float *output) {
  int input_num = 1;
  for (int le = 0; le < DIMENSION_8D; le++) {
    input_num = input_num * param_->in_shape_[le];
  }
  int output_num = 1;
  for (int len = 0; len < DIMENSION_8D; len++) {
    output_num = output_num * output_shape_[len];
  }

  if (input_num == 0) {
    res_arr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(sizeof(float) * output_num));
    for (int res_len = 0; res_len < output_num; res_len++) {
      res_arr_[res_len] = static_cast<float>(0);
    }
    memcpy(output, res_arr_, output_num * sizeof(float));
    FreeRunBuffer();
    return RET_OK;
  }

  int temp_num = input_num;
  int max_num = input_num;
  int step = 1;
  for (int i = DIMENSION_8D - 1; i >= 0; --i) {
    temp_num = static_cast<int>(temp_num * output_shape_[i] / param_->in_shape_[i]);
    max_num = MSMAX(max_num, temp_num);
  }
  temp_input_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(sizeof(float) * max_num));
  memset(temp_input_, 0, max_num * sizeof(float));
  memcpy(temp_input_, input, input_num * sizeof(float));
  temp_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(max_num * sizeof(float)));
  temp_num = input_num;
  for (int i = DIMENSION_8D - 1; i >= 0; --i) {
    temp_num = static_cast<int>(temp_num * output_shape_[i] / param_->in_shape_[i]);
    memset(temp_, 0, sizeof(float) * temp_num);
    int start1 = 0;
    int start2 = 0;
    while (start1 < temp_num) {
      int id = 0;
      for (int k = param_->begins_[i]; param_->strides_[i] > 0 ? k < param_->ends_[i] : k > param_->ends_[i];
           k += param_->strides_[i], id++) {
        memcpy(temp_ + start1 + k * step, temp_input_ + start2 + id * step, step * sizeof(float));
      }
      start1 += output_shape_[i] * step;
      start2 += param_->in_shape_[i] * step;
    }
    step *= output_shape_[i];
    std::swap(temp_input_, temp_);
  }
  memcpy(output, temp_input_, output_num * sizeof(float));
  FreeRunBuffer();
  return RET_OK;
}

void StridedSliceGradCPUKernel::FreeRunBuffer() {
  if (res_arr_ != nullptr) {
    ms_context_->allocator->Free(res_arr_);
    res_arr_ = nullptr;
  }
  if (temp_input_ != nullptr) {
    ms_context_->allocator->Free(temp_input_);
    temp_input_ = nullptr;
  }
  if (temp_ != nullptr) {
    ms_context_->allocator->Free(temp_);
    temp_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_StridedSliceGrad, LiteKernelCreator<StridedSliceGradCPUKernel>)
}  // namespace mindspore::kernel
