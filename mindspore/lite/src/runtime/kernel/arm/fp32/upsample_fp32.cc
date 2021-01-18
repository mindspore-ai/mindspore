/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/upsample_fp32.h"
#include <algorithm>
#include "nnacl/fp32/resize_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Upsample;

namespace mindspore::kernel {
int UpsampleCPUKernel::Init() {
  param_ = reinterpret_cast<UpsampleParameter *>(op_parameter_);
  MS_ASSERT(param_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int UpsampleCPUKernel::ReSize() {
  auto ret = RET_OK;
  auto out_tensor = out_tensors_.at(0);
  MS_ASSERT(out_tensor);
  auto out_shape = out_tensor->shape();
  if (out_shape.size() != 4) {
    MS_LOG(ERROR) << "Upsample out tensor dim should be 4";
    return RET_ERROR;
  }
  new_height_ = out_shape.at(1);
  new_width_ = out_shape.at(2);

  if (param_->method_ == 0) {  // bilinear
    FreeTmpBuffer();
    ret = MallocTmpBuffer();
    if (ret != RET_OK) {
      FreeTmpBuffer();
      return ret;
    }

    auto input = in_tensors_.at(0);
    MS_ASSERT(input);
    auto input_shape = input->shape();
    auto output = out_tensors().at(0);
    MS_ASSERT(output);
    auto output_shape = output->shape();
    ret = PrepareResizeBilinear(input_shape.data(), output_shape.data(), CalculateAsymmetric, y_bottoms_, y_tops_,
                                x_lefts_, x_rights_, y_bottom_weights_, x_left_weights_);
    if (ret != RET_OK) {
      FreeTmpBuffer();
    }
  }
  return ret;
}

int UpsampleImpl(void *cdata, int task_id) {
  auto upsample_kernel = reinterpret_cast<UpsampleCPUKernel *>(cdata);
  auto error_code = upsample_kernel->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Upsample Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int UpsampleCPUKernel::RunImpl(int task_id) {
  MS_ASSERT(in_tensors_.size() == 2);
  auto input = in_tensors_.at(0);  // input to be upsampled(resized)
  auto input_data = reinterpret_cast<float *>(input->data_c());
  MS_ASSERT(input_data);

  auto out_tensor = out_tensors_.at(0);
  MS_ASSERT(out_tensor);
  auto output_data = reinterpret_cast<float *>(out_tensor->data_c());
  MS_ASSERT(output_data);
  auto input_shape = input->shape();

  int ret = 0;
  switch (param_->method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      int n_h_begin, n_h_end;
      int n = out_tensor->shape().at(0);
      int h = new_height_;
      int unit = UP_DIV(n * h, context_->thread_num_);
      n_h_begin = unit * task_id;
      n_h_end = std::min(n_h_begin + unit, n * h);
      int c = in_tensors_.at(0)->shape().at(3);
      float *line0 = line_buffer_ + new_width_ * c * 2 * task_id;
      float *line1 = line0 + new_width_ * c;
      ret = ResizeBilinear(input_data, output_data, input_shape.data(), out_tensor->shape().data(), y_bottoms_, y_tops_,
                           x_lefts_, x_rights_, y_bottom_weights_, x_left_weights_, line0, line1, n_h_begin, n_h_end);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      ret = ResizeNearestNeighbor(input_data, output_data, input_shape.data(), out_tensor->shape().data(),
                                  CalculateAsymmetric, coordinate_transform_mode_, task_id, context_->thread_num_);
      break;
    }
    default: {
      MS_LOG(ERROR) << "Upsample unknown method " << param_->method_;
      ret = RET_ERROR;
    }
  }
  return ret;
}

int UpsampleCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, UpsampleImpl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Upsample run error, error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  return RET_OK;
}
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Upsample, LiteKernelCreator<UpsampleCPUKernel>)
}  // namespace mindspore::kernel
