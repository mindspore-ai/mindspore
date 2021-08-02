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

#include <vector>
#include "src/runtime/kernel/arm/fp16_grad/resize_fp16_grad.h"
#include "nnacl/fp16_grad/resize_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ResizeGrad;

namespace mindspore::kernel {
float16_t ScalingFp16(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float16_t>(out_size - 1)
                                         : in_size / static_cast<float16_t>(out_size);
}

int ResizeGradCPUKernelFp16::ReSize() {
  auto param = reinterpret_cast<ResizeFp16GradParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "ResizeGradCPUKernelFp16 op_parameter_ is nullptr";
    return RET_ERROR;
  }
  bool align_corners = param->align_corners_;
  param->in_height_ = static_cast<size_t>(in_tensors_.at(0)->Height());
  param->in_width_ = static_cast<size_t>(in_tensors_.at(0)->Width());
  param->out_height_ = static_cast<size_t>(out_tensors_.at(0)->Height());
  param->out_width_ = static_cast<size_t>(out_tensors_.at(0)->Width());
  param->height_scale_ = ScalingFp16(param->out_height_, param->in_height_, align_corners);
  param->width_scale_ = ScalingFp16(param->out_width_, param->in_width_, align_corners);
  return RET_OK;
}

int ResizeGradCPUKernelFp16::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ResizeGradCPUKernelFp16::Execute(int task_id) {
  auto in_addr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c());
  auto out_addr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data_c());
  auto param = reinterpret_cast<ResizeFp16GradParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "ResizeGradCPUKernelFp16 op_parameter_ is nullptr";
    return RET_ERROR;
  }
  auto batch_size = in_tensors_.at(0)->Batch();
  auto channel = in_tensors_.at(0)->Channel();
  if (param->method == static_cast<int>(schema::ResizeMethod_NEAREST)) {
    ResizeNearestNeighborFp16Grad(in_addr, out_addr, batch_size, channel, in_tensors_.at(0)->format(), param);
  } else {
    ResizeBiLinearFp16Grad(in_addr, out_addr, batch_size, channel, in_tensors_.at(0)->format(), param);
  }
  return RET_OK;
}

int ResizeFp16GradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto resize_grad_kernel = reinterpret_cast<ResizeGradCPUKernelFp16 *>(cdata);
  auto error_code = resize_grad_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "resize grad error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeGradCPUKernelFp16::Run() {
  auto out_addr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data_c());
  size_t elem_number = out_tensors_.at(0)->ElementsNum();
  std::fill(out_addr, out_addr + elem_number, 0.f);
  int error_code = ParallelLaunch(this->ms_context_, ResizeFp16GradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ResizeGradCPUKernelFp16 function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ResizeGrad, LiteKernelCreator<ResizeGradCPUKernelFp16>)
}  // namespace mindspore::kernel
