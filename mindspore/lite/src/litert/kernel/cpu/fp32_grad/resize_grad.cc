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
#include "src/litert/kernel/cpu/fp32_grad/resize_grad.h"
#include "nnacl/fp32_grad/resize_grad.h"
#include "nnacl/errorcode.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/nnacl_common.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ResizeGrad;

namespace mindspore::kernel {
float ResizeGradCPUKernel::Scaling(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / (static_cast<float>(out_size - 1))
                                         : in_size / (static_cast<float>(out_size));
}

int ResizeGradCPUKernel::ReSize() {
  auto param = reinterpret_cast<ResizeGradParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "ResizeGradCPUKernel op_parameter_ is nullptr";
    return RET_ERROR;
  }
  bool align_corners = param->align_corners_;
  param->in_height_ = static_cast<size_t>(in_tensors_.at(0)->Height());
  param->in_width_ = static_cast<size_t>(in_tensors_.at(0)->Width());
  param->out_height_ = static_cast<size_t>(out_tensors_.at(0)->Height());
  param->out_width_ = static_cast<size_t>(out_tensors_.at(0)->Width());
  param->height_scale_ = Scaling(param->out_height_, param->in_height_, align_corners);
  param->width_scale_ = Scaling(param->out_width_, param->in_width_, align_corners);

  return RET_OK;
}

int ResizeGradCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  CHECK_NULL_RETURN(op_parameter_);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ResizeGradCPUKernel::DoExecute(int task_id) {
  auto in_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto out_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto param = reinterpret_cast<ResizeGradParameter *>(op_parameter_);
  CHECK_NULL_RETURN(in_addr);
  CHECK_NULL_RETURN(out_addr);
  CHECK_NULL_RETURN(param);
  auto batch_size = in_tensors_.at(0)->Batch();
  auto channel = in_tensors_.at(0)->Channel();
  int error_code;
  if (param->method == static_cast<int>(schema::ResizeMethod_NEAREST)) {
    error_code = ResizeNearestNeighborGrad(in_addr, out_addr, batch_size, channel, in_tensors_.at(0)->format(), param);
  } else {
    error_code = ResizeBiLinearGrad(in_addr, out_addr, batch_size, channel, in_tensors_.at(0)->format(), param);
  }
  if (error_code != static_cast<int>(NNACL_OK)) {
    MS_LOG(ERROR) << "Resize fp32 grad failed.";
    return error_code;
  }
  return RET_OK;
}

int ResizeGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto resize_grad_kernel = reinterpret_cast<ResizeGradCPUKernel *>(cdata);
  CHECK_NULL_RETURN(resize_grad_kernel);
  auto error_code = resize_grad_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "resize grad error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeGradCPUKernel::Run() {
  auto out_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(out_addr);
  size_t elem_number = out_tensors_.at(0)->ElementsNum();
  std::fill(out_addr, out_addr + elem_number, 0.f);
  int error_code = ParallelLaunch(this->ms_context_, ResizeGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ResizeGradCPUKernel function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ResizeGrad, LiteKernelCreator<ResizeGradCPUKernel>)
}  // namespace mindspore::kernel
