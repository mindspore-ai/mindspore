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
#include "src/runtime/kernel/arm/fp32/roi_pooling.h"
#include "nnacl/fp32/roi_pooling.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ROIPooling;

namespace mindspore::kernel {

int ROIPoolingCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ROIPoolingCPUKernel::ReSize() {
  auto in_shape = in_tensors_.front()->shape();
  auto out_shape = out_tensors_.front()->shape();
  int ndims = in_shape.size();
  if (ndims > 4) {
    MS_LOG(ERROR) << "ROIPooling ReSzie error ,shape dim greater than 4!";
    return RET_ERROR;
  }
  param_->ndim_ = ndims;
  param_->input_n_ = in_shape[0];
  param_->input_h_ = in_shape[1];
  param_->input_w_ = in_shape[2];
  param_->input_c_ = in_shape[3];
  param_->output_n_ = out_shape[0];
  param_->output_h_ = out_shape[1];
  param_->output_w_ = out_shape[2];
  param_->output_c_ = out_shape[3];
  param_->in_strides_[ndims - 1] = 1;
  param_->out_strides_[ndims - 1] = 1;
  for (int i = ndims - 2; i >= 0; --i) {
    param_->in_strides_[i] = in_shape[i + 1] * param_->in_strides_[i + 1];
    param_->out_strides_[i] = out_shape[i + 1] * param_->out_strides_[i + 1];
  }
  param_->thread_num_ = MSMIN(param_->op_parameter_.thread_num_, out_shape[0]);
  return RET_OK;
}

int ROIPoolingCPUKernel::DoExecute(int task_id) {
  auto ret = ROIPooling(in_ptr_, out_ptr_, roi_ptr_, task_id, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ROIPooling Execute error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ROIPoolingRun(void *cdata, int task_id) {
  auto Data = reinterpret_cast<ROIPoolingCPUKernel *>(cdata);
  auto ret = Data->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ROIPooling Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ROIPoolingCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail! ret: " << ret;
    return ret;
  }
  in_ptr_ = reinterpret_cast<float *>(in_tensors_.front()->Data());
  out_ptr_ = reinterpret_cast<float *>(out_tensors_.front()->Data());
  roi_ptr_ = reinterpret_cast<float *>(in_tensors_.at(1)->Data());
  ret = ParallelLaunch(THREAD_POOL_DEFAULT, ROIPoolingRun, this, param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ROIPooling error: error_code[" << ret << "]";
    return ret;
  }
  return ret;
}

kernel::LiteKernel *CpuROIPoolingFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    return nullptr;
  }
  if (ctx->thread_num_ == 0) {
    MS_LOG(ERROR) << "context thread num is 0!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ROIPoolingCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ROIPoolingCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ROIPooling, CpuROIPoolingFp32KernelCreator)
}  // namespace mindspore::kernel
