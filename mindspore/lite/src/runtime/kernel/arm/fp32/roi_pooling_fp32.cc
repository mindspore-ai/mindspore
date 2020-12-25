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
#include "src/runtime/kernel/arm/fp32/roi_pooling_fp32.h"
#include "nnacl/fp32/roi_pooling_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
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
  if (max_c_ != nullptr) {
    free(max_c_);
    max_c_ = nullptr;
  }
  auto in_shape = in_tensors_.front()->shape();
  auto out_shape = out_tensors_.front()->shape();
  int ndims = in_shape.size();
  if (ndims > 4) {
    MS_LOG(ERROR) << "ROIPooling ReSzie error ,shape dim greater than 4!";
    return RET_ERROR;
  }
  param_->ndim_ = ndims;
  param_->input_n_ = in_shape.at(0);
  param_->input_h_ = in_shape.at(1);
  param_->input_w_ = in_shape.at(2);
  param_->input_c_ = in_shape.at(3);
  param_->output_n_ = out_shape.at(0);
  param_->output_h_ = out_shape.at(1);
  param_->output_w_ = out_shape.at(2);
  param_->output_c_ = out_shape.at(3);
  param_->in_strides_[ndims - 1] = 1;
  param_->out_strides_[ndims - 1] = 1;
  for (int i = ndims - 2; i >= 0; --i) {
    param_->in_strides_[i] = in_shape.at(i + 1) * param_->in_strides_[i + 1];
    param_->out_strides_[i] = out_shape.at(i + 1) * param_->out_strides_[i + 1];
  }
  param_->thread_num_ = MSMIN(param_->op_parameter_.thread_num_, out_shape.at(0));
  max_c_ = reinterpret_cast<float *>(malloc(param_->input_c_ * sizeof(float)));
  if (max_c_ == nullptr) {
    MS_LOG(ERROR) << "malloc max_c failed.";
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

int ROIPoolingCPUKernel::DoExecute(int task_id) {
  MS_ASSERT(in_ptr_);
  MS_ASSERT(out_ptr_);
  MS_ASSERT(roi_ptr_);
  MS_ASSERT(max_c_);
  MS_ASSERT(param_);
  auto ret = ROIPooling(in_ptr_, out_ptr_, roi_ptr_, max_c_, task_id, param_);
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
  in_ptr_ = reinterpret_cast<float *>(in_tensors_.front()->MutableData());
  out_ptr_ = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
  roi_ptr_ = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto ret = ParallelLaunch(this->context_->thread_pool_, ROIPoolingRun, this, param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ROIPooling error: error_code[" << ret << "]";
    return ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ROIPooling, LiteKernelCreator<ROIPoolingCPUKernel>)
}  // namespace mindspore::kernel
