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

#include "src/runtime/kernel/arm/fp32/softmax.h"
#include <string.h>
#include <vector>
#include "src/runtime/kernel/arm/nnacl/fp32/softmax.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SoftMax;

namespace mindspore::kernel {
int SoftmaxCPUKernel::Init() {
  auto ret = SoftmaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SoftmaxCPUKernel::ReSize() {
  auto ret = SoftmaxBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  auto n_dim = softmax_param_->n_dim_;
  auto axis = softmax_param_->axis_;
  if (axis == -1) {
    softmax_param_->axis_ += n_dim;
    axis = softmax_param_->axis_;
  }
  auto in_shape = in_tensors_.front()->shape();
  int out_plane_size = 1;
  for (int i = 0; i < axis; ++i) {
    out_plane_size *= in_shape[i];
  }
  int in_plane_size = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    in_plane_size *= in_shape[i];
  }
  if (sum_data_ != nullptr) {
    free(sum_data_);
  }
  sum_data_ = reinterpret_cast<float *>(malloc(out_plane_size * in_plane_size * sizeof(float)));
  if (sum_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc data for softmax fail!";
    return RET_ERROR;
  }
  memset(sum_data_, 0, out_plane_size * in_plane_size * sizeof(float));
  return RET_OK;
}

int SoftmaxCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return RET_ERROR;
  }
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->Data());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  Softmax(input_ptr, output_ptr, sum_data_, softmax_param_);
  return RET_OK;
}

}  // namespace mindspore::kernel
