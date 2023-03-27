/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "bolt/bolt_kernel.h"
#include "bolt/bolt_tensor_utils.h"

// get from ctx->num_threads
int OMP_NUM_THREADS = 1;
namespace mindspore::kernel::bolt {
int BoltKernel::Prepare() {
  auto ret = InitArch();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init bolt arch info failed.";
    return ret;
  }
  // Create bolt tensor
  ret = UpdateTensors();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update lite tensors to bolt tensors failed.";
    return ret;
  }
  dt_ = bolt_in_tensors_[0].get_desc().dt;
  return RET_OK;
}

int BoltKernel::Run() {
  // sync tensor info: shape(resize), data(output tensor malloc), etc.
  if (in_tensors_.size() != bolt_in_tensors_.size()) {
    MS_LOG(ERROR) << "The input lite tensors num: " << in_tensors_.size()
                  << " is not equal to input bolt tensors num: " << bolt_in_tensors_.size();
    return RET_ERROR;
  }
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto ret = LiteTensor2BoltTensor(in_tensors_[i], &(bolt_in_tensors_[i]));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sync info for input tensor " << in_tensors_[i]->tensor_name() << " failed.";
      return ret;
    }
  }
  if (out_tensors_.size() != bolt_out_tensors_.size()) {
    MS_LOG(ERROR) << "The output lite tensors num: " << out_tensors_.size()
                  << " is not equal to output bolt tensors num: " << bolt_out_tensors_.size();
    return RET_ERROR;
  }
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    auto ret = LiteTensor2BoltTensor(out_tensors_[i], &(bolt_out_tensors_[i]));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sync info for output tensor " << out_tensors_[i]->tensor_name() << " failed.";
      return ret;
    }
  }
  return RET_OK;
}

int BoltKernel::InitArch() {
#ifdef ENABLE_AVX
  arch_info_.arch = X86_AVX2;
#endif
#ifdef ENABLE_AVX512
  arch_info_.arch = X86_AVX512;
#endif
#ifdef ENABLE_ARM64
  arch_info_.arch = ARM_V8;
#endif
#ifdef ENABLE_ARM32
  arch_info_.arch = ARM_V7;
#endif
  if (arch_info_.arch) {
    return RET_OK;
  }
  MS_LOG(ERROR) << "Unsupported backend for bolt kernel.";
  return RET_NOT_SUPPORT;
}

int BoltKernel::UpdateTensors() {
  // convert lite::tensor to bolt Tensor
  bolt_in_tensors_.clear();
  for (const auto &in_tensor : in_tensors_) {
    auto bolt_tensor = BoltTensor(CPUMem);
    auto ret = LiteTensor2BoltTensor(in_tensor, &bolt_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sync lite tensor info to bolt tensor failed.";
      return ret;
    }
    bolt_in_tensors_.push_back(bolt_tensor);
  }
  bolt_out_tensors_.clear();
  for (const auto &out_tensor : out_tensors_) {
    auto bolt_tensor = BoltTensor(CPUMem);
    auto ret = LiteTensor2BoltTensor(out_tensor, &bolt_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sync lite tensor info to bolt tensor failed.";
      return ret;
    }
    bolt_out_tensors_.push_back(bolt_tensor);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel::bolt
