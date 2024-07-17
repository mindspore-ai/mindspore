/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/hal/device/ascend_memory_pool.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "acl/acl_rt.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace kernel {
InternalKernelMod::~InternalKernelMod() {
  for (auto t : inputs_) {
    delete t;
  }
  inputs_.clear();
  for (auto t : outputs_) {
    delete t;
  }
  outputs_.clear();
}

int InternalKernelMod::Build(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto param = CreateOpParam(inputs, outputs);
  param->op_fullname_ = fullname_;

  // abstract validation info from inputs
  internal::ValidateInfo info;
  info.input_num_ = inputsIdxMap_.size();
  info.output_num_ = outputsIdxMap_.size();
  param->in_dtypes_.resize(info.input_num_);
  param->out_dtypes_.resize(info.output_num_);

  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); iter++) {
    info.input_dtype_.emplace_back(InternalKernelUtils::ToInternalDType(inputs[iter->first]->dtype_id()));
    info.input_format_.emplace_back(InternalKernelUtils::ToInternalFormat(inputs[iter->first]->format()));
    param->in_dtypes_[iter->second] = InternalKernelUtils::ToInternalDType(inputs[iter->first]->dtype_id());
  }

  for (auto iter = outputsIdxMap_.begin(); iter != outputsIdxMap_.end(); iter++) {
    info.output_dtype_.emplace_back(InternalKernelUtils::ToInternalDType(outputs[iter->first]->dtype_id()));
    info.output_format_.emplace_back(InternalKernelUtils::ToInternalFormat(outputs[iter->first]->format()));
    param->out_dtypes_[iter->second] = InternalKernelUtils::ToInternalDType(outputs[iter->first]->dtype_id());
  }

  impl_ = internal::CreateInternalKernelImpl(param);
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Internal Op '" << kernel_name_ << "' create FAILED.";
    return KRET_RESIZE_FAILED;
  }

  if (!impl_->Init(info)) {
    MS_LOG(ERROR) << "Internal Op '" << kernel_name_ << "' is initialized FAILED.";
    return KRET_RESIZE_FAILED;
  }
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); iter++) {
    InternalKernelUtils::ToInternalTensor(inputs_[iter->second], inputs[iter->first]);
  }
  impl_->SetInputs(inputs_);

  return KRET_OK;
}

uint64_t InternalKernelMod::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(kernel_name_, primitive_, inputs);
}

void InternalKernelMod::SetTilingInfo(const uint64_t key) {
  size_t tiling_size = impl_->GetTilingBufSize();
  auto tiling_func = [this](internal::HostRawBuf &host_buf, internal::CacheInfo &cache_info) {
    auto ret = this->impl_->Tiling(host_buf);
    cache_info = this->impl_->GetCacheInfo();
    return ret;
  };
  if (tiling_info_.need_free_device_buf_) {
    free(tiling_info_.host_buf_.addr_);
    device::ascend::AscendMemoryPool::GetInstance().FreeTensorMem(tiling_info_.device_buf_.addr_);
  }
  tiling_info_ = TilingCacheMgr::GetInstance().GetOrCreateTilingInfo(key, tiling_func, tiling_size);
  impl_->SetCacheInfo(tiling_info_.cache_info_);
  impl_->SetDeviceTilingBuf(tiling_info_.device_buf_);
}

void InternalKernelMod::SetInOutIdx(size_t in_count, size_t out_count) {
  bool input_mutable = false;
  auto in_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelInMap(kernel_name_, &input_mutable);
  if (input_mutable) {
    for (size_t i = 0; i < in_count; i++) {
      inputsIdxMap_[i] = i;
    }
  } else {
    for (size_t i = 0; i < in_idx_list.size(); i++) {
      inputsIdxMap_[in_idx_list.at(i)] = i;
    }
  }

  bool output_mutable = false;
  auto out_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelOutMap(kernel_name_, &output_mutable);
  if (output_mutable) {
    for (size_t i = 0; i < out_count; i++) {
      outputsIdxMap_[i] = i;
    }
  } else {
    for (size_t i = 0; i < out_idx_list.size(); i++) {
      outputsIdxMap_[out_idx_list.at(i)] = i;
    }
  }
}

bool InternalKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  SetInOutIdx(inputs.size(), outputs.size());
  inputs_.resize(inputsIdxMap_.size());
  std::generate(inputs_.begin(), inputs_.end(), []() { return new internal::Tensor(); });
  outputs_.resize(outputsIdxMap_.size());
  std::generate(outputs_.begin(), outputs_.end(), []() { return new internal::Tensor(); });
  tiling_info_.device_buf_.size_ = 0;
  tiling_info_.device_buf_.addr_ = nullptr;
  return true;
}

int InternalKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto &llm_manager = LLMManager::GetInstance();
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " invoke resize failed";
    return KRET_RESIZE_FAILED;
  }
  if (impl_ == nullptr) {
    ret = Build(inputs, outputs);
    if (ret != 0) {
      MS_LOG(ERROR) << "op " << op_type_ << " build kernel failed";
      return KRET_RESIZE_FAILED;
    }
  }

  if (op_type_ == "PagedAttention" && llm_manager.enable_multi_level_seq_length_) {
    MS_LOG(INFO) << "Update multi_level_seq_length for Internal Op: " << op_type_;
    auto param = CreateOpParam(inputs, outputs);
    impl_->UpdateParam(param);
  }

  std::vector<internal::DIMS> input_shapes(inputs_.size());
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); iter++) {
    InternalKernelUtils::ToInternalTensor(inputs_[iter->second], inputs[iter->first]);
    input_shapes[iter->second] = inputs_[iter->second]->desc.dims;
  }
  impl_->SetInputs(inputs_);
  for (auto iter = outputsIdxMap_.begin(); iter != outputsIdxMap_.end(); iter++) {
    InternalKernelUtils::ToInternalTensor(outputs_[iter->second], outputs[iter->first]);
  }
  impl_->SetOutputs(outputs_);
  auto key = GenTilingCacheKey(inputs, outputs);
  SetTilingInfo(key);
  // update workspace_size list
  auto workspace_size_list = impl_->GetWorkSpaceSize();
  workspace_size_list_.resize(workspace_size_list.size());
  for (size_t i = 0; i < workspace_size_list.size(); ++i) {
    workspace_size_list_[i] = static_cast<size_t>(workspace_size_list[i]);
  }

  return KRET_OK;
}

bool InternalKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); ++iter) {
    inputs_[iter->second]->data = inputs[iter->first]->device_ptr();
  }
  impl_->SetInputs(inputs_);
  impl_->SetStream(stream_ptr);
  std::vector<internal::DeviceRawBuf> ws_raw_bufs(workspace.size());
  for (size_t i = 0; i < workspace.size(); ++i) {
    ws_raw_bufs[i] = InternalKernelUtils::ToDeviceRawBuf(workspace[i]);
  }
  impl_->SetWorkSpace(ws_raw_bufs);
  for (auto iter = outputsIdxMap_.begin(); iter != outputsIdxMap_.end(); ++iter) {
    InternalKernelUtils::ToInternalTensor(outputs_[iter->second], outputs[iter->first]);
  }
  impl_->SetOutputs(outputs_);
  int ret = 0;
  if (ascend_profiler_->GetEnableFlag()) {
    MS_LOG(INFO) << "The enable_profiler_flag is " << ascend_profiler_->GetEnableFlag();
    ret = impl_->LaunchWithProfiling();
  } else {
    ret = impl_->Launch();
  }
  return (ret == 0);
}
}  // namespace kernel
}  // namespace mindspore
