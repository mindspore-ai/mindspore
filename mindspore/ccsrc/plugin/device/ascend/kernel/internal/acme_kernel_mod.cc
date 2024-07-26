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

#include "plugin/device/ascend/kernel/internal/acme_kernel_mod.h"

#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "transform/acl_ir/op_api_cache.h"

namespace mindspore {
namespace kernel {
bool AcmeKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  acme_to_ms_input_indices_mapper_.clear();
  acme_to_ms_output_indices_mapper_.clear();

  bool input_mutable = false;
  auto in_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelInMap(kernel_name_, &input_mutable);
  if (input_mutable) {
    for (size_t i = 0; i < inputs.size(); i++) {
      (void)acme_to_ms_input_indices_mapper_.emplace_back(i);
    }
  } else {
    for (size_t i = 0; i < in_idx_list.size(); i++) {
      (void)acme_to_ms_input_indices_mapper_.emplace_back(static_cast<size_t>(in_idx_list.at(i)));
    }
  }

  bool output_mutable = false;
  auto out_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelOutMap(kernel_name_, &output_mutable);
  if (output_mutable) {
    for (size_t i = 0; i < outputs.size(); i++) {
      (void)acme_to_ms_output_indices_mapper_.emplace_back(i);
    }
  } else {
    for (size_t i = 0; i < out_idx_list.size(); i++) {
      (void)acme_to_ms_output_indices_mapper_.emplace_back(out_idx_list.at(i));
    }
  }

  for (size_t i = 0; i < acme_to_ms_input_indices_mapper_.size(); i++) {
    acme_inputs_addr_.emplace_back(nullptr);
    acme_inputs_shape_.emplace_back(acme::ShapeInfo{0});
  }

  for (size_t i = 0; i < acme_to_ms_output_indices_mapper_.size(); i++) {
    acme_outputs_addr_.emplace_back(nullptr);
    acme_outputs_shape_.emplace_back(acme::ShapeInfo{0});
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    for (auto idx : in_idx_list) {
      if (i == static_cast<size_t>(idx)) {
        continue;
      }

      recreate_cared_indices_.emplace_back(idx);
    }
  }

  return true;
}

bool AcmeKernelMod::IsNeedRecreate(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  transform::g_hash_offset = 0;
  for (auto idx : recreate_cared_indices_) {
    auto input = inputs[idx];
    auto type = input->type_id();
    if (type == kObjectTypeNumber) {
      auto data_type = input->dtype_id();
      switch (data_type) {
        case kNumberTypeBool: {
          auto value = input->GetValueWithCheck<bool>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeInt32: {
          auto value = input->GetValueWithCheck<int32_t>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<int64_t>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeFloat32: {
          auto value = input->GetValueWithCheck<float>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeFloat64: {
          auto value = input->GetValueWithCheck<double>();
          transform::GatherInfo(value);
          break;
        }
        default:
          MS_LOG(INTERNAL_EXCEPTION) << "Unsupported dtype " << data_type << ", kenrel_name: " << kernel_name_
                                     << ", index: " << idx;
      }
    } else if (type == kObjectTypeTuple || type == kObjectTypeList) {
      auto data_type = input->dtype_id();
      switch (data_type) {
        case kNumberTypeInt32: {
          auto value = input->GetValueWithCheck<std::vector<int32_t>>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<std::vector<int64_t>>();
          transform::GatherInfo(value);
          break;
        }
        default:
          MS_LOG(INTERNAL_EXCEPTION) << "Unsupported dtype " << data_type << ", kenrel_name: " << kernel_name_
                                     << ", index: " << idx;
      }
    } else if (type != kObjectTypeTensorType) {
      MS_LOG(INTERNAL_EXCEPTION) << "Unsupported type: " << type << ", kenrel_name: " << kernel_name_
                                 << ", index: " << idx;
    }
  }

  if (transform::g_hash_offset == 0) {
    return false;
  }

  auto hash_id = transform::calc_hash_id();
  if (hash_id != last_key_) {
    last_key_ = hash_id;
    return true;
  }
  return false;
}

void AcmeKernelMod::GetOrGenerateTiling(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto key = AcmeTilingCache::GenerateKey(kernel_name_, inputs);
  auto tiling_cache_item = AcmeTilingCache::GetInstance().Bind(key);
  AcmeTilingCache::GetInstance().Unbind(last_item_);
  if (tiling_cache_item == nullptr) {
    std::lock_guard<SimpleSpinLock> lock(lock_);
    auto tiling_size = acme_op_->GetTilingSize();
    auto host_addr = TilingMemMgr::GetInstance().pool_host_.Malloc(tiling_size);
    acme::HostRunInfoPtr host_run_info_ptr = nullptr;
    auto status = acme_op_->Tiling(host_addr, &host_run_info_ptr);
    if (status != acme::kAcmeOk || host_run_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Tiling error for " << kernel_name_ << ", status: " << status
                        << ", host_run_info_ptr: " << host_run_info_ptr;
    }

    auto device_addr = TilingMemMgr::GetInstance().pool_device_.Malloc(tiling_size);
    TilingMemMgr::GetInstance().CopyAsync(host_addr, device_addr, tiling_size);
    auto tiling_info = std::make_shared<acme::TilingInfo>(device_addr, nullptr);
    acme_op_->SetTilingInfo(tiling_info);
    tiling_info->host_run_info_ = host_run_info_ptr;
    workspace_size_list_ = acme_op_->GetWorkspaceSize();
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list_);
    auto tiling_info_ptr = std::make_shared<TilingCacheItem>(tiling_info, host_addr, tiling_size);
    auto ret = AcmeTilingCache::GetInstance().Insert(key, tiling_info_ptr);
    if (!ret) {
      // op cache is full, comb out some items which are not recently used with high probability
      auto erased_items = AcmeTilingCache::GetInstance().CombOutSuspectedUselessItems();
      for (auto &item : erased_items) {
        TilingMemMgr::GetInstance().pool_device_.Free(item->tiling_info_->tiling_addr_, item->size_);
        TilingMemMgr::GetInstance().pool_host_.Free(item->host_addr_, item->size_);
      }
      TilingMemMgr::GetInstance().pool_device_.Rearrange();
      TilingMemMgr::GetInstance().pool_host_.Rearrange();

      // try insert again, ignore the result of the insertion
      (void)AcmeTilingCache::GetInstance().Insert(key, tiling_info_ptr);
    }
    last_item_ = tiling_info_ptr;
  } else {
    acme_op_->SetTilingInfo(tiling_cache_item->tiling_info_);
    workspace_size_list_ = tiling_cache_item->tiling_info_->host_run_info_->GetWorkspaceSize();
    last_item_ = tiling_cache_item;
  }
  acme_wss_addr_.resize(workspace_size_list_.size());
}

void AcmeKernelMod::GetAcmeKernel(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (acme_op_ == nullptr || IsNeedRecreate(inputs, outputs)) {
    acme::InputsImmutableInfoList inputs_ii;
    acme::OutputsImmutableInfoList outputs_ii;
    for (size_t i = 0; i < acme_to_ms_input_indices_mapper_.size(); i++) {
      auto ms_index = acme_to_ms_input_indices_mapper_[i];
      auto dtype = TransAcmeDataType(inputs[ms_index]->dtype_id());
      auto format = TransAcmeFormat(inputs[ms_index]->format());
      inputs_ii.emplace_back(dtype, format);
    }

    for (size_t i = 0; i < acme_to_ms_output_indices_mapper_.size(); i++) {
      auto ms_index = acme_to_ms_output_indices_mapper_[i];
      auto dtype = TransAcmeDataType(outputs[ms_index]->dtype_id());
      auto format = TransAcmeFormat(outputs[ms_index]->format());
      outputs_ii.emplace_back(dtype, format);
    }
    acme_op_ = CreateKernel(inputs_ii, outputs_ii, inputs, outputs);
    MS_EXCEPTION_IF_NULL(acme_op_);
    auto status = acme_op_->Init();
    if (status != acme::kAcmeOk) {
      acme_op_ = nullptr;
      MS_LOG(ERROR) << "Init AcmeKernel failed, kenrel_name: " << kernel_name_;
    }
  }
}

int AcmeKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Kernel " << kernel_name_ << " Resize failed";
    return ret;
  }

  GetAcmeKernel(inputs, outputs);
  if (acme_op_ == nullptr) {
    return KRET_RESIZE_FAILED;
  }

  acme::ShapeInfoList acme_inputs_shape;
  acme::ShapeInfoList acme_outputs_shape;
  for (size_t i = 0; i < acme_to_ms_input_indices_mapper_.size(); i++) {
    auto ms_index = acme_to_ms_input_indices_mapper_[i];
    auto shape = TransAcmeShape(inputs[ms_index]->GetShapeVector());
    acme_inputs_shape.emplace_back(shape);
  }

  for (size_t i = 0; i < acme_to_ms_output_indices_mapper_.size(); i++) {
    auto ms_index = acme_to_ms_output_indices_mapper_[i];
    auto shape = TransAcmeShape(outputs[ms_index]->GetShapeVector());
    acme_outputs_shape.emplace_back(shape);
  }

  auto acme_ret = acme_op_->UpdateShape(acme_inputs_shape, acme_outputs_shape);
  if (acme_ret != acme::kAcmeOk) {
    MS_LOG(ERROR) << "AcmeKernel UpdateShape failed, kernel_name: " << kernel_name_;
    return KRET_RESIZE_FAILED;
  }

  GetOrGenerateTiling(inputs, outputs);
  return KRET_OK;
}

void AcmeKernelMod::UpdateAddr(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                               const std::vector<KernelTensor *> &workspace) {
  for (size_t i = 0; i < acme_to_ms_input_indices_mapper_.size(); i++) {
    auto ms_index = acme_to_ms_input_indices_mapper_[i];
    acme_inputs_addr_[i] = inputs[ms_index]->device_ptr();
  }

  for (size_t i = 0; i < acme_to_ms_output_indices_mapper_.size(); i++) {
    auto ms_index = acme_to_ms_output_indices_mapper_[i];
    acme_outputs_addr_[i] = outputs[ms_index]->device_ptr();
  }

  for (size_t i = 0; i < workspace.size(); i++) {
    acme_wss_addr_[i] = workspace[i]->device_ptr();
  }
}

bool AcmeKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  UpdateAddr(inputs, outputs, workspace);
  acme::AcmeStatus status;
  if (ascend_profiler_->GetEnableFlag()) {
    status =
      acme_op_->LaunchWithProfiling(acme_inputs_addr_, acme_outputs_addr_, acme_wss_addr_, stream_ptr, fullname_);
  } else {
    status = acme_op_->Launch(acme_inputs_addr_, acme_outputs_addr_, acme_wss_addr_, stream_ptr);
  }
  return (status == acme::AcmeStatus::kAcmeOk);
}
}  // namespace kernel
}  // namespace mindspore
