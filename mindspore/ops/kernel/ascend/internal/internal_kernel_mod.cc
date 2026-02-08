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

#include "kernel/ascend/internal/internal_kernel_mod.h"

#include <functional>
#include <utility>
#include "utils/ms_context.h"
#include "kernel/ascend/internal/internal_helper.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"
#include "kernel/ascend/internal/internal_tiling_cache.h"
#include "include/backend/common/ms_device_shape_transfer.h"

namespace mindspore {
namespace kernel {
SimpleSpinLock InternalKernelMod::lock_ = SimpleSpinLock();

inline bool IsContiguous(KernelTensor *tensor) {
  const auto &storage = tensor->tensor_storage_info();
  if (storage == nullptr) {
    return true;
  }

  return storage->is_contiguous;
}

inline int64_t GetStorageOffset(KernelTensor *tensor) {
  const auto &storage = tensor->tensor_storage_info();
  if (storage == nullptr) {
    return 0;
  }

  return static_cast<int64_t>(storage->storage_offset);
}

bool InternalKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto soc = ms_context->ascend_soc_version();
  if (soc.find("ascend910_93") != std::string::npos || soc.find("ascend910b") != std::string::npos) {
    is_aclgraph_supported_ = true;
  }

  internal_to_ms_input_indices_mapper_.clear();
  internal_to_ms_output_indices_mapper_.clear();
  bool input_mutable = false;
  auto in_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelInMap(kernel_name_, &input_mutable);
  if (input_mutable) {
    for (size_t i = 0; i < inputs.size(); i++) {
      (void)internal_to_ms_input_indices_mapper_.emplace_back(i);
    }
  } else {
    for (size_t i = 0; i < in_idx_list.size(); i++) {
      (void)internal_to_ms_input_indices_mapper_.emplace_back(static_cast<size_t>(in_idx_list.at(i)));
    }
  }

  bool output_mutable = false;
  auto out_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelOutMap(kernel_name_, &output_mutable);
  if (output_mutable) {
    for (size_t i = 0; i < outputs.size(); i++) {
      (void)internal_to_ms_output_indices_mapper_.emplace_back(i);
    }
  } else {
    for (size_t i = 0; i < out_idx_list.size(); i++) {
      (void)internal_to_ms_output_indices_mapper_.emplace_back(out_idx_list.at(i));
    }
  }

  for (size_t i = 0; i < internal_to_ms_input_indices_mapper_.size(); i++) {
    internal_inputs_addr_.emplace_back(nullptr);
    internal_inputs_shape_.emplace_back(internal::ShapeInfo{0});
  }

  for (size_t i = 0; i < internal_to_ms_output_indices_mapper_.size(); i++) {
    internal_outputs_addr_.emplace_back(nullptr);
    internal_outputs_shape_.emplace_back(internal::ShapeInfo{0});
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    bool is_include = false;
    for (auto idx : in_idx_list) {
      if (i == static_cast<size_t>(idx)) {
        is_include = true;
        break;
      }
    }
    if (!is_include) {
      recreate_cared_indices_.emplace_back(i);
    }
  }

  // find NZ format output to do extra resize
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->GetStringFormat() == kOpFormat_FRAC_NZ) {
      nz_output_indices_.emplace_back(i);
    }
  }

  return true;
}

bool InternalKernelMod::IsNeedRecreate(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  g_hash_offset = 0;
  for (auto idx : recreate_cared_indices_) {
    auto input = inputs[idx];
    auto type = input->type_id();
    if (type == kObjectTypeNumber) {
      auto data_type = input->dtype_id();
      switch (data_type) {
        case kNumberTypeBool: {
          auto value = input->GetValueWithCheck<bool>();
          GatherHash(value);
          break;
        }
        case kNumberTypeInt32: {
          auto value = input->GetValueWithCheck<int32_t>();
          GatherHash(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<int64_t>();
          GatherHash(value);
          break;
        }
        case kNumberTypeFloat32: {
          auto value = input->GetValueWithCheck<float>();
          GatherHash(value);
          break;
        }
        case kNumberTypeFloat64: {
          auto value = input->GetValueWithCheck<double>();
          GatherHash(value);
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
          GatherHash(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<std::vector<int64_t>>();
          GatherHash(value);
          break;
        }
        default:
          MS_LOG(INTERNAL_EXCEPTION) << "Unsupported dtype " << data_type << ", kenrel_name: " << kernel_name_
                                     << ", index: " << idx;
      }
    } else if (type == kMetaTypeNone) {
      GatherHash(type);
    } else if (type != kObjectTypeTensorType) {
      MS_LOG(INTERNAL_EXCEPTION) << "Unsupported type: " << type << ", kenrel_name: " << kernel_name_
                                 << ", index: " << idx;
    }
  }

  if (g_hash_offset == 0) {
    return internal_op_ == nullptr;
  }

  auto hash_id = calc_hash_id();
  if (hash_id != last_key_) {
    last_key_ = hash_id;
    return true;
  }
  return false;
}

uint64_t InternalKernelMod::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  return InternalTilingCache::GenerateKey(kernel_name_, inputs);
}

void InternalKernelMod::GetOrGenerateTiling(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  auto key = GenerateTilingKey(inputs);
  std::lock_guard<SimpleSpinLock> lock(lock_);
  auto tiling_cache_item = InternalTilingCache::GetInstance().Bind(key);
  InternalTilingCache::GetInstance().Unbind(last_item_);
  if (tiling_cache_item == nullptr) {
    auto tiling_size = internal_op_->GetTilingSize();
    auto host_addr = TilingMemMgr::GetInstance().pool_host_.Malloc(tiling_size);
    internal::HostRunInfoPtr host_run_info_ptr = nullptr;
    auto status = internal_op_->Tiling(host_addr, &host_run_info_ptr);
    if (status != internal::kInternalOk || host_run_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Tiling error for " << kernel_name_ << ", status: " << status
                        << ", host_run_info_ptr: " << host_run_info_ptr;
    }

    auto device_addr = TilingMemMgr::GetInstance().pool_device_.Malloc(tiling_size);
    TilingMemMgr::GetInstance().CopyAsync(host_addr, device_addr, tiling_size);
    auto tiling_info = std::make_shared<internal::TilingInfo>(device_addr, nullptr);
    internal_op_->SetTilingInfo(tiling_info);
    tiling_info->host_run_info_ = host_run_info_ptr;
    workspace_size_list_ = internal_op_->GetWorkspaceSize();
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list_);
    auto tiling_info_ptr = std::make_shared<TilingCacheItem>(tiling_info, host_addr, tiling_size);
    if (TilingMemMgr::GetInstance().pool_device_.IsOneOffMem(device_addr)) {
      // tiling mem pool is full, comb out some items which are not recently used with high probability
      auto erased_items = InternalTilingCache::GetInstance().CombOutSuspectedUselessItems();
      if (!erased_items.empty()) {
        for (auto &item : erased_items) {
          TilingMemMgr::GetInstance().pool_device_.Free(item->tiling_info_->tiling_addr_, item->size_);
          TilingMemMgr::GetInstance().pool_host_.Free(item->host_addr_, item->size_);
        }
        TilingMemMgr::GetInstance().pool_device_.Rearrange();
        TilingMemMgr::GetInstance().pool_host_.Rearrange();
      }
      MS_LOG(INFO) << "The tiling memory pool is full, comb out not used items: " << erased_items.size();
    }
    (void)InternalTilingCache::GetInstance().Insert(key, tiling_info_ptr);
    last_item_ = tiling_info_ptr;
  } else {
    internal_op_->SetTilingInfo(tiling_cache_item->tiling_info_);
    workspace_size_list_ = tiling_cache_item->tiling_info_->host_run_info_->GetWorkSpaceSize();
    last_item_ = tiling_cache_item;
  }
  internal_wss_addr_.resize(workspace_size_list_.size());
}

void InternalKernelMod::GetInternalKernel(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (IsNeedRecreate(inputs, outputs)) {
    internal::InputsImmutableInfoList inputs_ii;
    internal::OutputsImmutableInfoList outputs_ii;
    for (size_t i = 0; i < internal_to_ms_input_indices_mapper_.size(); i++) {
      auto ms_index = internal_to_ms_input_indices_mapper_[i];
      auto dtype = TransInternalDataType(inputs[ms_index]->dtype_id());
      auto format = TransInternalFormat(inputs[ms_index]->format());
      inputs_ii.emplace_back(dtype, format);
    }

    for (size_t i = 0; i < internal_to_ms_output_indices_mapper_.size(); i++) {
      auto ms_index = internal_to_ms_output_indices_mapper_[i];
      auto dtype = TransInternalDataType(outputs[ms_index]->dtype_id());
      auto format = TransInternalFormat(outputs[ms_index]->format());
      outputs_ii.emplace_back(dtype, format);
    }
    internal_op_ = CreateKernel(inputs_ii, outputs_ii, inputs, outputs);
    MS_EXCEPTION_IF_NULL(internal_op_);
    auto status = internal_op_->Init();
    if (status != internal::kInternalOk) {
      internal_op_ = nullptr;
      MS_LOG(ERROR) << "Init InternalKernel failed, kenrel_name: " << kernel_name_;
    }
  }
}

int InternalKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Kernel " << kernel_name_ << " Resize failed";
    return ret;
  }

  // update NZ format output output_size
  for (size_t i = 0; i < nz_output_indices_.size(); ++i) {
    auto index = nz_output_indices_[i];
    auto &output = outputs[index];
    MS_EXCEPTION_IF_NULL(output);
    auto shape = output->GetShapeVector();
    auto dev_shape = trans::TransShapeToDevice(shape, kOpFormat_FRAC_NZ, output->dtype_id());
    auto type_size = GetTypeByte(TypeIdToType(output->dtype_id()));
    auto tensor_size = dev_shape.empty()
                         ? std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>())
                         : std::accumulate(dev_shape.begin(), dev_shape.end(), type_size, std::multiplies<size_t>());
    output_size_list_[index] = tensor_size;
  }

  GetInternalKernel(inputs, outputs);
  if (internal_op_ == nullptr) {
    return KRET_RESIZE_FAILED;
  }

  for (size_t i = 0; i < internal_to_ms_input_indices_mapper_.size(); i++) {
    auto ms_index = internal_to_ms_input_indices_mapper_[i];
    auto shape = TransInternalShape(inputs[ms_index]->GetShapeVector());
    if (inputs[ms_index]->dtype_id() == kMetaTypeNone) {
      shape = {};
    }
    internal_inputs_shape_[i] = std::move(shape);
  }

  for (size_t i = 0; i < internal_to_ms_output_indices_mapper_.size(); i++) {
    auto ms_index = internal_to_ms_output_indices_mapper_[i];
    auto shape = TransInternalShape(outputs[ms_index]->GetShapeVector());
    if (outputs[ms_index]->dtype_id() == kMetaTypeNone) {
      shape = {};
    }
    internal_outputs_shape_[i] = std::move(shape);
  }
  if (!UpdateParam(inputs, outputs)) {
    MS_LOG(ERROR) << "UpdateParam failed, kernel_name: " << kernel_name_;
    return KRET_RESIZE_FAILED;
  }
  auto internal_ret = internal_op_->UpdateShape(internal_inputs_shape_, internal_outputs_shape_);
  if (internal_ret != internal::kInternalOk) {
    MS_LOG(ERROR) << "InternalKernel UpdateShape failed, kernel_name: " << kernel_name_;
    return KRET_RESIZE_FAILED;
  }

  GetOrGenerateTiling(inputs, outputs);
  return KRET_OK;
}

void InternalKernelMod::UpdateAddr(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs,
                                   const std::vector<KernelTensor *> &workspace) {
  for (size_t i = 0; i < internal_to_ms_input_indices_mapper_.size(); i++) {
    auto ms_index = internal_to_ms_input_indices_mapper_[i];
    if (!IsContiguous(inputs[ms_index])) {
      MS_LOG(EXCEPTION) << "Internal op " << kernel_name_
                        << " does not support noncontiguous input, index: " << ms_index;
    }

    internal_inputs_addr_[i] =
      static_cast<void *>(static_cast<int8_t *>(inputs[ms_index]->device_ptr()) + GetStorageOffset(inputs[ms_index]));
  }

  for (size_t i = 0; i < internal_to_ms_output_indices_mapper_.size(); i++) {
    auto ms_index = internal_to_ms_output_indices_mapper_[i];
    if (!IsContiguous(outputs[ms_index])) {
      MS_LOG(EXCEPTION) << "Internal op " << kernel_name_
                        << " does not support noncontiguous output, index: " << ms_index;
    }
    internal_outputs_addr_[i] =
      static_cast<void *>(static_cast<int8_t *>(outputs[ms_index]->device_ptr()) + GetStorageOffset(outputs[ms_index]));
  }

  for (size_t i = 0; i < workspace.size(); i++) {
    internal_wss_addr_[i] = workspace[i]->device_ptr();
  }
}

bool InternalKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_aclgraph_supported_) {
    auto acl_ret =
      CALL_ASCEND_API(aclmdlRICaptureGetInfo, reinterpret_cast<aclrtStream>(stream_ptr), &capture_status_, &ri_model_);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Op " << kernel_name_ << " call aclmdlRICaptureGetInfo failed, ret: " << acl_ret;
      return false;
    }

    if (capture_status_ == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
      InternalTilingCache::GetInstance().SetItemToPermanent(last_item_);
      MS_LOG(INFO) << "aclgraph is capturing model, set tiling item to permanent, op_name: " << kernel_name_
                   << ", item: " << last_item_ << ", tiling_addr: " << last_item_->tiling_info_->tiling_addr_
                   << ", inputs info: ";
      for (const auto input : inputs) {
        MS_LOG(INFO) << input->ToString();
      }
    }
  }

  UpdateAddr(inputs, outputs, workspace);
  internal::InternalStatus status =
    internal_op_->Launch(internal_inputs_addr_, internal_outputs_addr_, internal_wss_addr_, stream_ptr, fullname_);
  return (status == internal::InternalStatus::kInternalOk);
}
}  // namespace kernel
}  // namespace mindspore
