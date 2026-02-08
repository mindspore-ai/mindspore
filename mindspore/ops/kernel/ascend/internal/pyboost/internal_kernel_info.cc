/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/internal/pyboost/internal_kernel_info.h"
#include <memory>
#include <functional>
#include <utility>
#include "kernel/ascend/acl_ir/op_api_cache.h"

namespace mindspore {
namespace kernel {
void InternalKernelInfo::TransInternalShapes(internal::ShapeInfoList *shapelist, const TensorPtrList &tensorlist,
                                             bool is_input) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    if (tensorlist[i] == nullptr) {
      shapelist->at(i) = internal::ShapeInfo{0};
      continue;
    }

    if (!tensorlist[i]->is_contiguous()) {
      if (is_input) {
        MS_LOG(EXCEPTION) << "For internal op [" << kernel_name_ << "], the input tensorlist[" << i
                          << "] is not contiguous: " << tensorlist[i]->ToString()
                          << ", please convert it to contiguous tensor using tensor.contiguous().";
      } else {
        MS_LOG(EXCEPTION) << "For internal op [" << kernel_name_ << "], the output tensorlist[" << i
                          << "] is not contiguous: " << tensorlist[i]->ToString()
                          << ", please convert it to contiguous tensor using tensor.contiguous().";
      }
    }

    auto shape =
      tensorlist[i]->data_type() != kMetaTypeNone ? TransInternalShape(tensorlist[i]->shape()) : internal::ShapeInfo{0};
    shapelist->at(i) = std::move(shape);
  }
}

void InternalKernelInfo::TransInternalShapes(const TensorPtrList &inputs, const TensorPtrList &outputs) {
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs, true);
  TransInternalShapes(&internal_outputs_shape_, outputs);
}

void InternalKernelInfo::UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo, const TensorPtr &tensor,
                                                internal::DataType dtype) {
  arginfo->SetDtype(dtype);
  if (tensor == nullptr) {
    arginfo->SetFormat(internal::TensorFormat::kFormatND);
    return;
  }
  arginfo->SetFormat(TransInternalFormat(tensor->format()));
}

void InternalKernelInfo::UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> *arginfos,
                                                const TensorPtrList &tensorlist, bool is_input) {
  arginfos->resize(tensorlist.size());
  for (size_t i = 0; i < tensorlist.size(); ++i) {
    if (is_input) {
      UpdateArgImmutableInfo(&(arginfos->at(i)), tensorlist[i], internal_inputs_dtype_[i]);
    } else {
      UpdateArgImmutableInfo(&(arginfos->at(i)), tensorlist[i], internal_outputs_dtype_[i]);
    }
  }
}

bool InternalKernelInfo::IsInternalDtypeSupport(const TensorPtrList *ms_inputs, const TensorPtrList *ms_outputs) {
  internal_inputs_dtype_.resize(ms_inputs->size());
  internal_outputs_dtype_.resize(ms_outputs->size());

  for (size_t i = 0; i < ms_inputs->size(); ++i) {
    if (ms_inputs->at(i) == nullptr) {
      internal_inputs_dtype_[i] = internal::DataType::kTypeNone;
      continue;
    }

    internal_inputs_dtype_[i] = TransInternalDataType(ms_inputs->at(i)->data_type());
  }

  for (size_t i = 0; i < ms_outputs->size(); ++i) {
    if (ms_outputs->at(i) == nullptr) {
      internal_outputs_dtype_[i] = internal::DataType::kTypeNone;
      continue;
    }
    internal_outputs_dtype_[i] = TransInternalDataType(ms_outputs->at(i)->data_type());
  }

  return internal::IsInternalKernelDtypesSupported(TransInternalOpName(kernel_name_), internal_inputs_dtype_,
                                                   internal_outputs_dtype_);
}

void InternalKernelInfo::GetOrCreateKernel(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                                           const uint64_t &tiling_key, const TensorPtrList &inputs,
                                           const TensorPtrList &outputs) {
  auto key = GetOrGenerateOpKey(op_key);
  auto it = hash_map_.find(key);
  if (it != hash_map_.end()) {
    internal_op_ = it->second;
    MS_LOG(DEBUG) << "Internal Op [" << kernel_name_ << "] hit cache";
  } else {
    MS_LOG(DEBUG) << "Internal Op [" << kernel_name_ << "] miss cache";
    if (!IsInternalDtypeSupport(&inputs, &outputs)) {
      MS_LOG(EXCEPTION) << "Input dtype is not supported for internal op [" << kernel_name_ << "]";
    }
    UpdateArgImmutableInfo(&inputs_ii_, inputs, true);
    UpdateArgImmutableInfo(&outputs_ii_, outputs);
    internal_op_ = CreateKernel(inputs_ii_, outputs_ii_);
    MS_EXCEPTION_IF_NULL(internal_op_);
    auto status = internal_op_->Init();
    if (status != internal::kInternalOk) {
      internal_op_ = nullptr;
      MS_LOG(EXCEPTION) << "Init internal kernel failed, kenrel_name: " << kernel_name_;
      return;
    }
    hash_map_[key] = internal_op_;
  }

  if (!UpdateParam()) {
    MS_LOG(EXCEPTION) << "UpdateParam failed, kenrel_name: " << kernel_name_;
  }
  auto internal_ret = internal_op_->UpdateShape(internal_inputs_shape_, internal_outputs_shape_);
  if (internal_ret != internal::kInternalOk) {
    MS_LOG(EXCEPTION) << "InternalKernel UpdateShape failed, kernel_name: " << kernel_name_;
  }

  tiling_info_ = GetOrGenerateTiling(op, tiling_key);
  if (tiling_info_ == nullptr) {
    MS_LOG(EXCEPTION) << "Create tiling info failed for internal kernel, kernel_name: " << kernel_name_;
  }
}

TilingCacheItemPtr InternalKernelInfo::GetOrGenerateTiling(const std::shared_ptr<pyboost::OpRunner> &op,
                                                           const uint64_t &tiling_key) {
  std::lock_guard<SimpleSpinLock> lock(lock_);
  auto key = GetOrGenerateOpTilingKey(tiling_key);
  auto tiling_info_ptr = InternalTilingCache::GetInstance().Bind(key);
  if (tiling_info_ptr == nullptr) {
    auto device_ctx = op->device_context();
    device_ctx->device_res_manager_->BindDeviceToCurrentThread(false);
    MS_LOG(INFO) << "start create tiling info for " << kernel_name_;
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
    tiling_info->host_run_info_ = host_run_info_ptr;
    auto workspace_size_list = internal_op_->GetWorkspaceSize();
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list);
    tiling_info_ptr = std::make_shared<TilingCacheItem>(tiling_info, host_addr, tiling_size);
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
    MS_LOG(INFO) << "end create tiling info for " << kernel_name_;
  }
  return tiling_info_ptr;
}

}  // namespace kernel
}  // namespace mindspore
