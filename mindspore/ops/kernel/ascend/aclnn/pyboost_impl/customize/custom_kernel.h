/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_KERNEL_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <list>
#include <unordered_map>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class CustomAclnnPyboostKernelModBase {
 public:
  explicit CustomAclnnPyboostKernelModBase(std::string &&op_type) : op_type_(std::move(op_type)) {
    int64_t capaticy_from_user = device::ascend::GetCacheCapaticy();
    if (capaticy_from_user >= 0) {
      capacity_ = LongToSize(capaticy_from_user);
    }
  }
  ~CustomAclnnPyboostKernelModBase() = default;
  virtual bool Launch(const std::vector<ValuePtr> &inputs, const std::vector<tensor::TensorPtr> &outputs,
                      const std::shared_ptr<pyboost::OpRunner> &op) = 0;
  std::string op_type_;
  std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;
  std::list<CacheTuple> hash_cache_;
  size_t capacity_{1024};
  std::mutex mutex_;
};

template <size_t N>
class CustomAclnnPyboostKernelMod : public CustomAclnnPyboostKernelModBase {
 public:
  explicit CustomAclnnPyboostKernelMod(std::string op_type) : CustomAclnnPyboostKernelModBase(std::move(op_type)) {}
  ~CustomAclnnPyboostKernelMod() = default;
  bool Launch(const std::vector<ValuePtr> &inputs, const std::vector<tensor::TensorPtr> &outputs,
              const std::shared_ptr<pyboost::OpRunner> &op) override {
    const auto &res_tuple = GetKernelTuple<N>(inputs, outputs);
    std::apply([this, op](const auto &...args) { CallRun(op, args...); }, res_tuple);
    return true;
  }

 private:
  template <typename... Args>
  auto GetExecutor(const std::shared_ptr<pyboost::OpRunner> &op, const Args &...args) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (capacity_ == 0) {
      auto [ws_size, executor, cache, release_func] = GEN_CUSTOM_EXECUTOR(op_type_, args...);
      std::function<void()> update_func = nullptr;
      return std::make_tuple(ws_size, executor, cache, release_func, update_func);
    }
    uint64_t hash_id = mindspore::device::ascend::AclnnHash(op_type_, args...);
    auto iter = hash_map_.find(hash_id);
    if (hash_id != 0 && iter != hash_map_.end()) {
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, iter->second);
      auto cur_run = hash_cache_.front();
      const auto &ws_size = std::get<kIndex3>(cur_run);
      const auto &executor = std::get<kIndex1>(cur_run);
      const auto &cache = std::get<kIndex2>(cur_run);
      auto address_list = mindspore::device::ascend::GetTensorAddress(args...);
      std::function<void()> update_func = [cache, address_list]() -> void {
        cache(device::ascend::ProcessCacheType::kUpdateTensorAddress, address_list);
      };
      auto release_func = std::function<void()>(nullptr);
      return std::make_tuple(ws_size, executor, cache, release_func, update_func);
    } else {
      MS_LOG(INFO) << "Api " << op_type_ << " miss cache, with hash id:" << hash_id;
      auto [ws_size, executor, cache, fail_cache] = GEN_CUSTOM_EXECUTOR_FOR_RESIZE(op_type_, args...);
      auto update_func = std::function<void()>(nullptr);
      if (hash_id != 0 && !fail_cache) {
        hash_cache_.emplace_front(hash_id, executor, cache, ws_size);
        hash_map_[hash_id] = hash_cache_.begin();
        if (hash_cache_.size() > capacity_) {
          hash_map_.erase(std::get<kIndex0>(hash_cache_.back()));
          auto release_func = std::get<kIndex2>(hash_cache_.back());
          runtime::Pipeline::Get().launch_stage()->Wait();
          release_func(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
          hash_cache_.pop_back();
        }
        auto release_func = std::function<void()>(nullptr);
        return std::make_tuple(ws_size, executor, cache, release_func, update_func);
      } else {
        std::function<void()> release_func = [cache]() -> void {
          cache(device::ascend::ProcessCacheType::kReleaseParams, std::vector<std::vector<void *>>{});
        };
        return std::make_tuple(ws_size, executor, cache, release_func, update_func);
      }
    }
  }

  template <typename... Args>
  void CallRun(const std::shared_ptr<pyboost::OpRunner> &op, const Args &...args) {
    MS_EXCEPTION_IF_NULL(op);
    static auto simu = common::IsCompileSimulation();
    if (simu) {
      return;
    }

    auto stream_id = op->stream_id();
    auto device_context = op->device_context();
    auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
    auto aclnn_name = op_type_;

    auto return_values = GetExecutor(op, args...);
    auto ws_size = std::get<kIndex0>(return_values);
    auto executor_handle = std::get<kIndex1>(return_values);
    auto release_function = std::get<kIndex3>(return_values);
    auto update_func = std::get<kIndex4>(return_values);
    if (ws_size == 0) {
      DISPATCH_LAUNCH_CUSTOM_KERNEL(device_context, aclnn_name, nullptr, 0, executor_handle, stream_ptr,
                                    release_function, update_func);
    } else {
      auto work_ptr = std::make_shared<kernel::pyboost::MemBlock>(device_context, ws_size, stream_id);
      DISPATCH_LAUNCH_CUSTOM_KERNEL(device_context, aclnn_name, work_ptr->ptr_, ws_size, executor_handle, stream_ptr,
                                    release_function, update_func);
    }
  }
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_KERNEL_H_
