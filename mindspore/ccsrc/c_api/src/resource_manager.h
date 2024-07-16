/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_C_API_SRC_RESOURCE_MANAGER_H_
#define MINDSPORE_CCSRC_C_API_SRC_RESOURCE_MANAGER_H_

#include <utility>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include "base/base.h"
#include "include/c_api/ms/base/handle_types.h"
#include "c_api/src/common.h"
#include "pipeline/jit/ps/resource.h"
#include "utils/ms_context.h"
#include "backend/graph_compiler/backend_base.h"
#include "backend/graph_compiler/op_backend.h"
#include "c_api/src/dynamic_op_info.h"

static const size_t maxOpPoolSize = 500;
class ResourceManager {
 public:
  ResourceManager() {
    context_ = mindspore::MsContext::GetInstance();
    (void)context_->set_backend_policy("ms");
    context_->set_param<int>(mindspore::MS_CTX_EXECUTION_MODE, mindspore::kGraphMode);
  }

  ~ResourceManager() {
    for (auto iter : backends_) {
      const auto &backend = iter.second;
      if (backend != nullptr) {
        backend->ClearOpExecutorResource();
      }
    }
    backends_.clear();
    ptr_res_pool_.clear();
    dynamic_op_pool_.clear();
    results_.clear();
  }

  void SetResult(const std::string &key, const mindspore::Any &value) { results_[key] = value; }

  mindspore::Any GetResult(const std::string &key) const {
    auto iter = results_.find(key);
    if (iter == results_.end()) {
      MS_LOG(EXCEPTION) << "this key is not in resource list:" << key;
    }
    return iter->second;
  }

  void CacheBackend(const std::string &device_target, const MindRTBackendPtr &backend) {
    backends_[device_target] = backend;
  }

  MindRTBackendPtr GetBackendFromCache(const std::string &device_target) {
    auto iter = backends_.find(device_target);
    if (iter == backends_.end()) {
      MS_LOG(INFO) << "Current backend has not been cached in backends pool.";
      return nullptr;
    }
    return iter->second;
  }

  const OpBackendPtr &GetOpBackend() {
    if (op_backend_ == nullptr) {
      op_backend_ = std::make_unique<mindspore::compile::OpBackend>();
    }
    return op_backend_;
  }

  void CacheOpRunInfo(std::shared_ptr<InnerOpInfo> inner_info, FrontendOpRunInfoPtr run_info) {
    if (dynamic_op_pool_.size() > maxOpPoolSize) {
      dynamic_op_pool_.erase(dynamic_op_pool_.begin());
    }
    dynamic_op_pool_[*inner_info] = run_info;
  }

  FrontendOpRunInfoPtr GetOpRunInfoFromCache(std::shared_ptr<InnerOpInfo> inner_info) {
    auto iter = dynamic_op_pool_.find(*inner_info);
    if (iter == dynamic_op_pool_.end()) {
      MS_LOG(INFO) << "The OpInfo has not been cached in dynamic operator pool.";
      return nullptr;
    }
    return iter->second;
  }

  size_t GetCachedOpNum() const { return dynamic_op_pool_.size(); }

  void SetInfer(bool infer) { auto_infer_ = infer; }

  bool GetInfer() const { return auto_infer_; }

  void StoreSrcPtr(const BasePtr &src_ptr) {
    (void)ptr_res_pool_.insert(std::make_pair(reinterpret_cast<Handle>(src_ptr.get()), src_ptr));
  }

  BasePtr GetSrcPtr(ConstHandle ptr) {
    auto iter = ptr_res_pool_.find(ptr);
    if (iter == ptr_res_pool_.end()) {
      MS_LOG(ERROR) << "The key handle " << ptr << " is not exist in resource pool.";
      return nullptr;
    }
    return iter->second;
  }

  void ReleaseSrcPtr(ConstHandle ptr) {
    auto iter = ptr_res_pool_.find(ptr);
    if (iter != ptr_res_pool_.end()) {
      (void)ptr_res_pool_.erase(iter);
    }
  }

 private:
  std::unordered_map<ConstHandle, BasePtr> ptr_res_pool_{};
  std::unordered_map<InnerOpInfo, FrontendOpRunInfoPtr> dynamic_op_pool_{};
  std::unordered_map<std::string, MindRTBackendPtr> backends_{};
  OpBackendPtr op_backend_;
  mindspore::HashMap<std::string, mindspore::Any> results_{};
  std::shared_ptr<mindspore::MsContext> context_ = nullptr;
  bool auto_infer_ = true;
};
#endif  // MINDSPORE_CCSRC_C_API_SRC_RESOURCE_MANAGER_H_
