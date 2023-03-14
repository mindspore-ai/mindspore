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
#include <memory>
#include "base/base.h"
#include "c_api/base/handle_types.h"
#include "c_api/src/common.h"
#include "pipeline/jit/resource.h"
#include "utils/ms_context.h"
#include "backend/graph_compiler/backend_base.h"

class ResourceManager {
 public:
  ResourceManager() {
    context_ = mindspore::MsContext::GetInstance();
    context_->set_backend_policy("ms");
    context_->set_param<int>(mindspore::MS_CTX_EXECUTION_MODE, mindspore::kGraphMode);
  }

  void SetBackend(std::shared_ptr<mindspore::compile::Backend> backend) { backend_ = std::move(backend); }

  std::shared_ptr<mindspore::compile::Backend> GetBackend() { return backend_; }

  void SetResult(const std::string &key, const mindspore::Any &value) { results_[key] = value; }

  mindspore::Any GetResult(const std::string &key) const {
    auto iter = results_.find(key);
    if (iter == results_.end()) {
      MS_LOG(EXCEPTION) << "this key is not in resource list:" << key;
    }
    return iter->second;
  }

  void SetInfer(bool infer) { auto_infer_ = infer; }

  bool GetInfer() { return auto_infer_; }

  void StoreSrcPtr(const BasePtr &src_ptr) {
    (void)ptr_res_pool_.insert(std::make_pair(reinterpret_cast<Handle>(src_ptr.get()), src_ptr));
  }

  BasePtr GetSrcPtr(ConstHandle ptr) {
    auto iter = ptr_res_pool_.find(ptr);
    if (iter != ptr_res_pool_.end()) {
      return iter->second;
    } else {
      MS_LOG(ERROR) << "The key handle " << ptr << " is not exist in resource pool.";
      return nullptr;
    }
  }

  void ReleaseSrcPtr(ConstHandle ptr) {
    auto iter = ptr_res_pool_.find(ptr);
    if (iter != ptr_res_pool_.end()) {
      (void)ptr_res_pool_.erase(iter);
    }
  }

 private:
  std::unordered_map<ConstHandle, BasePtr> ptr_res_pool_;
  mindspore::HashMap<std::string, mindspore::Any> results_{};
  std::shared_ptr<mindspore::compile::Backend> backend_ = nullptr;
  std::shared_ptr<mindspore::MsContext> context_ = nullptr;
  bool auto_infer_ = true;
};

#endif  // MINDSPORE_CCSRC_C_API_SRC_RESOURCE_MANAGER_H_
