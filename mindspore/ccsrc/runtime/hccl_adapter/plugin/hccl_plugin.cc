/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "runtime/hccl_adapter/plugin/hccl_plugin.h"
#define google ascend_private
#include "register/ops_kernel_builder_registry.h"
#include "common/opskernel/ops_kernel_info_store.h"
#undef google
#include "hccl/hcom.h"

extern "C" {
ge::Status Initialize(const std::map<std::string, std::string> &);
ge::Status Finalize();
void GetOpsKernelInfoStores(std::map<std::string, std::shared_ptr<ge::OpsKernelInfoStore>> &);

ge::Status PluginInitHcomGraphAdapter(const std::map<std::string, std::string> &options) {
  return ::Initialize(options);
}

ge::Status PluginFinalizeHcomGraphAdapter() { return ::Finalize(); }

void PluginGetHcclKernelInfoStore(std::shared_ptr<ge::OpsKernelInfoStore> *hccl_kernel_info_store) {
  if (hccl_kernel_info_store == nullptr) {
    return;
  }

  std::map<std::string, std::shared_ptr<ge::OpsKernelInfoStore>> all_ops_kernel_info_stores;
  ::GetOpsKernelInfoStores(all_ops_kernel_info_stores);
  for (auto &[name, ptr] : all_ops_kernel_info_stores) {
    if (name == kHcclOpsKernelInfoStore) {
      *hccl_kernel_info_store = ptr;
      return;
    }
  }

  *hccl_kernel_info_store = nullptr;
}

void PluginGetAllKernelBuilder(std::map<std::string, ge::OpsKernelBuilderPtr> *all_ops_kernel_builder) {
  if (all_ops_kernel_builder == nullptr) {
    return;
  }

  *all_ops_kernel_builder = ge::OpsKernelBuilderRegistry::GetInstance().GetAll();
}
};  // extern C
