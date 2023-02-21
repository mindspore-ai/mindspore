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

#ifndef LITE_OFFLINE_PACKING_OPTIMIZER_H
#define LITE_OFFLINE_PACKING_OPTIMIZER_H
#include <string>
#include <map>
#include "base/base.h"
#include "ir/anf.h"
#include "ops/core_ops.h"
#include "litert/lite_kernel.h"
#include "litert/kernel_registry.h"

namespace mindspore::lite {
using OfflinePackingFunc = STATUS (*)(const mindspore::CNodePtr &cnode_ptr, const FuncGraphPtr &funcGraphPtr,
                                      const lite::InnerContext *ctx);
using InnerContextCreatorFunc = mindspore::lite::InnerContext *(*)();

STATUS MatmulPacking(const mindspore::CNodePtr &cnode_ptr, const FuncGraphPtr &funcGraphPtr,
                     const lite::InnerContext *ctx);
mindspore::lite::InnerContext *InitInnerContextForAndroidArmCpu();

enum class BackendType : uint8_t {
  kUnknownBackend = 0,
  kAndroidArmCpuBackend,
};

class PackDataWrapper {
 public:
  static PackDataWrapper &GetInstance() {
    static PackDataWrapper instance;
    return instance;
  }

  const kernel::LiteKernel *GetPackedKernel(const std::string &node_name) {
    if (this->pack_mapping_.find(node_name) == this->pack_mapping_.end()) {
      return nullptr;
    }
    return this->pack_mapping_[node_name];
  }

  void AddPackedKernel(const std::string &node_name, const kernel::LiteKernel *data) {
    if (this->pack_mapping_.find(node_name) != this->pack_mapping_.end()) {
      MS_LOG(WARNING) << "Key conflict when add packed kernel.";
    }
    this->pack_mapping_[node_name] = data;
  }

 private:
  PackDataWrapper() = default;
  ~PackDataWrapper() = default;

 private:
  std::map<std::string, const kernel::LiteKernel *> pack_mapping_;
};

class OfflinePackingOptimizer {
 public:
  OfflinePackingOptimizer() {
    this->packing_strategies_selector_[BackendType::kAndroidArmCpuBackend] =
      std::map<schema::PrimitiveType, OfflinePackingFunc>{
        {schema::PrimitiveType::PrimitiveType_MatMulFusion, MatmulPacking},
      };
    this->ctx_creator_selector_[BackendType::kAndroidArmCpuBackend] = InitInnerContextForAndroidArmCpu;
  }

  STATUS Optimize(const FuncGraphPtr &func_graph, const std::string &target_backend);

 private:
  std::map<BackendType, std::map<schema::PrimitiveType, OfflinePackingFunc>> packing_strategies_selector_;
  std::map<BackendType, InnerContextCreatorFunc> ctx_creator_selector_;
};
};      // namespace mindspore::lite
#endif  // LITE_OFFLINE_PACKING_OPTIMIZER_H
