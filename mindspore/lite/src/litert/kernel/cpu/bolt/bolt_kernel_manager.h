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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_KERNEL_MANAGER_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_KERNEL_MANAGER_H_

#include <map>
#include <vector>
#include <utility>
#include "src/executor/kernel_exec.h"
#include "bolt/common/uni/include/parameter_spec.h"

namespace mindspore::kernel::bolt {
struct BoltKeyDesc {
  int op_;
  TypeId dt_;
  bool operator<(const BoltKeyDesc &comp) const { return (op_ != comp.op_) ? (op_ < comp.op_) : (dt_ < comp.dt_); }
};

typedef LiteKernel *(*BoltCreator)(const ParameterSpec &param_spec, const std::vector<lite::Tensor *> &in,
                                   const std::vector<lite::Tensor *> &out, const lite::InnerContext *ctx);

class BoltKernelRegistry {
 public:
  static BoltKernelRegistry *GetInstance() {
    static BoltKernelRegistry instance;
    return &instance;
  }
  void Register(BoltKeyDesc desc, Format df, BoltCreator creator) { bolt_map_[desc] = std::make_pair(df, creator); }

  BoltCreator Creator(BoltKeyDesc desc) {
    auto iter = bolt_map_.find(desc);
    if (iter != bolt_map_.end()) {
      return iter->second.second;
    }
    return nullptr;
  }

  Format GetKernelFormat(BoltKeyDesc desc) {
    auto iter = bolt_map_.find(desc);
    if (iter != bolt_map_.end()) {
      return iter->second.first;
    }
    return DEFAULT_FORMAT;
  }

 protected:
  std::map<BoltKeyDesc, std::pair<Format, BoltCreator>> bolt_map_;
};

class BoltKernelRegistrar {
 public:
  BoltKernelRegistrar(int op_type, TypeId data_type, Format format, BoltCreator creator) {
    BoltKernelRegistry::GetInstance()->Register({op_type, data_type}, format, creator);
  }
  ~BoltKernelRegistrar() = default;
};

template <class T>
LiteKernel *BoltOpt(const ParameterSpec &param_spec, const std::vector<lite::Tensor *> &in,
                    const std::vector<lite::Tensor *> &out, const lite::InnerContext *ctx) {
  auto *kernel = new (std::nothrow) T(param_spec, in, out, ctx);
  return kernel;
}

#define BLOT_REG_KERNEL(op_type, data_type, format, creator) \
  static BoltKernelRegistrar g_kernel##op_type##data_type##kernelReg(op_type, data_type, format, creator);

bool BoltSupportKernel(int op_type, TypeId data_type);

// registry for extendrt
LiteKernel *BoltKernelRegistry(const ParameterSpec &param_spec, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                               kernel::KernelKey *key);

// registry for litert
LiteKernel *BoltKernelRegistry(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                               kernel::KernelKey *key);
}  // namespace mindspore::kernel::bolt
#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_KERNEL_MANAGER_H_
