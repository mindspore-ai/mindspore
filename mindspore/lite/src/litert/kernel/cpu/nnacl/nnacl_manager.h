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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_MANAGER_H_

#include <map>
#include <vector>
#include "nnacl/nnacl_kernel.h"

namespace mindspore::nnacl {
struct KeyDesc {
  int op_;
  TypeId dt_;
  bool operator=(const KeyDesc &comp) const { return ((dt_ == comp.dt_) && (op_ == comp.op_)); }
  bool operator<(const KeyDesc &comp) const { return (op_ != comp.op_) ? (op_ < comp.op_) : (dt_ < comp.dt_); }
};

typedef NnaclKernel *(*NnaclCreator)(OpParameter *parameter, const std::vector<lite::Tensor *> &in,
                                     const std::vector<lite::Tensor *> &out, const lite::InnerContext *ctx);

class KernelRegistry {
 public:
  static KernelRegistry *GetInstance() {
    static KernelRegistry instance;
    return &instance;
  }
  void Register(KeyDesc desc, NnaclCreator creator) { nnacl_map_[desc] = creator; }
  NnaclCreator Creator(KeyDesc desc) {
    auto iter = nnacl_map_.find(desc);
    if (iter != nnacl_map_.end()) {
      return iter->second;
    }
    return nullptr;
  }

 protected:
  std::map<KeyDesc, NnaclCreator> nnacl_map_;
};

class NnaclKernelRegistrar {
 public:
  NnaclKernelRegistrar(int op_type, TypeId data_type, NnaclCreator creator) {
    KernelRegistry::GetInstance()->Register({op_type, data_type}, creator);
  }
  ~NnaclKernelRegistrar() = default;
};

template <class T>
NnaclKernel *NnaclOpt(OpParameter *parameter, const std::vector<lite::Tensor *> &in,
                      const std::vector<lite::Tensor *> &out, const lite::InnerContext *ctx) {
  auto *kernel = new (std::nothrow) T(parameter, in, out, ctx);
  return kernel;
}

#define NNACL_KERNEL(op_type, data_type, creator) \
  static NnaclKernelRegistrar g_kernel##op_type##data_type##kernelReg(op_type, data_type, creator);

NnaclKernel *NnaclKernelRegistry(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                 const kernel::KernelKey &key);
}  // namespace mindspore::nnacl
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_KERNEL_H_
