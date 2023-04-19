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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_LIB_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_LIB_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <utility>
#include "src/common/log_adapter.h"
#include "mindapi/base/format.h"
#include "src/extendrt/kernel/primitive_type.h"
#include "src/infer/kernel.h"
#include "src/infer/tensor.h"
#include "src/infer/context.h"
#include "src/litert/lite_kernel.h"
#include "ops/base_operator.h"
#include "kernel/common_utils.h"
#include "src/executor/kernel_exec.h"

namespace mindspore::kernel {
using InferTensor = mindspore::infer::abstract::Tensor;
using InferContext = mindspore::infer::abstract::Context;

struct KernelSpec {
  PrimitiveType op_type;
  KernelAttr attr;
  Format format;
  BaseOperatorPtr primitive;
  std::string backend;
};

class KernelLib {
 public:
  KernelLib(std::string name, std::string backend) : name_(std::move(name)), backend_(std::move(backend)) {}
  virtual ~KernelLib() = default;
  virtual bool Support(const PrimitiveType &op_type, const KernelAttr &attr, const Format &format) const = 0;
  virtual LiteKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                                   const std::vector<InferTensor *> &outputs, const InferContext *ctx) const = 0;

  KernelExec *CreateKernelExec(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                               const std::vector<InferTensor *> &outputs, const InferContext *ctx) {
    auto *lite_kernel = this->CreateKernel(spec, inputs, outputs, ctx);
    if (lite_kernel == nullptr) {
      MS_LOG(ERROR) << "Create lite kernel failed. kernel: " << spec.op_type.PBType();
      return nullptr;
    }
    auto *kernel_exec = new (std::nothrow) KernelExec(std::shared_ptr<LiteKernel>(lite_kernel));
    if (kernel_exec == nullptr) {
      MS_LOG(ERROR) << "Create kernel exec failed. kernel: " << spec.op_type;
      return nullptr;
    }
    auto desc = kernel_exec->desc();
    desc.format = spec.format;
    kernel_exec->set_desc(desc);
    kernel_exec->set_context(ctx);
    return kernel_exec;
  }

  std::string Name() const { return name_; }
  std::string Backend() const { return backend_; }

 protected:
  static bool MatchFormat(const Format &format1, const Format &format2) {
    if (format1 == Format::DEFAULT_FORMAT || format2 == Format::DEFAULT_FORMAT) {
      return true;
    }
    return format1 == format2;
  }

 protected:
  std::string name_;  //  provider
  std::string backend_;
};

class KernelLibRegister {
 public:
  static KernelLibRegister &Instance() {
    static KernelLibRegister instance;
    return instance;
  }

  virtual ~KernelLibRegister() {
    for (auto &iter : kernel_libs_) {
      delete iter.second;
    }
    kernel_libs_.clear();
  }

  bool RegKernelLib(const std::string &provider, const KernelLib *lib) {
    auto iter = kernel_libs_.find(provider);
    if (MS_LIKELY(iter != kernel_libs_.end())) {
      MS_LOG(ERROR) << "KernelLib " << provider << " is already exist.";
      return false;
    }
    kernel_libs_[provider] = lib;
    return true;
  }

  KernelLib *GetKernelLib(const std::string &provider) {
    auto iter = kernel_libs_.find(provider);
    if (MS_LIKELY(iter == kernel_libs_.end())) {
      MS_LOG(ERROR) << "KernelLib " << provider << " is not exist.";
      return nullptr;
    }
    return const_cast<KernelLib *>(iter->second);
  }

  const std::unordered_map<std::string, const KernelLib *> &GetAllLibs() { return kernel_libs_; }

 private:
  KernelLibRegister() = default;

 private:
  // map from provider/name of kernel-lib to kernel-lib
  std::unordered_map<std::string, const KernelLib *> kernel_libs_;
};

class KernelLibRegistry {
 public:
  KernelLibRegistry(const std::string &provider, const KernelLib *lib) {
    if (MS_UNLIKELY(lib == nullptr)) {
      MS_LOG(WARNING) << "KernelLib " << provider << " is nullptr, ignored.";
      return;
    }
    KernelLibRegister::Instance().RegKernelLib(provider, lib);
  }
};

#define REG_KERNEL_LIB(name, class) static KernelLibRegistry g_##class##Registry(name, new (std::nothrow) class())
}  // namespace mindspore::kernel
#endif
