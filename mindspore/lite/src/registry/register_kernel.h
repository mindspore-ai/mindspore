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

#ifndef MINDSPORE_LITE_SRC_REGISTRY_REGISTER_KERNEL_H_
#define MINDSPORE_LITE_SRC_REGISTRY_REGISTER_KERNEL_H_

#include <set>
#include <string>
#include <vector>
#include <memory>
#include "schema/model_generated.h"
#include "include/context.h"
#include "include/ms_tensor.h"
#include "src/kernel.h"

namespace mindspore {
namespace kernel {
struct MS_API KernelDesc {
  TypeId data_type;
  int type;
  std::string arch;
  std::string provider;

  bool operator<(const KernelDesc &dst) const {
    if (provider != dst.provider) {
      return provider < dst.provider;
    } else if (arch != dst.arch) {
      return arch < dst.arch;
    } else if (data_type != dst.data_type) {
      return data_type < dst.data_type;
    } else {
      return type < dst.type;
    }
  }
};

using CreateKernel MS_API = std::function<std::shared_ptr<kernel::Kernel>(
  const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
  const schema::Primitive *primitive, const lite::Context *ctx)>;

class MS_API RegisterKernel {
 public:
  static int RegKernel(const std::string &arch, const std::string &provider, TypeId data_type, int type,
                       CreateKernel creator);
  static int RegCustomKernel(const std::string &arch, const std::string &provider, TypeId data_type,
                             const std::string &type, CreateKernel creator);
  static CreateKernel GetCreator(const kernel::KernelDesc &desc, const schema::Primitive *primitive);
};

class MS_API KernelReg {
 public:
  ~KernelReg() = default;

  KernelReg(const std::string &arch, const std::string &provider, TypeId data_type, int op_type, CreateKernel creator) {
    RegisterKernel::RegKernel(arch, provider, data_type, op_type, creator);
  }

  KernelReg(const std::string &arch, const std::string &provider, TypeId data_type, const std::string &op_type,
            CreateKernel creator) {
    RegisterKernel::RegCustomKernel(arch, provider, data_type, op_type, creator);
  }
};

#define REGISTER_KERNEL(arch, provider, data_type, op_type, creator)                                                 \
  namespace {                                                                                                        \
  static KernelReg g_##arch##provider##data_type##op_type##kernelReg(#arch, #provider, data_type, op_type, creator); \
  }  // namespace

#define REGISTER_CUSTOM_KERNEL(arch, provider, data_type, op_type, creator)                                           \
  namespace {                                                                                                         \
  static KernelReg g_##arch##provider##data_type##op_type##kernelReg(#arch, #provider, data_type, #op_type, creator); \
  }  // namespace
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_REGISTRY_REGISTER_KERNEL_H_
