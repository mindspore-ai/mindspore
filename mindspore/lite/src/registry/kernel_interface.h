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

#ifndef MINDSPORE_LITE_SRC_REGISTRY_KERNEL_INTERFACE_H_
#define MINDSPORE_LITE_SRC_REGISTRY_KERNEL_INTERFACE_H_

#include <set>
#include <string>
#include <vector>
#include <memory>
#include "include/model.h"
#include "include/ms_tensor.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace kernel {
struct MS_API CapabilityParam {
  float exec_time_;
  float power_usage_;
};

class MS_API KernelInterface {
 public:
  virtual ~KernelInterface() = default;
  virtual int Infer(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
                    const schema::Primitive *primitive) {
    return 0;
  }

  virtual int GetCapability(const std::vector<tensor::MSTensor *> &tensor_in, const schema::Primitive *primitive,
                            CapabilityParam *param) {
    return 0;
  }
};

using KernelInterfaceCreator MS_API = std::function<std::shared_ptr<KernelInterface>()>;

class MS_API RegisterKernelInterface {
 public:
  static int CustomReg(const std::string &provider, const std::string &op_type, KernelInterfaceCreator creator);
  static int Reg(const std::string &provider, int op_type, KernelInterfaceCreator creator);
  static bool CheckReg(const lite::Model::Node *node, std::set<std::string> &&providers);
  static std::shared_ptr<kernel::KernelInterface> GetKernelInterface(const std::string &provider,
                                                                     const schema::Primitive *primitive);
};

class MS_API KernelInterfaceReg {
 public:
  KernelInterfaceReg(const std::string &provider, int op_type, KernelInterfaceCreator creator) {
    RegisterKernelInterface::Reg(provider, op_type, creator);
  }

  KernelInterfaceReg(const std::string &provider, const std::string &op_type, KernelInterfaceCreator creator) {
    RegisterKernelInterface::CustomReg(provider, op_type, creator);
  }
};

#define REGISTER_KERNEL_INTERFACE(provider, op_type, creator)                               \
  namespace {                                                                               \
  static KernelInterfaceReg g_##provider##op_type##_inter_reg(#provider, op_type, creator); \
  }  // namespace

#define REGISTER_CUSTOM_KERNEL_INTERFACE(provider, op_type, creator)                                \
  namespace {                                                                                       \
  static KernelInterfaceReg g_##provider##op_type##_custom_inter_reg(#provider, #op_type, creator); \
  }  // namespace
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_REGISTRY_KERNEL_INTERFACE_H_
