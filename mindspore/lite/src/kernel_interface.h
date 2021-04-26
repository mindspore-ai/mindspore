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

#ifndef MINDSPORE_LITE_SRC_KERNEL_DEV_DELEGATE_H_
#define MINDSPORE_LITE_SRC_KERNEL_DEV_DELEGATE_H_

#include <string>
#include <vector>
#include "include/ms_tensor.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace kernel {
struct CapabilityParam {
  float exec_time_;
  float power_usage_;
};

class KernelInterface {
 public:
  virtual ~KernelInterface() = default;
  virtual int Infer(const std::vector<tensor::MSTensor *> &tensor_in, std::vector<tensor::MSTensor *> *outputs,
                    const schema::Primitive *primitive) {
    return 0;
  }

  virtual int GetCapability(const std::vector<tensor::MSTensor *> &tensor_in, const schema::Primitive *primitive,
                            CapabilityParam *param) {
    return 0;
  }
};
typedef KernelInterface *(*KernelInterfaceCreator)();

class RegisterKernelInterface {
 public:
  static RegisterKernelInterface *Instance();
  int Reg(const std::string &vendor, const int op_type, KernelInterfaceCreator creator);
  virtual ~RegisterKernelInterface() = default;

 private:
  RegisterKernelInterface() = default;
};

class KernelInterfaceReg {
 public:
  KernelInterfaceReg(const std::string &vendor, const int op_type, KernelInterfaceCreator creator) {
    RegisterKernelInterface::Instance()->Reg(vendor, op_type, creator);
  }
  ~KernelInterfaceReg() = default;
};

#define REGISTER_KERNEL_INTERFACE(vendor, op_type, creator) \
  static KernelInterfaceReg g_##vendor##op_type##_inter_reg(vendor, op_type, creator);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_KERNEL_DEV_DELEGATE_H_
