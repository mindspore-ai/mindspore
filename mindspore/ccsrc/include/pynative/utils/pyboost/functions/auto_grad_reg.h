/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_REG_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_REG_H_

#include <functional>
#include <any>
#include <unordered_map>
#include "include/pynative/utils/pyboost/op_runner.h"
#include "mindspore/ccsrc/pynative/utils/pyboost/functions/auto_generate/auto_grad_op_reg.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class PYBOOST_API AutoGradFactory {
 public:
  static AutoGradFactory &Get();

  void Register(OpType op_type, const std::any &grad_function) {
    auto_grad_functions_[op_type] = grad_function;
    MS_LOG(DEBUG) << "Register auto grad functions for " << op_type;
  }

  template <typename Func>
  Func GetGradFunction(OpType op_type) {
    auto iter = auto_grad_functions_.find(op_type);
    if (iter == auto_grad_functions_.end()) {
      MS_LOG(EXCEPTION) << "Not found auto grad functions for op " << op_type;
    }
    try {
      return std::any_cast<Func>(iter->second);
    } catch (std::bad_any_cast &e) {
      MS_LOG(EXCEPTION) << "Cast grad function " << op_type << " failed, error message " << e.what();
    }
  }

  const OpsAutoGradRegisters &ops_auto_grad_registers() const { return ops_auto_grad_registers_; }

 private:
  AutoGradFactory() = default;
  ~AutoGradFactory() = default;
  DISABLE_COPY_AND_ASSIGN(AutoGradFactory);

  OpsAutoGradRegisters ops_auto_grad_registers_;
  std::unordered_map<OpType, std::any> auto_grad_functions_{};
};

class PYBOOST_API AutoGradRegister {
 public:
  AutoGradRegister(OpType op_type, const std::any &func) { AutoGradFactory::Get().Register(op_type, func); }
  ~AutoGradRegister() = default;
};

#define MS_AUTO_GRAD_REG(OP_TYPE, FUNC) \
  static const kernel::pyboost::AutoGradRegister g_##FUNC##_AutoGrad_register(OP_TYPE, FUNC);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_REG_H_
