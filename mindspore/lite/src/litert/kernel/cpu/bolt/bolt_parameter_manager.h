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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_POPULATOR_PARAMETER_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_POPULATOR_PARAMETER_H_

#include <map>
#include "bolt/common/uni/include/parameter_spec.h"
#include "nnacl/op_base.h"
#include "src/common/log_adapter.h"

namespace mindspore::kernel::bolt {
typedef ParameterSpec *(*BoltParameterPtrGen)(const OpParameter *op_parameter);

class BoltParameterRegistry {
 public:
  static BoltParameterRegistry *GetInstance() {
    static BoltParameterRegistry registry;
    return &registry;
  }

  void InsertParameterMap(int type, BoltParameterPtrGen creator) { op_parameters_[type] = creator; }

  ParameterSpec *CreateBoltParameter(const OpParameter *op_parameter) {
    MS_CHECK_TRUE_RET(op_parameter != nullptr, nullptr);
    auto iter = op_parameters_.find(op_parameter->type_);
    if (iter == op_parameters_.end()) {
      MS_LOG(DEBUG) << "Unsupported op in creator " << op_parameter->type_;
      return nullptr;
    }
    auto bolt_param = iter->second(op_parameter);
    return bolt_param;
  }

 protected:
  std::map<int, BoltParameterPtrGen> op_parameters_;
};

class Registry {
 public:
  Registry(int type, BoltParameterPtrGen creator) noexcept {
    BoltParameterRegistry::GetInstance()->InsertParameterMap(type, creator);
  }
  ~Registry() = default;
};

#define REG_BOLT_PARAMETER_POPULATE(type, creator) static Registry g_##type(type, creator);
}  // namespace mindspore::kernel::bolt
#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_POPULATOR_PARAMETER_H_
