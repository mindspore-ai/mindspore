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

#ifndef MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_H_
#define MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_H_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include "nnacl/op_base.h"
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace lite {
using BaseOperatorPtr = std::shared_ptr<mindspore::ops::BaseOperator>;
typedef OpParameter *(*ParameterPtrGen)(const BaseOperatorPtr &base_operator);

static const std::map<std::string, int> kOpNameWithPrimitiveType = {{"Add", 5},
                                                                    {"Assert", 17},
                                                                    {"Equal", 53},
                                                                    {"FloorDiv", 64},
                                                                    {"FloorMod", 65},
                                                                    {"Greater", 71},
                                                                    {"GreaterEqual", 72},
                                                                    {"Less", 77},
                                                                    {"LessEqual", 78},
                                                                    {"LogicalAnd", 81},
                                                                    {"LogicalOr", 83},
                                                                    {"Maximum", 90},
                                                                    {"Minimum", 96},
                                                                    {"Mod", 98},
                                                                    {"NotEqual", 103},
                                                                    {"RealDiv", 117},
                                                                    {"SquaredDifference", 149}};

class OperatorPopulateRegistry {
 public:
  static OperatorPopulateRegistry *GetInstance();

  void InsertOperatorParameterMap(const std::string &name, ParameterPtrGen creator) { op_parameters_[name] = creator; }

  ParameterPtrGen GetParameterPtrCreator(const std::string &name) {
    auto iter = op_parameters_.find(name);
    if (iter == op_parameters_.end()) {
      MS_LOG(ERROR) << "Unsupported op in creator " << name;
      return nullptr;
    }
    return iter->second;
  }

 protected:
  std::map<std::string, ParameterPtrGen> op_parameters_;
};
class Registry {
 public:
  Registry(std::string name, ParameterPtrGen creator) noexcept {
    OperatorPopulateRegistry::GetInstance()->InsertOperatorParameterMap(name, creator);
  }

  ~Registry() = default;
};

#define REG_OPERATOR_POPULATE(name, creator) static Registry g_##name(name, creator);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_H_
