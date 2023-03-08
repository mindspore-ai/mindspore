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
#include <utility>
#include "nnacl/op_base.h"
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "ops/base_operator.h"
#include "mindspore/core/ir/primitive.h"
#include "ops/primitive_c.h"
#include "ops/op_name.h"
#include "schema/ops_generated.h"

namespace mindspore {
namespace lite {
using BaseOperatorPtr = std::shared_ptr<mindspore::ops::BaseOperator>;
typedef OpParameter *(*ParameterPtrGen)(const BaseOperatorPtr &base_operator);
OpParameter *CreatePopulatePtr(const BaseOperatorPtr &base_operator);

class OperatorPopulateRegistry {
 public:
  static OperatorPopulateRegistry *GetInstance();

  void InsertOperatorParameterMap(const std::string &name, int type, ParameterPtrGen creator) {
    op_parameters_[name] = std::make_pair(creator, type);
  }
  ParameterPtrGen GetParameterPtrCreator(const std::string &name) { return CreatePopulatePtr; }

  OpParameter *CreatePopulateByOp(const BaseOperatorPtr &base_operator) {
    MS_CHECK_TRUE_RET(base_operator != nullptr, nullptr);
    auto iter = op_parameters_.find(base_operator->name());
    if (iter == op_parameters_.end()) {
      MS_LOG(ERROR) << "Unsupported op in creator " << base_operator->name();
      return nullptr;
    }
    if (base_operator->GetPrim() == nullptr) {
      MS_LOG(ERROR) << "invalid op " << base_operator->name();
      return nullptr;
    }
    auto param = iter->second.first(base_operator);
    if (param != nullptr) {
      param->type_ = iter->second.second;
    }
    return param;
  }

 protected:
  std::map<std::string, std::pair<ParameterPtrGen, int>> op_parameters_;
};
class Registry {
 public:
  Registry(std::string name, int type, ParameterPtrGen creator) noexcept {
    OperatorPopulateRegistry::GetInstance()->InsertOperatorParameterMap(name, type, creator);
  }

  ~Registry() = default;
};

template <typename T>
OpParameter *PopulateOpParameter() {
  auto op_parameter_ptr = reinterpret_cast<OpParameter *>(malloc(sizeof(T)));
  if (op_parameter_ptr == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter ptr failed";
    return nullptr;
  }
  memset(op_parameter_ptr, 0, sizeof(T));
  return reinterpret_cast<OpParameter *>(op_parameter_ptr);
}

template <typename T>
OpParameter *PopulateOpParameter(const BaseOperatorPtr &base_operator) {
  auto op_parameter_ptr = reinterpret_cast<OpParameter *>(malloc(sizeof(T)));
  if (op_parameter_ptr == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter ptr failed";
    return nullptr;
  }
  memset(op_parameter_ptr, 0, sizeof(T));
  return reinterpret_cast<OpParameter *>(op_parameter_ptr);
}

#define REG_OPERATOR_POPULATE(name, type, creator) static Registry g_##name(name, type, creator);

#define REG_OP_BASE_POPULATE(op)               \
  using mindspore::ops::kName##op;             \
  using mindspore::schema::PrimitiveType_##op; \
  REG_OPERATOR_POPULATE(kName##op, PrimitiveType_##op, PopulateOpParameter<OpParameter>)

#define REG_OP_DEFAULT_POPULATE(op)            \
  using mindspore::ops::kName##op;             \
  using mindspore::schema::PrimitiveType_##op; \
  REG_OPERATOR_POPULATE(kName##op, PrimitiveType_##op, PopulateOpParameter<op##Parameter>)
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_H_
