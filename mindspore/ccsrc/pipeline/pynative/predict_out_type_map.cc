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

#include "pipeline/pynative/predict_out_type_map.h"
#include <string>
#include <vector>
#include "ops/op_def.h"

namespace mindspore {
namespace pynative {
namespace {
inline TypePtr PredictOutTypeByOutputNum(const int64_t &output_num) {
  static const std::vector<TypePtr> types({kTuple, kTensorType, kTupleTensor2, kTupleTensor3, kTupleTensor4,
                                           kTupleTensor5, kTupleTensor6, kTupleTensor7, kTupleTensor8, kTupleTensor9});
  constexpr int64_t kZero = 0;
  constexpr int64_t kTen = 10;
  if (output_num > kZero && output_num < kTen) {
    return types[output_num];
  }
  return kTuple;
}
}  // namespace

TypePtr PredictOutTypeByOpDef(const ops::OpDefPtr op_def) {
  auto returns_num = op_def->returns_.size();
  if (returns_num == 1) {
    if (op_def->returns_[0].arg_dtype_ == ops::OP_DTYPE::DT_TENSOR) {
      return kTensorType;
    }

    if (op_def->returns_[0].arg_dtype_ == ops::OP_DTYPE::DT_LIST_TENSOR ||
        op_def->returns_[0].arg_dtype_ == ops::OP_DTYPE::DT_TUPLE_TENSOR) {
      return kTuple;
    }

    return kTypeNone;
  }

  static const std::vector<TypePtr> kSequenceTypes = {
    kTuple,  // this is only a placeholder
    kTuple,  // this is only a placeholder
    kTupleTensor2, kTupleTensor3, kTupleTensor4, kTupleTensor5,
    kTupleTensor6, kTupleTensor7, kTupleTensor8, kTupleTensor9,
  };

  if (returns_num >= kSequenceTypes.size()) {
    MS_LOG(EXCEPTION) << "For " << op_def->name_ << ", the number of output must be less than " << kSequenceTypes.size()
                      << ", but got " << returns_num << ".";
  }

  return kSequenceTypes[returns_num];
}

TypePtr PredictOutTypeByName(const std::string &op_name) {
  static PredictOutTypeMap ops_map;
  const auto iter = ops_map.find(op_name);
  if (iter != ops_map.end()) {
    return iter->second;
  }
  auto op_def = ops::GetOpDef(op_name);
  if (op_def != nullptr) {
    auto type = PredictOutTypeByOpDef(op_def);
    MS_LOG(DEBUG) << "PredictOutTypeByOpDef: " << type->ToString();
    return ops_map[op_name] = type;
  }

  static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  if (operator_fns.find(op_name) == operator_fns.end()) {
    return ops_map[op_name] = kTypeNone;
  }
  const auto pre_iter = out_type_prediction.find(op_name);
  auto type = pre_iter == out_type_prediction.end() ? kTensorType : pre_iter->second;
  return ops_map[op_name] = type;
}

TypePtr PredictOutType(const FrontendOpRunInfoPtr &op_run_info) {
  const auto &op_name = op_run_info->base_op_run_info.op_name;
  auto type = PredictOutTypeByName(op_name);
  if (type == kTypeAny) {
    const auto &op_prim = op_run_info->op_grad_info->op_prim;
    if (const auto &attr = op_prim->GetAttr("output_num"); attr != nullptr) {
      type = PredictOutTypeByOutputNum(GetValue<int64_t>(attr));
    }
  }
  return type;
}
}  // namespace pynative
}  // namespace mindspore
