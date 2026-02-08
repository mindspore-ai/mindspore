/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "ops/infer_info/infer_info_utils.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include "ops/op_def.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/infer_info/infer_info.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "abstract/utils.h"
#include "ops/op_def_utils.h"
#include "mindspore/core/ops/infer_info/value_infer_info_adapter.h"
#include "ops/infer_info/abstract_infer_info_adapter.h"

namespace mindspore::ops {
namespace {
constexpr size_t kSingleTensor = 1;
inline bool IsTupleOutput(const OpDefPtr &op_def) {
  MS_EXCEPTION_IF_NULL(op_def);
  const auto &returns = op_def->returns_;
  if (returns.size() == 0) {
    MS_LOG(EXCEPTION) << op_def->name_ << "'s output is empty, which is not allowed.";
  }
  auto output_type = returns[0].arg_dtype_;
  auto is_tuple_output = (returns.size() > kSingleTensor) ||
                         (output_type <= OP_DTYPE::DT_LIST_ANY && output_type >= OP_DTYPE::DT_TUPLE_BOOL);
  return is_tuple_output;
}
}  // namespace
/*
  Deal with dynamic sequence.
  Dynamic sequence would be split to tensors, so abstract_list.size() might not match the defined input size.
*/
static std::vector<std::string> GetArgNames(const AbstractBasePtrList &abstract_list, const OpDefPtr op_def) {
  size_t n = abstract_list.size();
  const std::string &op_type = op_def->name_;
  std::vector<std::string> arg_names;
  MS_EXCEPTION_IF_NULL(op_def);
  if (op_def->args_.empty()) {
    return arg_names;
  }
  if (!abstract_list.empty()) {  // ignore monad for side effect ops
    MS_EXCEPTION_IF_NULL(abstract_list.back());
    if (abstract_list.back()->isa<abstract::AbstractMonad>()) {
      n -= 1;
    }
  }
  size_t op_arg_size = op_def->args_.size();
  MS_ASSERT_TRUE(n >= op_arg_size) << "Abstract size [" << n << "] is smaller than the defined input size ["
                                   << op_arg_size << "] for primitive " << op_type;

  std::vector<size_t> sequence_input_indices;
  for (size_t i = 0; i < op_arg_size; ++i) {
    if (op_def->args_[i].arg_dtype_ > DT_ANY) {
      sequence_input_indices.push_back(i);
    }
  }
  if (sequence_input_indices.size() > 1) {  // more than one sequence input, unable to distinguish inputs
    for (size_t i = 0; i < n; ++i) {
      arg_names.push_back(op_type + "_" + std::to_string(i));
    }
  } else if (sequence_input_indices.size() == 1) {  // one sequence input
    const size_t sequence_idx = sequence_input_indices.front();
    size_t tuple_size = n - op_arg_size + 1;
    auto sequence_arg_name = op_def->args_[sequence_idx].arg_name_;
    std::transform(op_def->args_.begin(), op_def->args_.begin() + sequence_idx, std::back_inserter(arg_names),
                   [](const OpInputArg &arg) { return arg.arg_name_; });
    for (size_t i = 0; i < tuple_size; ++i) {
      arg_names.push_back(sequence_arg_name + "_" + std::to_string(i));
    }
    std::transform(op_def->args_.begin() + sequence_idx + 1, op_def->args_.end(), std::back_inserter(arg_names),
                   [](const OpInputArg &arg) { return arg.arg_name_; });
  } else {
    MS_CHECK_VALUE(n == op_arg_size, "Abstract number [" + std::to_string(n) +
                                       "] doesn't match defined input number [" + std::to_string(op_arg_size) +
                                       "] for primitive " + op_type);
    std::transform(op_def->args_.begin(), op_def->args_.end(), std::back_inserter(arg_names),
                   [](const OpInputArg &arg) { return arg.arg_name_; });
  }
  return arg_names;
}

InferInfoPtrList ConvertAbstractListToInferInfoList(const AbstractBasePtrList &abstract_list, const OpDefPtr op_def) {
  InferInfoPtrList infer_infos;
  MS_EXCEPTION_IF_NULL(op_def);
  const auto &op_type = op_def->name_;
  const auto &arg_names = GetArgNames(abstract_list, op_def);
  auto size = arg_names.size();
  for (size_t i = 0; i < size; ++i) {
    infer_infos.push_back(std::make_unique<AbstractInferInfoAdapter>(abstract_list[i], op_type, arg_names[i]));
  }
  return infer_infos;
}

AbstractBasePtr DoGeneralInfer(const PrimitivePtr primitive, const AbstractBasePtrList &abstract_list,
                               const OpFrontendFuncImplPtr frontend_func_impl) {
  const auto &op_type = primitive->name();
  MS_LOG(DEBUG) << "DoGeneralInfer for op " << op_type;
  const auto op_def = GetOpDef(op_type);
  MS_EXCEPTION_IF_NULL(op_def);
  const std::vector<InferInfoPtr> &infer_infos = ConvertAbstractListToInferInfoList(abstract_list, op_def);
  (void)op_def->func_impl_.CheckValidation(primitive, infer_infos);
  if (frontend_func_impl != nullptr) {
    auto infer_result = frontend_func_impl->InferAbstract(primitive, abstract_list);
    if (infer_result != nullptr) {
      return infer_result;
    }
  }
  const auto &types = op_def->func_impl_.InferType(primitive, infer_infos);
  const auto &shapes = op_def->func_impl_.InferShape(primitive, infer_infos);
  if (types.size() != shapes.size()) {
    MS_LOG(EXCEPTION) << "Infer shape size " << shapes.size() << " not equal to infer type size " << types.size()
                      << " for op " << op_type;
  }
  return abstract::MakeAbstract(shapes, types, IsTupleOutput(op_def));
}

ValueSimpleInfoPtr DoGeneralInfer(const PrimitivePtr &prim, const ValuePtrList &values) {
  MS_EXCEPTION_IF_NULL(prim);
  const auto &op_type = prim->name();
  const auto op_def = mindspore::ops::GetOpDef(op_type);
  if (!op_def || !op_def->func_impl_.GeneralInferRegistered()) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "DoGeneralInfer for op " << op_type;
  MS_ASSERT_TRUE(values.size() == op_def->args_.size())
    << "ValuePtr number [" << values.size() << "] does not match the defined input number [" << op_def->args_.size()
    << "] for primitive " << op_type;
  std::vector<ops::InferInfoPtr> input_infos(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    input_infos[i] = std::make_unique<ops::ValueInferInfoAdapter>(values[i], op_type, op_def->args_[i].arg_name_);
  }
  (void)op_def->func_impl_.CheckValidation(prim, input_infos);
  auto &&shapes = op_def->func_impl_.InferShape(prim, input_infos);
  const auto &types = op_def->func_impl_.InferType(prim, input_infos);
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "Infer shape size " << shapes.size() << " not equal to infer type size " << types.size()
                      << " for op " << prim->name();
  }
  auto value_simple_info = std::make_shared<ValueSimpleInfo>();
  value_simple_info->is_tuple_output_ = IsTupleOutput(op_def);
  value_simple_info->shape_vector_ = std::move(shapes);
  std::transform(types.begin(), types.end(), std::back_inserter(value_simple_info->dtype_vector_),
                 [](const TypeId &type) { return TypeIdToType(type); });
  value_simple_info->size_ = types.size();
  return value_simple_info;
}
}  // namespace mindspore::ops
