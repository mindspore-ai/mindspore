/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "abstract/param_validator.h"
#include "abstract/ops/infer_functions.h"
#include "abstract/abstract_function.h"
#include "abstract/utils.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplReturn(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_abs_list) {
  // Inputs: a pointer to an AbstractBase object
  if (args_abs_list.size() != 1) {
    MS_LOG(INFO) << "Return evaluator requires 1 parameter, is this the default value attached? "
                    "while the input size is "
                 << args_abs_list.size() << ".";
  }
  AbstractBasePtr abs_base = args_abs_list[0];
  return abs_base;
}

void CheckTensorCondValid(const AbstractBasePtr &cond) {
  // Tensor condition must be one element or dynamic shape.
  auto base_shape = cond->BuildShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  ShapeVector cond_shape = base_shape->cast<ShapePtr>()->shape();
  if (cond_shape.empty()) {
    return;
  }
  constexpr auto num_one = 1;
  for (size_t i = 0; i < cond_shape.size(); i++) {
    if (cond_shape[i] != num_one && cond_shape[i] != Shape::kShapeDimAny && cond_shape[i] != Shape::kShapeRankAny) {
      MS_LOG(ERROR) << "The condition value of control flow can be a tensor with one element, "
                    << "but got tensor with shape " << base_shape->ToString();
      MS_EXCEPTION(ValueError) << "The truth value of an array with more than one element is ambiguous.";
    }
  }
}

AbstractBasePtr InferImplSwitch(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_abs_list) {
  // Inputs: condition, true branch, false branch
  constexpr auto switch_input_size = 3;
  if (args_abs_list.size() != switch_input_size) {
    MS_LOG(EXCEPTION) << "Switch evaluator requires 3 parameters, while the input size is " << args_abs_list.size()
                      << ".";
  }

  auto cond_abstract = args_abs_list[0];
  auto true_branch = args_abs_list[1];
  auto false_branch = args_abs_list[2];
  MS_EXCEPTION_IF_NULL(cond_abstract);

  ValuePtr cond_value = cond_abstract->GetValueTrack();
  MS_EXCEPTION_IF_NULL(cond_value);
  // If the value of condition is ValueAny or the abstract of condition is AbstractTensor,
  // keeps both true and false branch.
  if (cond_value->isa<ValueAny>() || cond_abstract->isa<AbstractTensor>()) {
    if (cond_abstract->isa<AbstractTensor>()) {
      CheckTensorCondValid(cond_abstract);
    }
    MS_EXCEPTION_IF_NULL(true_branch);
    // Need record two func_graph
    SetVariableFlag(true_branch);
    SetVariableFlag(false_branch);
    return true_branch->Join(false_branch);
  }

  if (cond_value->isa<Scalar>()) {
    if (cond_value->cast<ScalarPtr>()->IsOne()) {
      return true_branch;
    } else {
      return false_branch;
    }
  }
  MS_LOG(EXCEPTION) << "Not support this condition value: " << cond_value->ToString();
}

AbstractBasePtr InferImplSwitchLayer(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_abs_list) {
  // Inputs: {index, MakeTuple{branch1,branch2,branch3....}}
  constexpr auto kSwitchLayerInputNum = 2;
  const std::string op_name = primitive->name();
  abstract::CheckArgsSize(op_name, args_abs_list, kSwitchLayerInputNum);
  auto index = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  auto &input_shape = index->shape()->shape();
  if (!input_shape.empty() && (input_shape.size() != 1 || input_shape[0] != 1)) {
    MS_EXCEPTION(ValueError) << op_name << " index must be a 0 dimension tensor, but got a " << input_shape.size()
                             << " dimension tensor";
  }
  auto dtype = index->element()->BuildType();
  if (dtype->type_id() != kInt32->type_id()) {
    MS_EXCEPTION(ValueError) << op_name << " index must be an int32, but got " << dtype->ToString();
  }

  AbstractTuplePtr branches_abs = CheckArg<AbstractTuple>(op_name, args_abs_list, 1);
  AbstractBasePtrList branches = branches_abs->elements();
  const size_t maximum_layer_num = 1000;
  if (branches.empty() || branches.size() > maximum_layer_num) {
    MS_EXCEPTION(ValueError) << op_name << " support at least 1 and at most " << maximum_layer_num << " but got "
                             << branches.size() << " branches.";
  }

  for (size_t i = 0; i < branches.size(); i++) {
    MS_EXCEPTION_IF_NULL(branches[i]);
    if (!branches[i]->isa<FuncGraphAbstractClosure>() && !branches[i]->isa<PartialAbstractClosure>()) {
      MS_EXCEPTION(ValueError) << op_name << " requires that the 2th arg be tuple of functions, but got "
                               << branches[i]->ToString() << " as the " << i << "th element.";
    }
  }

  auto b = branches[0];
  SetVariableFlag(b);
  // Return AbstractFuncUnion, otherwise the switch_layer will be replaced by branches[0]
  // which will cancel the out of bound checking for index
  if (branches.size() == 1) {
    AbstractFuncAtomPtrList func_list{b->cast<AbstractFuncAtomPtr>()};
    return std::make_shared<AbstractFuncUnion>(func_list);
  }
  for (size_t i = 1; i < branches.size(); i++) {
    SetVariableFlag(branches[i]);
    b = b->Join(branches[i]);
  }
  return b;
}

bool SupportedIsTargetValue(const ValuePtr t) {
  std::vector<ValuePtr> list = {kNone, MakeValue(false), MakeValue(true)};
  return std::any_of(list.begin(), list.end(), [&t](const ValuePtr &v) { return *v == *t; });
}

bool CheckIfDataIsTarget(const std::string &op_name, const AbstractBasePtr &data_abs,
                         const AbstractBasePtr &target_abs) {
  MS_EXCEPTION_IF_NULL(target_abs);
  // Check if data and target are both None.
  if (data_abs->isa<AbstractNone>() || target_abs->isa<AbstractNone>()) {
    return data_abs->isa<AbstractNone>() && target_abs->isa<AbstractNone>();
  }
  const auto &target_value = target_abs->BuildValue();
  const auto &target_type = target_abs->BuildType();
  MS_EXCEPTION_IF_NULL(target_value);
  MS_EXCEPTION_IF_NULL(target_type);
  if (!SupportedIsTargetValue(target_value) && !target_type->isa<TypeType>()) {
    MS_LOG(EXCEPTION) << "For syntax like 'a " << op_name << " b', b supports True, False, None and Type, but got "
                      << target_value->ToString();
  }
  const auto &data_value = data_abs->BuildValue();
  MS_EXCEPTION_IF_NULL(data_value);
  return *data_value == *target_value;
}

AbstractBasePtr InferImplIs_(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_abs_list) {
  // Statement: x is t
  // Inputs: x, t
  constexpr size_t kInputsNum = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, kInputsNum);
  constexpr size_t data_index = 0;
  constexpr size_t target_index = 1;
  bool res = CheckIfDataIsTarget("is", args_abs_list[data_index], args_abs_list[target_index]);
  return std::make_shared<AbstractScalar>(res);
}

AbstractBasePtr InferImplIsNot(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_abs_list) {
  // Statement: x is not t
  // Inputs: x, t
  constexpr size_t kInputsNum = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, kInputsNum);
  constexpr size_t data_index = 0;
  constexpr size_t target_index = 1;
  bool res = CheckIfDataIsTarget("is not", args_abs_list[data_index], args_abs_list[target_index]);
  return std::make_shared<AbstractScalar>(!res);
}

bool IsInDict(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list) {
  constexpr size_t kInputsNum = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_abs_list, kInputsNum);
  const auto &key = args_abs_list[0];
  auto dict = CheckArg<AbstractDictionary>(op_name, args_abs_list, 1);

  ValuePtr key_value = key->BuildValue();
  MS_EXCEPTION_IF_NULL(key_value);
  std::vector<AbstractElementPair> dict_elems = dict->elements();
  auto it = std::find_if(dict_elems.cbegin(), dict_elems.cend(), [&key_value](const AbstractElementPair &item) {
    return *key_value == *item.first->BuildValue();
  });
  return it != dict_elems.end();
}

AbstractBasePtr InferImplInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_abs_list) {
  // Statement: x in t
  // Inputs: x, t
  return std::make_shared<AbstractScalar>(IsInDict(primitive, args_abs_list));
}

AbstractBasePtr InferImplNotInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_abs_list) {
  // Statement: x not in t
  // Inputs: x, t
  return std::make_shared<AbstractScalar>(!IsInDict(primitive, args_abs_list));
}

AbstractBasePtr InferImplIsConstant(const AnalysisEnginePtr &, const PrimitivePtr &,
                                    const AbstractBasePtrList &args_abs_list) {
  // Statement: isconstant(x)
  // Inputs: x
  if (args_abs_list.size() != 1) {
    MS_LOG(EXCEPTION) << "IsConstant requires args input size = 1";
  }
  ValuePtr v = args_abs_list[0]->BuildValue();
  return std::make_shared<AbstractScalar>(!v->isa<ValueAny>());
}
}  // namespace abstract
}  // namespace mindspore
