/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: a pointer to an AbstractBase object
  if (args_spec_list.size() != 1) {
    MS_LOG(INFO) << "Return evaluator requires 1 parameter, is this the default value attached? "
                    "while the input size is "
                 << args_spec_list.size() << ".";
  }
  AbstractBasePtr abs_base = args_spec_list[0];
  return abs_base;
}

void SetVariableFlag(const AbstractBasePtr abs) {
  if (abs->isa<abstract::AbstractFunction>()) {
    const auto &func_abs = abs->cast<abstract::AbstractFunctionPtr>();
    MS_EXCEPTION_IF_NULL(func_abs);
    auto closure_abs = func_abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    if (closure_abs) {
      auto func = closure_abs->func_graph();
      MS_EXCEPTION_IF_NULL(func);
      func->set_is_tensor_condition_branch(true);
      MS_LOG(DEBUG) << "Set is_tensor_condition_branch for func_graph:" << func->ToString();
    }
  }
}

std::pair<bool, bool> CheckCondAbstractIsInterpretedObj(const AbstractBasePtr &cond) {
  if (abstract::AbstractBase::interpret_bool_checker()) {
    return abstract::AbstractBase::interpret_bool_checker()(cond);
  }
  return {false, false};
}

AbstractBasePtr InferImplSwitch(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: condition, true branch, false branch
  constexpr auto switch_input_size = 3;
  if (args_spec_list.size() != switch_input_size) {
    MS_LOG(EXCEPTION) << "Switch evaluator requires 3 parameters, while the input size is " << args_spec_list.size()
                      << ".";
  }

  auto cond = args_spec_list[0];
  auto tb = args_spec_list[1];
  auto fb = args_spec_list[2];
  MS_EXCEPTION_IF_NULL(cond);

  ValuePtr v = cond->GetValueTrack();
  MS_EXCEPTION_IF_NULL(v);
  // If the value of condition is AnyValue or the abstract of condition is AbstractTensor,
  // keeps both true and false branch.
  if (v->isa<AnyValue>() || cond->isa<AbstractTensor>()) {
    MS_EXCEPTION_IF_NULL(tb);
    // Need record two func_graph
    SetVariableFlag(tb);
    SetVariableFlag(fb);
    return tb->Join(fb);
  }

  if (v->isa<Scalar>()) {
    if (v->cast<ScalarPtr>()->IsOne()) {
      return tb;
    } else {
      return fb;
    }
  }
  TypePtr cond_type = cond->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(cond_type);
  if (cond->isa<AbstractScalar>() && cond_type->type_id() == kMetaTypeExternal) {
    // Check the abstract of condition, if the value is InterpretedObject, and has "True" in it, return true branch.
    // For example, AbstractScalar(Type: kMetaTypeExternal, Value: InterpretedObject: "True", ...)
    auto [is_interpret, has_true] = CheckCondAbstractIsInterpretedObj(cond);
    if (is_interpret) {
      return (has_true ? tb : fb);
    }
  }

  MS_LOG(EXCEPTION) << "Not support this condition value: " << cond->GetValueTrack()->ToString();
}

AbstractBasePtr InferImplSwitchLayer(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: {index, MakeTuple{branch1,branch2,branch3....}}
  constexpr auto kSwitchLayerInputNum = 2;
  const std::string op_name = primitive->name();
  abstract::CheckArgsSize(op_name, args_spec_list, kSwitchLayerInputNum);
  auto index = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto &input_shape = index->shape()->shape();
  if (!input_shape.empty() && (input_shape.size() != 1 || input_shape[0] != 1)) {
    MS_EXCEPTION(ValueError) << op_name << " index must be a 0 dimension tensor, but got a " << input_shape.size()
                             << " dimension tensor";
  }
  auto dtype = index->element()->BuildType();
  if (dtype->type_id() != kInt32->type_id()) {
    MS_EXCEPTION(ValueError) << op_name << " index must be an int32, but got " << dtype->ToString();
  }

  AbstractTuplePtr branches_abs = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
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
  // Return AbstractFuncUnion, otherwise the switch_layer will be replaced by branches[0]
  // which will cancel the out of bound checking for index
  if (branches.size() == 1) {
    AbstractFuncAtomPtrList func_list{b->cast<AbstractFuncAtomPtr>()};
    return std::make_shared<AbstractFuncUnion>(func_list);
  }
  for (size_t i = 1; i < branches.size(); i++) {
    b = b->Join(branches[i]);
  }
  return b;
}

std::vector<ValuePtr> GetSupportedTargetValue() {
  std::vector<ValuePtr> list = {kNone, MakeValue(false), MakeValue(true)};
  return list;
}

bool SupportedIsTargetValue(const ValuePtr t) {
  auto list = GetSupportedTargetValue();
  auto match = std::any_of(list.begin(), list.end(), [&t](const ValuePtr &v) { return *v == *t; });
  return match;
}

AbstractBasePtr InferImplIs_(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  // Statement: x is t
  // Inputs: x, t
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  ValuePtr t = args_spec_list[1]->BuildValue();
  if (!SupportedIsTargetValue(t)) {
    MS_LOG(EXCEPTION) << "For syntax like 'a is b', b supports True, False and None, but got " << t->ToString();
  }
  ValuePtr x = args_spec_list[0]->BuildValue();

  return std::make_shared<AbstractScalar>(*t == *x);
}

AbstractBasePtr InferImplIsNot(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  // Statement: x is not t
  // Inputs: x, t
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  ValuePtr t = args_spec_list[1]->BuildValue();
  if (!SupportedIsTargetValue(t)) {
    MS_LOG(EXCEPTION) << "For syntax like 'a is not b', b supports True, False and None, but got " << t->ToString();
  }
  ValuePtr x = args_spec_list[0]->BuildValue();

  return std::make_shared<AbstractScalar>(!(*t == *x));
}

bool IsInDict(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  const auto &key = args_spec_list[0];
  auto dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  MS_EXCEPTION_IF_NULL(key_value);
  std::vector<AbstractElementPair> dict_elems = dict->elements();
  auto it = std::find_if(dict_elems.cbegin(), dict_elems.cend(), [&key_value](const AbstractElementPair &item) {
    return *key_value == *item.first->BuildValue();
  });
  return it != dict_elems.end();
}

AbstractBasePtr InferImplInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // Statement: x in t
  // Inputs: x, t
  return std::make_shared<AbstractScalar>(IsInDict(primitive, args_spec_list));
}

AbstractBasePtr InferImplNotInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Statement: x not in t
  // Inputs: x, t
  return std::make_shared<AbstractScalar>(!IsInDict(primitive, args_spec_list));
}

AbstractBasePtr InferImplIsConstant(const AnalysisEnginePtr &, const PrimitivePtr &,
                                    const AbstractBasePtrList &args_spec_list) {
  // Statement: isconstant(x)
  // Inputs: x
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "IsConstant requires args input size = 1";
  }
  ValuePtr v = args_spec_list[0]->BuildValue();
  return std::make_shared<AbstractScalar>(!v->isa<AnyValue>());
}
}  // namespace abstract
}  // namespace mindspore
