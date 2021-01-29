/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "abstract/infer_functions.h"
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

AbstractBasePtr InferImplSwitch(const AnalysisEnginePtr &, const PrimitivePtr &prim,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: condition, true branch, false branch
  if (args_spec_list.size() != 3) {
    MS_LOG(EXCEPTION) << "Switch evaluator requires 3 parameters, while the input size is " << args_spec_list.size()
                      << ".";
  }

  auto cond = args_spec_list[0];
  auto tb = args_spec_list[1];
  auto fb = args_spec_list[2];
  MS_EXCEPTION_IF_NULL(cond);

  ValuePtr v = cond->GetValueTrack();
  MS_EXCEPTION_IF_NULL(v);
  // for tensor as condition, keeps both true and false branch.
  if (v->isa<AnyValue>() || cond->isa<AbstractTensor>()) {
    MS_EXCEPTION_IF_NULL(tb);
    return tb->Join(fb);
  }

  if (v->isa<Scalar>()) {
    if (v->cast<ScalarPtr>()->IsOne()) {
      return tb;
    } else {
      return fb;
    }
  }

  MS_LOG(EXCEPTION) << "Invalid condition value for switch " << cond->ToString();
}

AbstractBasePtr InferImplSwitchLayer(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: index, branch
  const std::string op_name = primitive->name();
  abstract::CheckArgsSize(op_name, args_spec_list, 2);
  auto index = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto &input_shape = index->shape()->shape();
  if (input_shape.size() != 0) {
    MS_EXCEPTION(ValueError) << op_name << " index must be a 0 dimension tensor, but got a " << input_shape.size()
                             << " dimension tensor";
  }
  auto dtype = index->element()->BuildType();
  if (dtype->type_id() != kInt32->type_id()) {
    MS_EXCEPTION(ValueError) << op_name << " index must be a int32, but got " << dtype->ToString();
  }

  AbstractTuplePtr branches_abs = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  AbstractBasePtrList branches = branches_abs->elements();
  const size_t maximum_layer_num = 1000;
  if (branches.size() < 1 || branches.size() > maximum_layer_num) {
    MS_EXCEPTION(ValueError) << op_name << " support at least 1 and at most " << maximum_layer_num << " but got "
                             << branches.size() << " branches.";
  }

  for (size_t i = 0; i < branches.size(); i++) {
    MS_EXCEPTION_IF_NULL(branches[i]);
    if (!branches[i]->isa<FuncGraphAbstractClosure>()) {
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
  // statement: x is t
  // Inputs: x, t
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  ValuePtr t = args_spec_list[1]->BuildValue();
  if (!SupportedIsTargetValue(t)) {
    MS_LOG(EXCEPTION) << "Not supported type:" << t->ToString()
                      << " for statement is, supported list is:None, False, True ";
  }
  ValuePtr x = args_spec_list[0]->BuildValue();

  return std::make_shared<AbstractScalar>(*t == *x);
}

AbstractBasePtr InferImplIsNot(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  // statement: x is not t
  // Inputs: x, t
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  ValuePtr t = args_spec_list[1]->BuildValue();
  if (!SupportedIsTargetValue(t)) {
    MS_LOG(EXCEPTION) << "Not supported type:" << t->ToString()
                      << " for statement is not, supported list is:None, False, True ";
  }
  ValuePtr x = args_spec_list[0]->BuildValue();

  return std::make_shared<AbstractScalar>(!(*t == *x));
}

bool IsInDict(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto key = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  auto dict = CheckArg<AbstractDictionary>(op_name, args_spec_list, 1);

  ValuePtr key_value = key->BuildValue();
  if (!key_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " evaluator key should be string, but got " << key_value->ToString();
  }
  auto key_str = GetValue<std::string>(key_value);
  std::vector<AbstractAttribute> dict_elems = dict->elements();
  auto it = std::find_if(dict_elems.begin(), dict_elems.end(),
                         [key_str](const AbstractAttribute &item) { return item.first == key_str; });
  return it != dict_elems.end();
}

AbstractBasePtr InferImplInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // statement: x in t
  // Inputs: x, t
  return std::make_shared<AbstractScalar>(IsInDict(primitive, args_spec_list));
}

AbstractBasePtr InferImplNotInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // statement: x not in t
  // Inputs: x, t
  return std::make_shared<AbstractScalar>(!IsInDict(primitive, args_spec_list));
}

AbstractBasePtr InferImplIsConstant(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // statement: isconstant(x)
  // Inputs: x
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "IsConstant requires args input size = 1";
  }
  ValuePtr v = args_spec_list[0]->BuildValue();
  return std::make_shared<AbstractScalar>(!v->isa<AnyValue>());
}
}  // namespace abstract
}  // namespace mindspore
