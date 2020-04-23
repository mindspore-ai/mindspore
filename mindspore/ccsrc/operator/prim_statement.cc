/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/static_analysis/param_validator.h"
#include "pipeline/static_analysis/prim.h"
#include "operator/ops.h"
#include "pipeline/static_analysis/utils.h"
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

AbstractBasePtr InferImplTypeof(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: a pointer to an AbstractBase object
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "Typeof evaluator requires 1 parameter, while the input size is " << args_spec_list.size()
                      << ".";
  }
  AbstractBasePtr abs_base = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(abs_base);
  TypePtr type = abs_base->BuildType();
  return std::make_shared<AbstractType>(type);
}

AbstractBasePtr InferImplHasType(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  // Inputs: a pointer to an AbstractBase object and a pointer to a Type
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTypePtr abs_type = CheckArg<AbstractType>(op_name, args_spec_list, 1);

  auto mode_v = abs_type->GetValueTrack();
  MS_EXCEPTION_IF_NULL(mode_v);
  if (!mode_v->isa<Type>()) {
    MS_LOG(EXCEPTION) << "Get the type from AbstractType value failed.";
  }

  TypePtr mode_t = mode_v->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  bool v = IsSubtype(args_spec_list[0], mode_t);
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(v), kBool);
}

AbstractBasePtr InferImplDot(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTensorPtr input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  AbstractTensorPtr input_y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);

  ShapePtr x_shp = input_x->shape();
  auto x_shp_value = x_shp->shape();
  ShapePtr y_shp = input_y->shape();
  auto y_shp_value = y_shp->shape();
  // Should be matrix which shape size is 2.
  if (x_shp_value.size() != 2 || y_shp_value.size() != 2) {
    MS_LOG(EXCEPTION) << "" << op_name
                      << " evaluator requires input two 2D tensors, while the dimensions of two tensors are "
                      << x_shp_value.size() << ", " << y_shp_value.size() << " ";
  }
  if (x_shp_value[1] != y_shp_value[0] && x_shp_value[1] != Shape::SHP_ANY && y_shp_value[0] != Shape::SHP_ANY) {
    MS_LOG(EXCEPTION) << "Incompatible shapes in dot: {" << x_shp->ToString() << "} and {" << y_shp->ToString() << "}";
  }

  auto x_element = input_x->element();
  MS_EXCEPTION_IF_NULL(x_element);
  (void)x_element->Join(input_y->element());
  auto param = {x_shp_value[0], y_shp_value[1]};

  return std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(param));
}

AbstractBasePtr InferImplSwitch(const AnalysisEnginePtr &, const PrimitivePtr &,
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
  if (v->isa<AnyValue>()) {
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
}  // namespace abstract
}  // namespace mindspore
