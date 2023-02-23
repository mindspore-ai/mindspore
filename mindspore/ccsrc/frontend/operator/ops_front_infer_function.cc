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
#include "frontend/operator/ops_front_infer_function.h"

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>

#include "abstract/abstract_value.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "abstract/param_validator.h"
#include "pybind_api/ir/tensor_py.h"
#include "frontend/operator/ops.h"
#include "abstract/ops/infer_functions.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/utils.h"
#include "ops/exp.h"
#include "ops/log.h"
#include "ops/reciprocal.h"
#include "ops/real_div.h"
#include "ops/add.h"
#include "ops/arg_min.h"
#include "ops/equal.h"
#include "ops/greater_equal.h"
#include "ops/greater.h"
#include "ops/not_equal.h"
#include "ops/neg.h"
#include "ops/mul.h"
#include "ops/mod.h"
#include "ops/sub.h"
#include "ops/strided_slice.h"
#include "ops/strided_slice_v2.h"
#include "ops/grad/strided_slice_v2_grad.h"
#include "abstract/abstract_function.h"
#include "utils/ms_context.h"
#ifdef _MSC_VER
#include "include/common/pybind_api/api_register.h"
#endif

namespace mindspore {
namespace abstract {
enum class State {
  SAME,
  X_ONE,
  Y_ONE,
};

struct SlideInfo {
  int64_t start;
  int64_t step;
  int64_t stop;
};

void ComputeReduceIndex(const std::vector<int64_t> &reverse_x, const std::vector<int64_t> &reverse_y,
                        std::vector<int64_t> *grad_x_reduce_idx, std::vector<int64_t> *grad_y_reduce_idy) {
  MS_EXCEPTION_IF_NULL(grad_x_reduce_idx);
  MS_EXCEPTION_IF_NULL(grad_y_reduce_idy);
  const size_t n = reverse_x.size();
  if (reverse_y.size() < n) {
    MS_LOG(EXCEPTION) << "The size of reverse_y is less than the size of reverse_x.";
  }
  for (size_t i = 0; i < n; ++i) {
    State curr;
    const int64_t x_i = reverse_x[i];
    const int64_t y_i = reverse_y[i];
    const int64_t reduce_idx = SizeToLong(n - 1 - i);
    if (x_i == y_i) {
      curr = State::SAME;
    } else if (x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
      curr = State::X_ONE;
    } else if (y_i == 1) {
      grad_y_reduce_idy->push_back(reduce_idx);
      curr = State::Y_ONE;
    } else {
      MS_LOG(EXCEPTION) << "not compatible shape input for BroadcastGradientArgs.";
    }
    if (curr == State::SAME && x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
      grad_y_reduce_idy->push_back(reduce_idx);
      continue;
    }
  }

  std::reverse(grad_x_reduce_idx->begin(), grad_x_reduce_idx->end());
  std::reverse(grad_y_reduce_idy->begin(), grad_y_reduce_idy->end());
}

AbstractBasePtr BroadcastGradientArgsDiff(const std::vector<ValuePtr> &x_shape, const std::vector<ValuePtr> &y_shape) {
  std::vector<int64_t> reverse_x;
  std::vector<int64_t> reverse_y;

  (void)std::transform(x_shape.rbegin(), x_shape.rend(), std::back_inserter(reverse_x),
                       [](const ValuePtr &v) { return v->cast<Int64ImmPtr>()->value(); });
  (void)std::transform(y_shape.rbegin(), y_shape.rend(), std::back_inserter(reverse_y),
                       [](const ValuePtr &v) { return v->cast<Int64ImmPtr>()->value(); });

  if (reverse_x.size() > reverse_y.size()) {
    reverse_y.resize(reverse_x.size(), 1);
  } else {
    reverse_x.resize(reverse_y.size(), 1);
  }

  std::vector<int64_t> grad_x_reduce_idx;
  std::vector<int64_t> grad_y_reduce_idy;
  ComputeReduceIndex(reverse_x, reverse_y, &grad_x_reduce_idx, &grad_y_reduce_idy);

  AbstractBasePtrList abs_list_x;
  AbstractBasePtrList abs_list_y;
  (void)std::transform(grad_x_reduce_idx.begin(), grad_x_reduce_idx.end(), std::back_inserter(abs_list_x),
                       [](int64_t v) { return abstract::FromValue(v); });
  (void)std::transform(grad_y_reduce_idy.begin(), grad_y_reduce_idy.end(), std::back_inserter(abs_list_y),
                       [](int64_t v) { return abstract::FromValue(v); });
  auto x_reduce_idx = std::make_shared<AbstractTuple>(abs_list_x);
  auto y_reduce_idx = std::make_shared<AbstractTuple>(abs_list_y);
  AbstractBasePtrList elem_list;
  elem_list.push_back(x_reduce_idx);
  elem_list.push_back(y_reduce_idx);

  return std::make_shared<AbstractTuple>(elem_list);
}

AbstractBasePtr InferImplTypeof(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: a pointer to an AbstractBase object
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "The Typeof operator must requires 1 argument, but the size of arguments is "
                      << args_spec_list.size() << ".";
  }
  AbstractBasePtr abs_base = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(abs_base);
  TypePtr type = abs_base->BuildType();
  return std::make_shared<AbstractType>(type);
}

AbstractBasePtr InferImplTopTypeof(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a pointer to an AbstractBase object
  if (args_spec_list.size() != 1) {
    MS_LOG(EXCEPTION) << "The Typeof operator must requires 1 argument, but the size of arguments is "
                      << args_spec_list.size() << ".";
  }
  AbstractBasePtr abs_base = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(abs_base);
  TypeId type_id = abs_base->BuildType()->type_id();
  return std::make_shared<AbstractType>(TypeIdToType(type_id));
}

AbstractBasePtr InferImplLower(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "The lower must has one input at least.";
  }
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  auto input = args_spec_list[0]->BuildValue();
  if (input == nullptr || !input->isa<StringImm>()) {
    auto type = args_spec_list[0]->BuildType();
    MS_EXCEPTION(TypeError) << "Function lower should be call using a string type but got:" << type->ToString();
  }
  auto str = input->cast<StringImmPtr>()->value();
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  auto new_str = MakeValue(str);
  return new_str->ToAbstract();
}

AbstractBasePtr InferImplHasType(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Inputs: a pointer to an AbstractBase object and a pointer to a Type
  const std::string op_name = primitive->name();
  const size_t args_num = 2;
  CheckArgsSize(op_name, args_spec_list, args_num);
  AbstractTypePtr abs_type = CheckArg<AbstractType>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(abs_type);
  auto mode_v = abs_type->GetValueTrack();
  MS_EXCEPTION_IF_NULL(mode_v);
  if (!mode_v->isa<Type>()) {
    MS_LOG(EXCEPTION) << "Get the type from AbstractType value failed.";
  }

  auto tmpMode = mode_v->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  bool v = IsSubtype(args_spec_list[0], tmpMode);
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(v), kBool);
}

bool IsAdapterTensor(const AbstractBasePtr &x) {
  if (!x->isa<abstract::AbstractTensor>()) {
    return false;
  }
  return x->cast<abstract::AbstractTensorPtr>()->is_adapter();
}

bool IsAdapterTensorClassType(const AbstractBasePtr &cmp) {
  auto cmp_value = cmp->BuildValue();
  if (!cmp_value->isa<parse::ClassType>()) {
    return false;
  }
  auto class_obj = cmp_value->cast<parse::ClassTypePtr>()->obj();
  return py::hasattr(class_obj, PYTHON_ADAPTER_TENSOR);
}

bool CheckPythonIsInstance(const py::object &x, const AbstractBasePtr &cmp, const py::module &mod, bool is_const) {
  if (cmp->isa<abstract::AbstractTuple>()) {
    const auto &cmp_tuple_elements = cmp->cast<abstract::AbstractTuplePtr>()->elements();
    return std::any_of(cmp_tuple_elements.begin(), cmp_tuple_elements.end(),
                       [&x, &mod, is_const](const AbstractBasePtr &element) {
                         return CheckPythonIsInstance(x, element, mod, is_const);
                       });
  }
  if (std::find(kSparsePrimStr.begin(), kSparsePrimStr.end(), cmp->ToString()) != kSparsePrimStr.end()) {
    return false;
  }

  py::object cmp_type;
  if (cmp->isa<abstract::PartialAbstractClosure>()) {
    const auto &cmp_closure_args = cmp->cast<abstract::PartialAbstractClosurePtr>()->args();
    // CheckCmpValid ensures size of cmp_closure_args to be 1.
    auto cmp_closure_first_input = cmp_closure_args[0];
    cmp_type = ValueToPyData(cmp_closure_first_input->BuildValue());
  } else {
    auto cmp_value = cmp->BuildValue();
    if (cmp_value == kAnyValue) {
      return false;
    }
    cmp_type = ValueToPyData(cmp_value);
  }

  py::object result = is_const ? python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_PYTHON_ISINSTANCE, x, cmp_type)
                               : python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_MS_ISINSTANCE, x, cmp_type);
  return result.cast<bool>();
}

bool CheckIsInstanceForFunc(const py::object &x_py_obj, const AbstractBasePtr &cmp, const py::module &mod) {
  MS_EXCEPTION_IF_NULL(cmp);
  if (cmp->isa<abstract::AbstractTuple>()) {
    const auto &cmp_tuple_elements = cmp->cast<abstract::AbstractTuplePtr>()->elements();
    return std::any_of(
      cmp_tuple_elements.begin(), cmp_tuple_elements.end(),
      [&x_py_obj, &mod](const AbstractBasePtr &element) { return CheckIsInstanceForFunc(x_py_obj, element, mod); });
  }

  if (!cmp->isa<abstract::PartialAbstractClosure>()) {
    return false;
  }
  const auto &cmp_closure_args = cmp->cast<abstract::PartialAbstractClosurePtr>()->args();
  // CheckCmpValid ensures size of cmp_closure_args to be 1.
  auto cmp_closure_first_input = cmp_closure_args[0];
  auto cmp_py_obj = ValueToPyData(cmp_closure_first_input->BuildValue());
  auto result = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_PYTHON_ISINSTANCE, x_py_obj, cmp_py_obj);
  return result.cast<bool>();
}

bool CheckIsInstanceForSparse(const AbstractBasePtr &cmp, const std::string &target) {
  MS_EXCEPTION_IF_NULL(cmp);
  if (!cmp->isa<abstract::AbstractTuple>()) {
    return cmp->ToString() == target;
  }
  const auto &cmp_tuple_elements = cmp->cast<abstract::AbstractTuplePtr>()->elements();
  return std::any_of(cmp_tuple_elements.begin(), cmp_tuple_elements.end(),
                     [&target](const AbstractBasePtr &element) { return CheckIsInstanceForSparse(element, target); });
}

py::object GetPrimitivePyObj(const abstract::PrimitiveAbstractClosurePtr &prim_abs) {
  MS_EXCEPTION_IF_NULL(prim_abs);
  auto prim = prim_abs->prim();
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_signature = prim->cast<prim::DoSignaturePrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim_signature);
  auto function = prim_signature->function();
  MS_EXCEPTION_IF_NULL(function);
  auto primitive_py_function = function->cast<PrimitivePyPtr>();
  return primitive_py_function->GetPyObj();
}

py::object GetMsClassPyObj(const abstract::PartialAbstractClosurePtr &ms_class_abs) {
  MS_EXCEPTION_IF_NULL(ms_class_abs);
  const auto &ms_class_args = ms_class_abs->args();
  if (ms_class_args.size() != 1) {
    MS_LOG(EXCEPTION) << "When the first input to IsInstance is PartialAbstractClosure, its args size should be 1 but "
                      << "got: " << ms_class_args.size() << ".";
  }
  auto first_arg = ms_class_args[0];
  auto class_value = first_arg->BuildValue();
  MS_EXCEPTION_IF_NULL(class_value);
  return ValueToPyData(class_value);
}

bool CheckCmpValid(const AbstractBasePtr &cmp) {
  MS_EXCEPTION_IF_NULL(cmp);
  if (cmp->isa<abstract::AbstractSequence>()) {
    if (!cmp->isa<abstract::AbstractTuple>()) {
      return false;
    }
    const auto &elements = cmp->cast<abstract::AbstractTuplePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(),
                       [](const AbstractBasePtr &element) { return CheckCmpValid(element); });
  }
  if (cmp->isa<abstract::AbstractScalar>()) {
    auto cmp_type = cmp->BuildType();
    MS_EXCEPTION_IF_NULL(cmp_type);
    return cmp_type->type_id() == kMetaTypeTypeType;
  } else if (cmp->isa<abstract::PartialAbstractClosure>()) {
    auto cmp_closure = cmp->cast<abstract::PartialAbstractClosurePtr>();
    const auto &cmp_closure_args = cmp_closure->args();
    if (cmp_closure_args.size() != 1) {
      return false;
    }
    auto cmp_closure_first_input = cmp_closure_args[0];
    auto cmp_type = cmp_closure_first_input->BuildType();
    MS_EXCEPTION_IF_NULL(cmp_type);
    auto cmp_type_id = cmp_type->type_id();
    if (cmp_type_id == kObjectTypeClass) {
      // When cmp type is ms_class, fn should be create_instance.
      auto cmp_closure_fn = cmp_closure->fn();
      MS_EXCEPTION_IF_NULL(cmp_closure_fn);
      const std::string ms_class_type_fn_name = "Prim: create_instance";
      return cmp_closure_fn->ToString() == ms_class_type_fn_name;
    }
    return cmp_type_id == kMetaTypeTypeType;
  }
  return std::find(kSparsePrimStr.cbegin(), kSparsePrimStr.cend(), cmp->ToString()) != kSparsePrimStr.cend();
}

AbstractBasePtr InferImplIsInstance(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr size_t args_num = 2;
  CheckArgsSize(primitive->name(), args_spec_list, args_num);
  py::gil_scoped_acquire gil;
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  auto x = args_spec_list[0];
  auto cmp = args_spec_list[1];
  bool result = false;

  if (!CheckCmpValid(cmp)) {
    auto cmp_type = cmp->BuildType();
    MS_EXCEPTION_IF_NULL(cmp_type);
    MS_LOG(ERROR) << "cmp: " << cmp->ToString() << ", cmp_type: " << cmp_type->ToString()
                  << ", cmp_type_id: " << TypeIdToType(cmp_type->type_id());
    MS_EXCEPTION(TypeError) << "isinstance() arg 2 must be a type or tuple of types.";
  }

  // x is Cell object.
  MS_EXCEPTION_IF_NULL(x);
  if (x->isa<abstract::FuncGraphAbstractClosure>()) {
    auto x_fg = x->cast<abstract::FuncGraphAbstractClosurePtr>()->func_graph();
    MS_EXCEPTION_IF_NULL(x_fg);
    auto wrapper_obj = x_fg->python_obj();
    if (wrapper_obj != nullptr) {
      if (!wrapper_obj->isa<parse::PyObjectWrapper>()) {
        MS_LOG(EXCEPTION) << "The wrapper_obj of FuncGraphAbstractClosure must be PyObjectWrapper but got: "
                          << wrapper_obj->ToString() << ".";
      }
      auto x_py_obj = wrapper_obj->cast<parse::PyObjectWrapperPtr>()->obj();
      result = CheckIsInstanceForFunc(x_py_obj, cmp, mod);
    }
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
  }

  // x is Primitive.
  if (x->isa<abstract::PrimitiveAbstractClosure>()) {
    auto x_py_obj = GetPrimitivePyObj(x->cast<abstract::PrimitiveAbstractClosurePtr>());
    result = CheckIsInstanceForFunc(x_py_obj, cmp, mod);
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
  }

  // x is ms_class.
  if (x->isa<abstract::PartialAbstractClosure>()) {
    auto x_py = GetMsClassPyObj(x->cast<abstract::PartialAbstractClosurePtr>());
    result = CheckIsInstanceForFunc(x_py, cmp, mod);
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
  }

  // x is sparse tensor, now support RowTensor, CSRTensor, COOTensor.
  if (x->isa<abstract::AbstractCSRTensor>()) {
    const size_t csr_index = 0;
    result = CheckIsInstanceForSparse(cmp, kSparsePrimStr[csr_index]);
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
  } else if (x->isa<abstract::AbstractCOOTensor>()) {
    const size_t coo_index = 1;
    result = CheckIsInstanceForSparse(cmp, kSparsePrimStr[coo_index]);
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
  } else if (x->isa<abstract::AbstractRowTensor>()) {
    const size_t row_index = 2;
    result = CheckIsInstanceForSparse(cmp, kSparsePrimStr[row_index]);
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
  }

  // x is adapter tensor.
  if (IsAdapterTensor(x) && IsAdapterTensorClassType(cmp)) {
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(true), kBool);
  }

  auto x_value = x->BuildValue();
  // x is variable built-in type.
  if (x_value == kAnyValue) {
    auto x_abs_type = std::make_shared<AbstractType>(x->BuildType());
    auto py_x_type = ValueToPyData(x_abs_type->BuildValue());
    result = CheckPythonIsInstance(py_x_type, cmp, mod, false);
    return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
  }

  // x is python built-in constant type or external type.
  py::object x_py_obj = ValueToPyData(x_value);
  result = CheckPythonIsInstance(x_py_obj, cmp, mod, true);
  return std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(result), kBool);
}

bool CompareShape(const std::vector<ValuePtr> &x_shape, const std::vector<ValuePtr> &y_shape) {
  if (x_shape.size() != y_shape.size()) {
    return false;
  }

  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (GetValue<int64_t>(x_shape[i]) != GetValue<int64_t>(y_shape[i])) {
      return false;
    }
  }

  return true;
}

AbstractBasePtr DoInferReduceShape(const AbstractTuplePtr &x_shape, const ValuePtr &x_shp_value,
                                   const ValueSequencePtr &axis_value_ptr, const PrimitivePtr &primitive) {
  size_t x_rank = x_shape->size();
  std::set<int64_t> axis_set;
  auto axis_data = axis_value_ptr->value();
  if (axis_data.empty()) {
    int64_t size = 1;
    AbstractBasePtrList values(x_rank, std::make_shared<AbstractScalar>(size));
    return std::make_shared<AbstractTuple>(values);
  }

  for (auto &elem : axis_data) {
    auto x_rank_tmp = x_rank;
    if (x_rank_tmp == 0) {
      x_rank_tmp = 1;
    }
    int64_t e_value =
      CheckAxis(primitive->name(), "axis", elem, -SizeToLong(x_rank_tmp), SizeToLong(x_rank_tmp), "input_x");
    (void)axis_set.insert(e_value);
  }
  MS_EXCEPTION_IF_NULL(x_shp_value->cast<ValueTuplePtr>());
  auto x_shp_data = x_shp_value->cast<ValueTuplePtr>()->value();
  if (x_shp_data.size() < x_rank) {
    MS_LOG(EXCEPTION) << "x_shape_data.size() " << x_shp_data.size() << " less than x_shape.size() " << x_rank << ".";
  }
  AbstractBasePtrList values;
  for (size_t i = 0; i < x_rank; i++) {
    if (axis_set.count(SizeToLong(i)) || axis_set.count(SizeToLong(i) - SizeToLong(x_rank))) {
      auto axis_v = MakeValue(static_cast<int64_t>(1));
      values.push_back(std::make_shared<AbstractScalar>(axis_v, axis_v->type()));
    } else {
      int64_t dim_value = x_shp_data[i]->cast<Int64ImmPtr>()->value();
      auto dim = MakeValue(dim_value);
      values.push_back(std::make_shared<AbstractScalar>(dim, dim->type()));
    }
  }

  return std::make_shared<AbstractTuple>(values);
}

AbstractBasePtr InferImplBroadcastGradientArgs(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list) {
  // this primitive get the index that need to reduce
  // input: x's shape and y's shape, inputs should be tuple
  // output: tuple of x and y 's reduce index, reduce index should be a tuple
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 2;
  CheckArgsSize(op_name, args_spec_list, inputs_size);
  auto arg_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  auto arg_y = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);

  auto arg_x_value = arg_x->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(arg_x_value);

  auto arg_y_value = arg_y->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(arg_y_value);

  const std::vector<ValuePtr> x_shape = arg_x_value->value();
  const std::vector<ValuePtr> y_shape = arg_y_value->value();
  bool is_same_shape = CompareShape(x_shape, y_shape);
  // if it is the same shape , do not need reduce , return empty tuple
  if (is_same_shape) {
    AbstractBasePtrList empty_list;
    auto x_reduce_idx = std::make_shared<AbstractTuple>(empty_list);
    auto y_reduce_idx = std::make_shared<AbstractTuple>(empty_list);

    AbstractBasePtrList elem_list;
    elem_list.push_back(x_reduce_idx);
    elem_list.push_back(y_reduce_idx);

    return std::make_shared<AbstractTuple>(elem_list);
  }
  return BroadcastGradientArgsDiff(x_shape, y_shape);
}

AbstractBasePtr InferImplListReduce(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: a fn, a list and an object of a subclass of a AbstractBase.
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 3;
  CheckArgsSize(op_name, args_spec_list, inputs_size);
  AbstractFunctionPtr fn = CheckArg<AbstractFunction>(op_name, args_spec_list, 0);
  AbstractListPtr lst = CheckArg<AbstractList>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(lst);
  AbstractBasePtr dflt = args_spec_list[2];

  AbstractBasePtr list_type = AbstractJoin(lst->elements());
  auto result1 = engine->Execute(fn, lst->elements());
  MS_EXCEPTION_IF_NULL(result1);
  auto result2 = engine->Execute(fn, {dflt, list_type});
  MS_EXCEPTION_IF_NULL(result2);
  MS_EXCEPTION_IF_NULL(result1->abstract());
  MS_EXCEPTION_IF_NULL(result2->abstract());
  return result1->abstract()->Join(result2->abstract());
}

AbstractBasePtr InferImplTupleReversed(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr input = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input);
  auto tuple_elements = input->elements();
  AbstractBasePtrList elem_list;
  (void)std::transform(tuple_elements.rbegin(), tuple_elements.rend(), std::back_inserter(elem_list),
                       [](const AbstractBasePtr &elem) { return elem->Clone(); });
  return std::make_shared<AbstractTuple>(elem_list);
}

AbstractBasePtr InferImplReduceShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: x_shape, axis
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t arg_size = 2;
  CheckArgsSize(op_name, args_spec_list, arg_size);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(shape_x);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);

  auto x_shp_value = shape_x->BuildValue();
  if (x_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The ReduceShape operator's data field can't be anything: " << args_spec_list[1]->ToString()
                      << ".";
  }

  // Axis can be scalar, tuple or list
  AbstractSequencePtr axis = nullptr;
  if (args_spec_list[1]->isa<AbstractScalar>()) {
    MS_LOG(DEBUG) << op_name << " evaluator second parameter is scalar.";
    AbstractBasePtrList axis_list = {dyn_cast<AbstractScalar>(args_spec_list[1])};
    axis = std::make_shared<AbstractTuple>(axis_list);
  } else if (args_spec_list[1]->isa<AbstractSequence>()) {
    MS_LOG(DEBUG) << "The type of second argument of ReduceShape operator is sequence.";
    axis = args_spec_list[1]->cast<AbstractSequencePtr>();
  } else {
    MS_LOG(EXCEPTION) << "The second argument of ReduceShape operator should be a scalar or tuple or list, "
                      << "but got " << args_spec_list[1]->ToString() << ".";
  }

  auto axis_value = axis->BuildValue();
  if (axis_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The ReduceShape operator's data field can't be anything: " << args_spec_list[1]->ToString()
                      << ".";
  }
  auto axis_value_ptr = axis_value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(axis_value_ptr);
  return DoInferReduceShape(shape_x, x_shp_value, axis_value_ptr, primitive);
}

AbstractBasePtr InferImplTupleDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t arg_size = 2;
  CheckArgsSize(op_name, args_spec_list, arg_size);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractTuplePtr div_shp = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(shape_x);
  MS_EXCEPTION_IF_NULL(div_shp);
  MS_LOG(INFO) << "The shape of dividend:" << shape_x->ToString() << ", the shape of divisor:" << div_shp->ToString();

  auto div_shp_value = div_shp->BuildValue();
  MS_EXCEPTION_IF_NULL(div_shp_value);
  if (div_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The 'tuple_div' operator shape's data field can't be anything, but got "
                      << args_spec_list[0]->ToString() << ".";
  }

  auto shape_x_value = shape_x->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_x_value);
  if (shape_x_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The 'tuple_div' operator shape's data field can't be anything, but got "
                      << args_spec_list[1]->ToString() << ".";
  }

  if (div_shp->size() != shape_x->size()) {
    MS_LOG(EXCEPTION) << "The size of inputs of 'tuple_div' operator must be the same, but the size of divisor tuple is"
                      << " " << div_shp->size() << ", the size of dividend tuple is " << shape_x->size() << ".";
  }
  auto shape_x_tuple_value = shape_x_value->cast<ValueTuplePtr>();
  auto div_shape_tuple_value = div_shp_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape_x_tuple_value);
  MS_EXCEPTION_IF_NULL(div_shape_tuple_value);
  auto shape_x_data = shape_x_tuple_value->value();
  auto div_shape_data = div_shape_tuple_value->value();
  AbstractBasePtrList values;

  for (size_t i = 0; i < div_shape_data.size(); i++) {
    MS_EXCEPTION_IF_NULL(div_shape_data[i]);
    if (div_shape_data[i]->cast<Int64ImmPtr>() == nullptr) {
      auto value_type = div_shape_data[i]->type();
      std::string str_type;
      if (value_type) {
        str_type = value_type->ToString();
      } else {
        str_type = "AnyValue";
      }
      MS_LOG(EXCEPTION) << "The data type of inputs of 'tuple_div' operator should be an int64 number, but got a "
                        << str_type << " number " << div_shape_data[i]->ToString() << ".";
    }
    auto shapex_value = GetValue<int64_t>(shape_x_data[i]);
    auto div_value = GetValue<int64_t>(div_shape_data[i]);
    MS_LOG(DEBUG) << "div_shp_shape data shapex_value :" << shapex_value << " div_value: " << div_value;
    if (div_value == 0) {
      MS_LOG(EXCEPTION) << "The divisor value should not be 0!";
    }
    if ((shapex_value % div_value) != 0) {
      MS_LOG(EXCEPTION) << "The inputs of 'tuple_div' operator should be divisible, but they are not divisible now, "
                        << "the dividend is " << shapex_value << ", the divisor is " << div_value << ".";
    }

    int64_t result = shapex_value / div_value;
    auto result_v = MakeValue(result);
    values.push_back(std::make_shared<AbstractScalar>(result_v, result_v->type()));
  }
  return std::make_shared<AbstractTuple>(values);
}

AbstractBasePtr InferImplTuple2Array(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr input = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input);
  py::tuple data_tuple = ValueToPyData(input->BuildValue());
  py::array data = py::array(data_tuple);
  auto tensor = tensor::TensorPy::MakeTensor(data);
  auto ret = tensor->ToAbstract();
  ret->set_value(tensor);
  MS_LOG(DEBUG) << "The infer result of Tuple2Array operator is tensor, the infer result is " << ret->ToString() << ".";
  return ret;
}

AbstractBasePtr InferImplShapeMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  // example: tuple = (1, 2, 3), shape_mul(tuple) = 1*2*3 = 6
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);

  auto shpx_value = shape_x->BuildValue();
  if (shpx_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The ShapeMul operator shape's data field can't be anything, but got " << shape_x->ToString()
                      << ".";
  }

  auto shpx_data = shpx_value->cast<ValueTuplePtr>()->value();

  int64_t result = 1;
  for (size_t i = 0; i < shpx_data.size(); i++) {
    int64_t value = GetValue<int64_t>(shpx_data[i]);
    result = IntMulWithOverflowCheck(result, value);
  }

  auto result_v = MakeValue(result);
  MS_LOG(DEBUG) << "The infer result of ShapeMul is " << result_v->ToString();
  return std::make_shared<AbstractScalar>(result_v, result_v->type());
}

AbstractBasePtr InferImplSliceGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  auto op_name = primitive->name();
  constexpr auto slice_getitem_input_size = 2;
  CheckArgsSize(op_name, args_spec_list, slice_getitem_input_size);
  AbstractSlicePtr slice_abs = CheckArg<AbstractSlice>(op_name, args_spec_list, 0);
  const std::map<std::string, AbstractBasePtr> result_map = {
    {kSliceStart, slice_abs->start()}, {kSliceStop, slice_abs->stop()}, {kSliceStep, slice_abs->step()}};
  auto slice_attr = args_spec_list[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(slice_attr);
  if (!slice_attr->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The second argument of SliceGetItem operator should be a string, but got "
                      << slice_attr->ToString() << ".";
  }
  auto slice_str = GetValue<std::string>(slice_attr);
  auto iter = result_map.find(slice_str);
  if (iter == result_map.end()) {
    MS_EXCEPTION(AttributeError) << "The 'slice' object has no attribute:" << slice_str << ".";
  }
  return iter->second;
}

AbstractBasePtr InferImplMakeSlice(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: three scalars whose value is an int32 number.
  constexpr auto make_slice_input_size = 3;
  CheckArgsSize(primitive->name(), args_spec_list, make_slice_input_size);
  size_t args_size = args_spec_list.size();
  AbstractBasePtrList slice_args;
  for (size_t index = 0; index < args_size; index++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[index]);
    if (args_spec_list[index]->isa<AbstractNone>()) {
      slice_args.push_back(args_spec_list[index]);
    } else if (args_spec_list[index]->isa<AbstractScalar>()) {
      ValuePtr scalar_value = args_spec_list[index]->cast<AbstractScalarPtr>()->BuildValue();
      MS_EXCEPTION_IF_NULL(scalar_value);
      if (scalar_value->isa<IntegerImm>() || scalar_value == kAnyValue) {
        slice_args.push_back(args_spec_list[index]);
      } else if (scalar_value->isa<BoolImm>()) {
        ValuePtr scalar_index = MakeValue(static_cast<int64_t>(scalar_value->cast<BoolImmPtr>()->value()));
        slice_args.push_back(scalar_index->ToAbstract());
      } else {
        auto type = scalar_value->type();
        MS_EXCEPTION_IF_NULL(type);
        MS_EXCEPTION(TypeError) << "Slice indices must be integers or bool. But got a " << type->ToString()
                                << " number.";
      }
    } else if (args_spec_list[index]->isa<AbstractTensor>()) {
      auto arg = args_spec_list[index]->cast<AbstractTensorPtr>();
      TypePtr tensor_dtype = arg->element()->BuildType();
      auto build_value = arg->BuildValue();
      MS_EXCEPTION_IF_NULL(build_value);
      auto value = build_value->cast<tensor::TensorPtr>();
      if (value != nullptr) {
        if (value->DataSize() != 1) {
          MS_EXCEPTION(TypeError) << "The input tensor of the MakeSlice operator must contain only one element,"
                                  << "but " << value->ToString() << " has " << value->DataSize() << " elements.";
        }

        if (tensor_dtype->isa<Bool>()) {
          auto *bool_value = static_cast<bool *>(value->data_c());
          slice_args.push_back(MakeValue((static_cast<int64_t>(*bool_value)))->ToAbstract());
        } else if (tensor_dtype == kInt64) {
          auto *int_value = static_cast<int64_t *>(value->data_c());
          slice_args.push_back(MakeValue((*int_value))->ToAbstract());
        } else if (tensor_dtype == kInt32) {
          auto *int_value = static_cast<int32_t *>(value->data_c());
          slice_args.push_back(MakeValue((*int_value))->ToAbstract());
        } else {
          MS_EXCEPTION(TypeError) << "The input tensor type of the MakeSlice operator must be int or bool, but got "
                                  << tensor_dtype->ToString();
        }
      } else {
        slice_args.push_back(args_spec_list[index]);
      }
    } else {
      MS_EXCEPTION(TypeError) << "The " << index << "th input of MakeSlice operator should be scalar, none or tensor, "
                              << "but got " << args_spec_list[index]->ToString() << ".";
    }
  }
  // Slice: start, end, step
  constexpr size_t kMakeSliceInput0 = 0;
  constexpr size_t kMakeSliceInput1 = 1;
  constexpr size_t kMakeSliceInput2 = 2;
  return std::make_shared<AbstractSlice>(slice_args[kMakeSliceInput0], slice_args[kMakeSliceInput1],
                                         slice_args[kMakeSliceInput2]);
}

AbstractBasePtr InferImplStopGradient(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: any value;
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Clone();
}

AbstractBasePtr InferImplDictLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  return InferTupleOrListOrDictLen<AbstractDictionary>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplJ(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const AbstractBasePtrList &args_spec_list) {
  // args: An object of AbstractFunction.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  MS_LOG(DEBUG) << "evaluate J: " << args_spec_list[0]->ToString();

  AbstractFunctionPtr x = dyn_cast<AbstractFunction>(args_spec_list[0]);
  if (x == nullptr) {
    return std::make_shared<AbstractJTagged>(args_spec_list[0]);
  }

  AbstractFuncAtomPtrList jv;
  auto build_jv = [&jv](const AbstractFuncAtomPtr &func) {
    auto j_closure = std::make_shared<JTransformedAbstractClosure>(func);
    jv.push_back(j_closure);
  };
  x->Visit(build_jv);

  return AbstractFunction::MakeAbstractFunction(jv);
}

AbstractBasePtr InferImplTaylor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // args: An object of AbstractFunction.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  MS_LOG(DEBUG) << "evaluate Taylor: " << args_spec_list[0]->ToString();

  AbstractFunctionPtr x = dyn_cast<AbstractFunction>(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(x);

  AbstractFuncAtomPtrList taylor_v;
  auto build_taylor_v = [&taylor_v](const AbstractFuncAtomPtr &func) {
    auto taylor_closure = std::make_shared<TaylorTransformedAbstractClosure>(func);
    taylor_v.push_back(taylor_closure);
  };
  x->Visit(build_taylor_v);

  return AbstractFunction::MakeAbstractFunction(taylor_v);
}

AbstractBasePtr InferImplShard(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  // Inputs: func, in_axes, out_axes, device, level.
  constexpr size_t shard_input_size = 5;
  CheckArgsSize(primitive->name(), args_spec_list, shard_input_size);
  MS_LOG(DEBUG) << "Evaluate Shard: " << args_spec_list[0]->ToString();

  AbstractFunctionPtr x = dyn_cast<AbstractFunction>(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(x);

  AbstractFuncAtomPtrList shard_v;
  auto build_shard_v = [&shard_v](const AbstractFuncAtomPtr &func) {
    auto shard_closure = std::make_shared<ShardTransformedAbstractClosure>(func);
    shard_v.push_back(shard_closure);
  };
  x->Visit(build_shard_v);

  return AbstractFunction::MakeAbstractFunction(shard_v);
}

AbstractBasePtr InferImplVmap(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // args: An object of AbstractFunction.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  auto fn_arg = args_spec_list[0];
  MS_LOG(DEBUG) << "Evaluate Vmap: " << fn_arg->ToString() << ".";

  AbstractFuncAtomPtrList vmap_v;
  ValuePtr in_axes = primitive->GetAttr("in_axes");
  ValuePtr out_axes = primitive->GetAttr("out_axes");
  ValuePtr cell_size_value = primitive->GetAttr("cell_size");
  MS_EXCEPTION_IF_NULL(cell_size_value);
  auto cell_size = cell_size_value->isa<UInt64Imm>() ? dyn_cast<UInt64Imm>(cell_size_value)->value() : 0;

  auto traverse_fn = [&vmap_v, &in_axes, &out_axes, &cell_size](const AbstractBasePtr &fn_arg) {
    AbstractFunctionPtr x = dyn_cast<AbstractFunction>(fn_arg);
    MS_EXCEPTION_IF_NULL(x);
    auto build_vmap_v = [&vmap_v, &in_axes, &out_axes, &cell_size](const AbstractFuncAtomPtr &func) {
      auto vmap_closure = std::make_shared<VmapTransformedAbstractClosure>(func, in_axes, out_axes, cell_size);
      vmap_v.push_back(vmap_closure);
    };
    x->Visit(build_vmap_v);
  };

  AbstractTuplePtr cell_list = dyn_cast<AbstractTuple>(fn_arg);
  if (cell_list != nullptr) {
    const auto &cell_list_fns = cell_list->elements();
    for (const auto &fn : cell_list_fns) {
      traverse_fn(fn);
    }
  } else {
    traverse_fn(fn_arg);
  }

  return AbstractFunction::MakeAbstractFunction(vmap_v);
}

AbstractBasePtr InferImplFakeBprop(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

void GetStringAndNumberFromAbstract(const std::string &op_name, const AbstractBasePtrList &args_spec_list,
                                    std::string *str, int64_t *num) {
  constexpr size_t args_num = 2;
  CheckArgsSize(op_name, args_spec_list, args_num);
  AbstractScalarPtr scalar_x = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractScalarPtr scalar_y = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);
  ValuePtr value_x = scalar_x->BuildValue();
  ValuePtr value_y = scalar_y->BuildValue();

  bool is_match = false;
  if (value_x->isa<StringImm>()) {
    *str = GetValue<std::string>(value_x);
    if (value_y->isa<Int32Imm>()) {
      *num = IntToLong(GetValue<int32_t>(value_y));
      is_match = true;
    } else if (value_y->isa<Int64Imm>()) {
      *num = GetValue<int64_t>(value_y);
      is_match = true;
    }
  } else if (value_y->isa<StringImm>()) {
    *str = GetValue<std::string>(value_y);
    if (value_x->isa<Int32Imm>()) {
      *num = IntToLong(GetValue<int32_t>(value_x));
      is_match = true;
    } else if (value_x->isa<Int64Imm>()) {
      *num = GetValue<int64_t>(value_x);
      is_match = true;
    }
  }
  if (!is_match) {
    MS_LOG(EXCEPTION) << op_name << " requires the input to be a string and an integer, but got " << value_x->ToString()
                      << " and " << value_y->ToString() << ".";
  }
}

AbstractBasePtr InferImplStringMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a string and an integer.
  std::string str;
  int64_t num = 0;
  const std::string op_name = primitive->name();
  GetStringAndNumberFromAbstract(op_name, args_spec_list, &str, &num);
  std::string res;
  // If num is less than or equal to 0, return an empty string.
  if (num > 0) {
    for (auto i = 0; i < num; i++) {
      res += str;
    }
  }
  return std::make_shared<AbstractScalar>(res);
}

AbstractBasePtr InferImplStringGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a string and an integer.
  std::string str;
  int64_t num = 0;
  const std::string op_name = primitive->name();
  GetStringAndNumberFromAbstract(op_name, args_spec_list, &str, &num);
  int64_t len = SizeToLong(str.length());
  if (num >= len || num < -len) {
    MS_LOG(EXCEPTION) << "String index out of range, expect:[" << -len << ", " << (len - 1) << "], but got " << num
                      << ".";
  }
  if (num < 0) {
    num += len;
  }
  std::string res;
  (void)res.append(1, str.at(num));
  return std::make_shared<AbstractScalar>(res);
}

bool PrimNeedFrontendInferValue(const PrimitivePtr &primitive) {
  // The operators in this list are registered on the core/ops, which means operators are registered on both frontend
  // and backend, affects the infer value of the frontend. We use this list to skip the registration of the backend, so
  // that the optimization of the frontend like constant folding, can be carried out smoothly. We need to delete this
  // list when the infer value can be mapped to the CPU backend operator.
  static std::vector<PrimitivePtr> skip_frontend_registration_list{
    prim::kPrimAdd, prim::kPrimMod,          prim::kPrimMul,   prim::kPrimRealDiv,
    prim::kPrimSub, prim::kPrimStridedSlice, prim::kPrimStack, prim::kPrimTensorScatterUpdate,
    prim::kPrimTile};
  if (std::any_of(skip_frontend_registration_list.begin(), skip_frontend_registration_list.end(),
                  [&primitive](const PrimitivePtr &item) { return IsPrimitiveEquals(primitive, item); })) {
    return true;
  }
  return false;
}

// using R = PrimitiveEvalImplMap::mapped_type;
static PrimitiveEvalImplMap frontend_prim_infer_map{
  // frontend
};
PrimitiveEvalImplMap *GetFrontendPrimitiveInferMapPtr() { return &frontend_prim_infer_map; }
const PrimitiveEvalImplMap &GetFrontendPrimitiveInferMap() { return frontend_prim_infer_map; }
std::optional<StandardPrimitiveImplReg> GetFrontendPrimitiveInferImpl(const PrimitivePtr &primitive) {
  auto iter = GetFrontendPrimitiveInferMap().find(primitive);
  if (iter != GetFrontendPrimitiveInferMap().end()) {
    return iter->second;
  }

  // We need to delete this when the infer value can be mapped to the CPU backend operator.
  if (PrimNeedFrontendInferValue(primitive)) {
    return std::optional<StandardPrimitiveImplReg>();
  }

  auto find = abstract::GetPrimitiveInferImpl(primitive);
  if (find.has_value()) {
    return find.value();
  }
  return std::optional<StandardPrimitiveImplReg>();
}

AbstractBasePtr SetAdapterFlag(const std::string &op_name, const AbstractBasePtr &abs_input, bool adapter_flag) {
  MS_EXCEPTION_IF_NULL(abs_input);
  // Clone is needed here.
  if (abs_input->isa<AbstractRefTensor>()) {
    auto abs_ref = abs_input->Clone()->cast<AbstractRefPtr>();
    abs_ref->set_is_adapter(adapter_flag);
    return abs_ref;
  }
  if (abs_input->isa<AbstractTensor>()) {
    auto abs_tensor = abs_input->Clone()->cast<AbstractTensorPtr>();
    abs_tensor->set_is_adapter(adapter_flag);
    return abs_tensor;
  }
  MS_LOG(EXCEPTION) << op_name << " requires a tensor as the first argument, but got " << abs_input->ToString();
}

AbstractBasePtr InferImplConvertToAdapterTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  constexpr size_t args_num = 1;
  constexpr size_t input_index = 0;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, args_num);
  return SetAdapterFlag(op_name, args_spec_list[input_index], true);
}

AbstractBasePtr InferImplConvertToMsTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  constexpr size_t args_num = 1;
  constexpr size_t input_index = 0;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, args_num);
  return SetAdapterFlag(op_name, args_spec_list[input_index], false);
}

#ifndef _MSC_VER
// String
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(StringMul, prim::kPrimStringMul, InferImplStringMul, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(StringGetItem, prim::kPrimStringGetItem, InferImplStringGetItem, nullptr);
// Tuple
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TupleReversed, prim::kPrimTupleReversed, InferImplTupleReversed, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TupleDiv, prim::kPrimTupleDiv, InferImplTupleDiv, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TupleToArray, prim::kPrimTupleToArray, InferImplTuple2Array, nullptr);
// List
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ListReduce, prim::kPrimListReduce, InferImplListReduce, nullptr);
// Dict
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(DictLen, prim::kPrimDictLen, InferImplDictLen, nullptr);
// Slice
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(MakeSlice, prim::kPrimMakeSlice, InferImplMakeSlice, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(SliceGetItem, prim::kPrimSliceGetItem, InferImplSliceGetItem, nullptr);
// Type
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TypeOf, prim::kPrimTypeOf, InferImplTypeof, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(TopTypeOf, prim::kPrimTopTypeOf, InferImplTopTypeof, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(HasType, prim::kPrimHasType, InferImplHasType, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(IsInstance, prim::kPrimIsInstance, InferImplIsInstance, nullptr);
// Shape
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ReducedShape, prim::kPrimReducedShape, InferImplReduceShape, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ShapeMul, prim::kPrimShapeMul, InferImplShapeMul, nullptr);
// Auto-Grad
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(StopGradient, prim::kPrimStopGradient, InferImplStopGradient, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(FakeBprop, prim::kPrimFakeBprop, InferImplFakeBprop, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(J, prim::kPrimJ, InferImplJ, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(BroadcastGradientArgs, prim::kPrimBroadcastGradientArgs,
                                   InferImplBroadcastGradientArgs, nullptr);
// Other
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(Taylor, prim::kPrimTaylor, InferImplTaylor, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(Shard, prim::kPrimShard, InferImplShard, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(Vmap, prim::kPrimVmap, InferImplVmap, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(Lower, prim::kPrimLower, InferImplLower, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ConvertToAdapterTensor, prim::kPrimConvertToAdapterTensor,
                                   InferImplConvertToAdapterTensor, nullptr);
REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(ConvertToMsTensor, prim::kPrimConvertToMsTensor, InferImplConvertToMsTensor,
                                   nullptr);
#else
void RegPrimitiveFrontEval() {
  // String
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimStringMul,
                                                InferImplStringMul, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimStringGetItem,
                                                InferImplStringGetItem, nullptr);
  // Tuple
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimTupleReversed,
                                                InferImplTupleReversed, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimTupleDiv,
                                                InferImplTupleDiv, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimTupleToArray,
                                                InferImplTuple2Array, nullptr);
  // List
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimListReduce,
                                                InferImplListReduce, nullptr);
  // Dict
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimDictLen,
                                                InferImplDictLen, nullptr);
  // Slice
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimMakeSlice,
                                                InferImplMakeSlice, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimSliceGetItem,
                                                InferImplSliceGetItem, nullptr);
  // Type
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimTypeOf,
                                                InferImplTypeof, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimTopTypeOf,
                                                InferImplTopTypeof, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimHasType,
                                                InferImplHasType, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimIsInstance,
                                                InferImplIsInstance, nullptr);
  // Shape
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimReducedShape,
                                                InferImplReduceShape, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimShapeMul,
                                                InferImplShapeMul, nullptr);
  // Auto-Grad
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimStopGradient,
                                                InferImplStopGradient, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimFakeBprop,
                                                InferImplFakeBprop, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimJ, InferImplJ,
                                                nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(),
                                                prim::kPrimBroadcastGradientArgs, InferImplBroadcastGradientArgs,
                                                nullptr);
  // Other
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimTaylor,
                                                InferImplTaylor, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimShard,
                                                InferImplShard, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimVmap,
                                                InferImplVmap, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), prim::kPrimLower,
                                                InferImplLower, nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(),
                                                prim::kPrimConvertToAdapterTensor, InferImplConvertToAdapterTensor,
                                                nullptr);
  abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(),
                                                prim::kPrimConvertToMsTensor, InferImplConvertToMsTensor, nullptr);
}  // namespace abstract
#endif
}  // namespace abstract
}  // namespace mindspore
