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
#include "frontend/operator/ops_front_infer_function.h"

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "abstract/abstract_value.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "abstract/param_validator.h"
#include "pybind_api/ir/tensor_py.h"
#include "frontend/operator/ops.h"
#include "abstract/infer_functions.h"
#include "utils/convert_utils_py.h"

namespace mindspore {
namespace abstract {
enum State {
  SAME,
  X_ONE,
  Y_ONE,
};

struct SlideInfo {
  int64_t start;
  int64_t step;
  int64_t stop;
};

template <typename T>
AbstractBasePtr InferImplTupleOrListEqual(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples or two lists.
  CheckArgsSize(op_name, args_spec_list, 2);
  auto input_x = CheckArg<T>(op_name, args_spec_list, 0);
  auto input_y = CheckArg<T>(op_name, args_spec_list, 1);

  ValuePtr x_value = input_x->BuildValue();
  ValuePtr y_value = input_y->BuildValue();
  return std::make_shared<AbstractScalar>(*x_value == *y_value);
}

void CalcSlidePara(const AbstractBasePtrList &args_spec_list, SlideInfo *slide) {
  int64_t arg1 = 0;
  int64_t arg2 = 0;
  if (!args_spec_list.empty()) {
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    auto arg_value = args_spec_list[0]->BuildValue();
    if (!arg_value->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "Only supported input an int64 number.";
    }
    arg1 = GetValue<int64_t>(arg_value);
  }

  if (args_spec_list.size() >= 2) {
    MS_EXCEPTION_IF_NULL(args_spec_list[1]);
    auto arg_value = args_spec_list[1]->BuildValue();
    if (!arg_value->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "Only supported input an int64 number.";
    }
    arg2 = GetValue<int64_t>(arg_value);
  }

  if (args_spec_list.size() == 3) {
    MS_EXCEPTION_IF_NULL(args_spec_list[2]);
    auto arg_value = args_spec_list[2]->BuildValue();
    if (!arg_value->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "Only supported input an int64 number.";
    }
    slide->step = GetValue<int64_t>(arg_value);
    slide->start = arg1;
    slide->stop = arg2;
  }

  if (args_spec_list.size() == 2) {
    slide->start = arg1;
    slide->stop = arg2;
  }

  if (args_spec_list.size() == 1) {
    slide->stop = arg1;
  }
}

void ComputeReduceIndex(const std::vector<int64_t> &reverse_x, const std::vector<int64_t> &reverse_y,
                        std::vector<int64_t> *grad_x_reduce_idx, std::vector<int64_t> *grad_y_reduce_idy) {
  const size_t n = reverse_x.size();
  for (size_t i = 0; i < n; ++i) {
    State curr;
    const int64_t x_i = reverse_x[i];
    const int64_t y_i = reverse_y[i];
    const int64_t reduce_idx = SizeToLong(n - 1 - i);
    if (x_i == y_i) {
      curr = SAME;
    } else if (x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
      curr = X_ONE;
    } else if (y_i == 1) {
      grad_y_reduce_idy->push_back(reduce_idx);
      curr = Y_ONE;
    } else {
      MS_LOG(EXCEPTION) << "not compatible shape input for BroadcastGradientArgs";
    }
    if (curr == SAME && x_i == 1) {
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
                                   const ValueSequeuePtr &axis_value_ptr, const PrimitivePtr &primitive) {
  size_t x_rank = x_shape->size();
  std::set<int64_t> axis_set;
  auto axis_data = axis_value_ptr->value();
  if (axis_data.empty()) {
    int64_t size = 1;
    AbstractBasePtrList values(x_rank, std::make_shared<AbstractScalar>(size));
    return std::make_shared<AbstractTuple>(values);
  }

  for (auto &elem : axis_data) {
    int64_t e_value = CheckAxis(primitive->name(), elem, -SizeToLong(x_rank), SizeToLong(x_rank) - 1);
    (void)axis_set.insert(e_value);
  }

  auto x_shp_data = x_shp_value->cast<ValueTuplePtr>()->value();
  if (x_shp_data.size() < x_rank) {
    MS_LOG(EXCEPTION) << "x_shape_data.size() " << x_shp_data.size() << " less than x_shape.size() " << x_rank;
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
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto arg_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  auto arg_y = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);

  ValueTuplePtr arg_x_value = arg_x->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(arg_x_value);

  ValueTuplePtr arg_y_value = arg_y->BuildValue()->cast<ValueTuplePtr>();
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

AbstractBasePtr InferImplListMap(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  // Inputs: fn, list1, list2, ...
  MS_EXCEPTION_IF_NULL(engine);
  if (args_spec_list.size() <= 1) {
    MS_LOG(EXCEPTION) << "List_map requires at least 1 list. while the input size is  " << args_spec_list.size() << ".";
  }
  AbstractFunctionPtr fn = CheckArg<AbstractFunction>(primitive->name(), args_spec_list, 0);
  // check args from 1.
  CheckArgsSpec<AbstractList>(AbstractBasePtrList(args_spec_list.begin() + 1, args_spec_list.end()));

  AbstractBasePtrList subargs;
  for (std::size_t i = 1; i < args_spec_list.size(); i++) {
    AbstractListPtr l_ptr = dyn_cast<AbstractList>(args_spec_list[i]);
    if (l_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Argument[" << i << "] of list_map should be a list.";
    }
    subargs.push_back(AbstractJoin(l_ptr->elements()));
  }
  EvalResultPtr engin_exc = engine->Execute(fn, subargs);
  AbstractBasePtrList result;
  for (std::size_t i = 1; i < args_spec_list.size(); i++) {
    result.push_back(engin_exc->abstract());
  }
  return std::make_shared<AbstractList>(result);
}

AbstractBasePtr InferImplListReduce(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: a fn, a list and an object of a subclass of a AbstractBase.
  MS_EXCEPTION_IF_NULL(engine);
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  AbstractFunctionPtr fn = CheckArg<AbstractFunction>(op_name, args_spec_list, 0);
  AbstractListPtr lst = CheckArg<AbstractList>(op_name, args_spec_list, 1);
  AbstractBasePtr dflt = args_spec_list[2];

  AbstractBasePtr list_type = AbstractJoin(lst->elements());
  auto result1 = engine->Execute(fn, lst->elements());
  auto result2 = engine->Execute(fn, {dflt, list_type});
  MS_EXCEPTION_IF_NULL(result1->abstract());
  MS_EXCEPTION_IF_NULL(result2->abstract());
  return result1->abstract()->Join(result2->abstract());
}

AbstractBasePtr InferImplTupleReversed(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTuplePtr input = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);

  auto tuple_elements = input->elements();
  AbstractBasePtrList elem_list;
  (void)std::transform(tuple_elements.rbegin(), tuple_elements.rend(), std::back_inserter(elem_list),
                       [](const AbstractBasePtr &elem) { return elem->Clone(); });
  return std::make_shared<AbstractTuple>(elem_list);
}

AbstractBasePtr InferImplReduceShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: x_shape, axis
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);

  auto x_shp_value = shape_x->BuildValue();
  if (x_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << op_name
                      << " evaluator shape's data field can't be anything: " << args_spec_list[1]->ToString();
  }

  // Axis can be scalar, tuple or list
  AbstractSequeuePtr axis = nullptr;
  if (args_spec_list[1]->isa<AbstractScalar>()) {
    MS_LOG(DEBUG) << op_name << " evaluator second parameter is scalar";
    AbstractBasePtrList axis_list = {dyn_cast<AbstractScalar>(args_spec_list[1])};
    axis = std::make_shared<AbstractTuple>(axis_list);
  } else if (args_spec_list[1]->isa<AbstractSequeue>()) {
    MS_LOG(DEBUG) << op_name << " evaluator second parameter is sequeue";
    axis = args_spec_list[1]->cast<AbstractSequeuePtr>();
  } else {
    MS_LOG(EXCEPTION) << op_name << " evaluator second parameter should be a scalar or tuple or list, but got "
                      << args_spec_list[1]->ToString();
  }

  auto axis_value = axis->BuildValue();
  if (axis_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << op_name
                      << " evaluator shape's data field can't be anything: " << args_spec_list[1]->ToString();
  }
  auto axis_value_ptr = axis_value->cast<ValueSequeuePtr>();
  MS_EXCEPTION_IF_NULL(axis_value_ptr);

  return DoInferReduceShape(shape_x, x_shp_value, axis_value_ptr, primitive);
}

AbstractBasePtr InferImplTupleDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr shape_x = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractTuplePtr div_shp = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  MS_LOG(INFO) << "DivShape input:" << shape_x->ToString() << ", div:" << div_shp->ToString();

  auto div_shp_value = div_shp->BuildValue();
  if (div_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "shape's data field can't be anythin: " << args_spec_list[0]->ToString();
  }

  auto shpx_value = shape_x->BuildValue();
  if (shpx_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "shape's data field can't be anythin: " << args_spec_list[1]->ToString();
  }

  if (div_shp->size() != shape_x->size()) {
    MS_LOG(EXCEPTION) << "tileshape elems shape must the same div_shp: " << div_shp->size()
                      << ", shapex: " << shape_x->size() << ".";
  }

  auto shpx_data = shpx_value->cast<ValueTuplePtr>()->value();
  auto div_shp_data = div_shp_value->cast<ValueTuplePtr>()->value();
  AbstractBasePtrList values;

  for (size_t i = 0; i < div_shp_data.size(); i++) {
    if (div_shp_data[i]->cast<Int64ImmPtr>() == nullptr) {
      MS_LOG(EXCEPTION) << "div_shp_shape data should be an int64 number, but it's " << args_spec_list[1]->ToString();
    }
    int64_t shapex_value = GetValue<int64_t>(shpx_data[i]);
    int64_t div_value = GetValue<int64_t>(div_shp_data[i]);
    MS_LOG(DEBUG) << "div_shp_shape data shapex_value :" << shapex_value << " div_value: " << div_value;
    if (div_value == 0) {
      MS_LOG(EXCEPTION) << "error: division value should not be 0!";
    }
    if ((shapex_value % div_value) != 0) {
      MS_LOG(EXCEPTION) << "div_shp_shape data shapex must div int64_t:" << shapex_value << " div_value: " << div_value;
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

  py::tuple data_tuple = ValuePtrToPyData(input->BuildValue());
  py::array data = py::array(data_tuple);
  auto tensor = tensor::TensorPy::MakeTensor(data);
  auto ret = tensor->ToAbstract();
  ret->set_value(tensor);
  MS_LOG(DEBUG) << "Tuple2arry result AbstractTensor: " << ret->ToString();
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
    MS_LOG(EXCEPTION) << "shape's data field can't be anythin: " << shape_x->ToString();
  }

  auto shpx_data = shpx_value->cast<ValueTuplePtr>()->value();

  int64_t result = 1;
  for (size_t i = 0; i < shpx_data.size(); i++) {
    int64_t value = GetValue<int64_t>(shpx_data[i]);
    result = IntMulWithOverflowCheck(result, value);
  }

  auto result_v = MakeValue(result);
  MS_LOG(DEBUG) << "shape mul result:" << result_v->ToString();
  return std::make_shared<AbstractScalar>(result_v, result_v->type());
}

AbstractBasePtr InferImplMakeRange(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "Cannot make range from empty input.";
  }

  if (args_spec_list.size() > 3) {
    MS_LOG(EXCEPTION) << "Error args size of make range operational.";
  }

  SlideInfo slide = {0, 1, 0};
  CalcSlidePara(args_spec_list, &slide);

  if (slide.step == 0) {
    MS_LOG(EXCEPTION) << "Error, step value is 0.";
  }

  AbstractBasePtrList args;
  if (slide.start <= slide.stop) {
    if (slide.step <= 0) {
      MS_LOG(EXCEPTION) << "Error slice[" << slide.start << ", " << slide.stop << ", " << slide.step << "]";
    }

    for (int64_t i = slide.start; i < slide.stop; i += slide.step) {
      args.push_back(abstract::FromValue(i));
      if (i > 0 && INT_MAX - i < slide.step) {
        MS_EXCEPTION(ValueError) << "For make range, the required cycles number is greater than max cycles number, "
                                    "will cause integer overflow.";
      }
    }
  } else {
    if (slide.step >= 0) {
      MS_LOG(EXCEPTION) << "Error slice[" << slide.start << ", " << slide.stop << ", " << slide.step << "]";
    }

    for (int64_t i = slide.start; i > slide.stop; i += slide.step) {
      args.push_back(abstract::FromValue(i));
      if (i < 0 && INT_MIN - i > slide.step) {
        MS_EXCEPTION(ValueError) << "For make range, the required cycles number is greater than max cycles number, "
                                    "will cause integer overflow.";
      }
    }
  }

  return std::make_shared<AbstractTuple>(args);
}

AbstractBasePtr InferImplStopGradient(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: any value;
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Clone();
}

AbstractBasePtr InferImplTupleEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  return InferImplTupleOrListEqual<AbstractTuple>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplListEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  return InferImplTupleOrListEqual<AbstractList>(primitive->name(), args_spec_list);
}

AbstractBasePtr InferImplStringEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: two scalars whose value is a string.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr scalar_x = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractScalarPtr scalar_y = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr value_x = scalar_x->BuildValue();
  ValuePtr value_y = scalar_y->BuildValue();
  if (!value_x->isa<StringImm>() || !value_y->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " requires 2 parameters are string, but got param0: " << value_x->ToString()
                      << ", param1: " << value_y->ToString();
  }

  bool ret = (value_x->cast<StringImmPtr>()->value() == value_y->cast<StringImmPtr>()->value());
  return std::make_shared<AbstractScalar>(ret);
}

AbstractBasePtr InferImplStringConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: two scalars whose value is a string.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractScalarPtr scalar_x = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  AbstractScalarPtr scalar_y = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);

  ValuePtr value_x = scalar_x->BuildValue();
  ValuePtr value_y = scalar_y->BuildValue();
  if (!value_x->isa<StringImm>() || !value_y->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << op_name << " requires 2 parameters are string, but got param0: " << value_x->ToString()
                      << ", param1: " << value_y->ToString();
  }

  std::string ret = (value_x->cast<StringImmPtr>()->value() + value_y->cast<StringImmPtr>()->value());
  return std::make_shared<AbstractScalar>(ret);
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

AbstractBasePtr InferImplFakeBprop(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

// Eval the return type of make_record
AbstractBasePtr InferImplMakeRecord(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: at lease two objects of a subclass of AbstractBase.
  if (args_spec_list.size() < 2) {
    MS_LOG(EXCEPTION) << "Typeof evaluator requires more than 1 parameter, while the input size is "
                      << args_spec_list.size() << ".";
  }

  // args_spec_list[0] maybe AbstractScalarPtr or AbstractTypePtr
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  TypePtr type = args_spec_list[0]->GetTypeTrack();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() != kMetaTypeTypeType) {
    MS_LOG(EXCEPTION) << "Can not make type(" << type->ToString() << ")not TypeType";
  }

  ValuePtr value_track = args_spec_list[0]->GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  TypePtr type_ptr = value_track->cast<TypePtr>();
  if (type_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Value type error, not Me type:" << value_track->ToString();
  }

  auto cls = dyn_cast<Class>(type_ptr);
  MS_EXCEPTION_IF_NULL(cls);
  ClassAttrVector attributes = cls->GetAttributes();
  CheckArgsSize(primitive->name(), args_spec_list, attributes.size() + 1);

  std::vector<AbstractAttribute> abs_attributes;
  for (size_t i = 0; i < attributes.size(); i++) {
    AbstractAttribute elem(attributes[i].first, args_spec_list[i + 1]);
    abs_attributes.push_back(elem);
  }

  return std::make_shared<AbstractClass>(cls->tag(), abs_attributes, cls->methods());
}

AbstractBasePtr InferImplAssign(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: Ref, value, [universal]
  CheckRequiredArgsSize(primitive->name(), args_spec_list, 2);

  MS_LOG(DEBUG) << "InferImplAssign " << args_spec_list[0];
  auto type = args_spec_list[0]->BuildType();
  if (type->type_id() == kObjectTypeRefKey) {
    return args_spec_list[1]->Broaden();
  } else {
    return args_spec_list[0];
  }
}

AbstractBasePtr InferImplLoad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: Ref/Tensor, universal
  CheckArgsSize(primitive->name(), args_spec_list, 2);
  auto ref_abs = dyn_cast<abstract::AbstractRef>(args_spec_list[0]);
  if (ref_abs != nullptr) {
    // Return tensor value if input is Ref.
    return ref_abs->CloneAsTensor();
  }
  return args_spec_list[0]->Broaden();
}

REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(TypeOf, prim::kPrimTypeOf, InferImplTypeof);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(HasType, prim::kPrimHasType, InferImplHasType);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(MakeRecord, prim::kPrimMakeRecord, InferImplMakeRecord);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(ListMap, prim::kPrimListMap, InferImplListMap);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(ListReduce, prim::kPrimListReduce, InferImplListReduce);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(TupleReversed, prim::kPrimTupleReversed, InferImplTupleReversed);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(ReducedShape, prim::kPrimReducedShape, InferImplReduceShape);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(TupleDiv, prim::kPrimTupleDiv, InferImplTupleDiv);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(TupleToArray, prim::kPrimTupleToArray, InferImplTuple2Array);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(ShapeMul, prim::kPrimShapeMul, InferImplShapeMul);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(TupleEqual, prim::kPrimTupleEqual, InferImplTupleEqual);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(ListEqual, prim::kPrimListEqual, InferImplListEqual);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(MakeRange, prim::kPrimMakeRange, InferImplMakeRange);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(StopGradient, prim::kPrimStopGradient, InferImplStopGradient);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(StringEqual, prim::kPrimStringEqual, InferImplStringEqual);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(StringConcat, prim::kPrimStringConcat, InferImplStringConcat);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(DictLen, prim::kPrimDictLen, InferImplDictLen);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(FakeBprop, prim::kPrimFakeBprop, InferImplFakeBprop);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(J, prim::kPrimJ, InferImplJ);
REGISTER_FRONTENT_PRIMITIVE_EVAL_IMPL(BroadcastGradientArgs, prim::kPrimBroadcastGradientArgs,
                                      InferImplBroadcastGradientArgs);
REGISTER_PRIMITIVE_EVAL_IMPL(Assign, prim::kPrimAssign, InferImplAssign);
REGISTER_PRIMITIVE_EVAL_IMPL(Load, prim::kPrimLoad, InferImplLoad);
}  // namespace abstract
}  // namespace mindspore
