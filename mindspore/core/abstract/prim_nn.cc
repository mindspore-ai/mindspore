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

#include <cmath>
#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
int64_t GetAndCheckFormat(const ValuePtr &value) {
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value, &data_format);
  if (!result ||
      (data_format != static_cast<int64_t>(Format::NHWC) && data_format != static_cast<int64_t>(Format::NCHW) &&
       data_format != static_cast<int64_t>(Format::NCDHW))) {
    MS_LOG(EXCEPTION) << "data format is invalid, only support NCHW, NHWC and NCDHW";
  }
  return data_format;
}

AbstractBasePtr InferImplPooling(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input_tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  (void)CheckTensorDType(input_tensor, {kFloat16, kFloat32}, "Input 0 of Pooling should be %s");

  ShapePtr input_shape = dyn_cast<Shape>(input_tensor->GetShapeTrack());  // NCHW
  MS_EXCEPTION_IF_NULL(input_shape);
  const size_t input_shape_size = 4;
  if (input_shape->shape().size() != input_shape_size) {
    MS_LOG(EXCEPTION) << "Pooling input should be a 4-D tensor.";
  }
  const size_t H_INDEX = 2;
  const size_t W_INDEX = 3;
  int64_t h_input = input_shape->shape()[H_INDEX];
  int64_t w_input = input_shape->shape()[W_INDEX];

  int64_t window = GetValue<int64_t>(primitive->GetAttr("window"));
  int64_t stride = GetValue<int64_t>(primitive->GetAttr("stride"));
  int64_t padding = GetValue<int64_t>(primitive->GetAttr("pad"));
  int64_t nan_opt = GetValue<int64_t>(primitive->GetAttr("nan_opt"));
  int64_t data_mode = GetValue<int64_t>(primitive->GetAttr("data_mode"));
  int64_t ceil_mode = GetValue<int64_t>(primitive->GetAttr("ceil_mode"));

  if (stride <= 0) {
    MS_LOG(EXCEPTION) << "Invalid stride value: " << stride << ", should greater then 0";
  }
  if (nan_opt != 0) {
    MS_LOG(EXCEPTION) << "Invalid nan_opt value: " << nan_opt << ", should be 0";
  }
  if (data_mode != 1) {
    MS_LOG(EXCEPTION) << "Invalid data_mode value: " << data_mode << ", should be 1";
  }
  if (ceil_mode != 0) {
    MS_LOG(EXCEPTION) << "Invalid ceil_mode value: " << ceil_mode << ", should be 0";
  }

  auto pad_mode_ptr = primitive->GetAttr("pad_mode");
  if (pad_mode_ptr != nullptr) {
    int64_t pad_mode;
    const int64_t middle = 2;
    CheckAndConvertUtils::GetPadModEnumValue(pad_mode_ptr, &pad_mode, true);
    if (pad_mode == static_cast<int64_t>(PadMode::VALID)) {
      padding = 0;
    } else if (pad_mode == static_cast<int64_t>(PadMode::SAME)) {
      padding = (window - 1) / middle;
    }
  }
  std::set<std::string> available_mode{"max", "avg"};
  auto mode_ptr = primitive->GetAttr("mode");
  if ((mode_ptr != nullptr) && mode_ptr->isa<StringImm>()) {
    auto mode = mode_ptr->cast<StringImmPtr>()->value();
    if (available_mode.find(mode) == available_mode.end()) {
      MS_LOG(EXCEPTION) << "Unsupported pooling mode: " << mode << ".";
    }
  }
  const int64_t twice = 2;
  int64_t h_out = (((h_input + twice * padding - (window - 1)) - 1) / stride) + 1;
  int64_t w_out = (((w_input + twice * padding - (window - 1)) - 1) / stride) + 1;
  ShapeVector shape_out = {input_shape->shape()[0], input_shape->shape()[1], h_out, w_out};
  AbstractBasePtr ret = input_tensor->Broaden();
  ret->set_shape(std::make_shared<Shape>(shape_out));
  return ret;
}

AbstractBasePtr InferImplPoolingGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors(y, dy, x).
  constexpr auto kPoolingGradInputNum = 3;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kPoolingGradInputNum);
  auto out_y = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto d_out = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 2);
  (void)CheckTensorsDTypeSame({out_y, d_out, input_x}, {kInt, kUInt, kFloat},
                              op_name + "evaluator three inputs should be %s");

  AbstractBasePtr ret = d_out->Broaden();
  auto x_shape = dyn_cast<Shape>(args_spec_list[2]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(x_shape);

  ret->set_shape(x_shape);
  return ret;
}

AbstractBasePtr InferImplBatchNorm(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: five tensors(x, gamma, beta, mean, variance).
  constexpr auto kBatchNormInputNum = 5;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kBatchNormInputNum);
  AbstractTensorPtr input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_x->shape());
  ShapeVector x_shape = input_x->shape()->shape();
  ShapeVector x_min_shape = input_x->shape()->min_shape();
  ShapeVector x_max_shape = input_x->shape()->max_shape();
  CheckMinMaxShape(x_shape, &x_min_shape, &x_max_shape);

  auto input_tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  (void)CheckTensorDType(input_tensor, {kFloat16, kFloat32}, "For 'BatchNorm', input argument \'input_x\'");
  AbstractTensorPtrList tensorPtrList = std::vector<AbstractTensorPtr>();
  for (size_t i = 1; i < args_spec_list.size(); ++i) {
    auto param = CheckArg<AbstractTensor>(op_name, args_spec_list, i);
    tensorPtrList.push_back(param);
  }
  (void)CheckTensorsDTypeSame(tensorPtrList, {kFloat16, kFloat32},
                              "For 'BatchNorm',the type of arguments['scale', 'bias', 'mean', 'variance']");

  auto data_format_ptr = primitive->GetAttr("format");
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  int64_t data_format = GetAndCheckFormat(data_format_ptr);

  size_t c_axis = 1;
  if (data_format == static_cast<int64_t>(Format::NHWC)) {
    c_axis = 3;
  }
  for (size_t i = 1; i < args_spec_list.size(); ++i) {
    AbstractTensorPtr arg_spec = CheckArg<AbstractTensor>(op_name, args_spec_list, i);
    MS_EXCEPTION_IF_NULL(arg_spec);
    MS_EXCEPTION_IF_NULL(arg_spec->shape());
    ShapeVector arg_shape = arg_spec->shape()->shape();
    if (arg_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "Arg " << i << " rank should be 1, but got " << arg_shape.size();
    }
    if ((x_shape[c_axis] != Shape::SHP_ANY) && (arg_shape[0] != x_shape[c_axis])) {
      MS_EXCEPTION(ValueError) << "Arg " << i << " shape[0] should equal to x_shape[" << c_axis
                               << "]=" << x_shape[c_axis] << ", but got " << arg_shape[0];
    }
  }
  AbstractTensorPtr input_gamma = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  ShapeVector gamma_shape = input_gamma->shape()->shape();
  ShapeVector gamma_min_shape = input_gamma->shape()->min_shape();
  ShapeVector gamma_max_shape = input_gamma->shape()->max_shape();
  CheckMinMaxShape(gamma_shape, &gamma_min_shape, &gamma_max_shape);
  ShapePtr output_shape_ptr = std::make_shared<Shape>(x_shape, x_min_shape, x_max_shape);
  AbstractTensorPtr output = std::make_shared<AbstractTensor>(input_x->element(), output_shape_ptr);
  ShapePtr gamma_shape_ptr = std::make_shared<Shape>(gamma_shape, gamma_min_shape, gamma_max_shape);
  AbstractTensorPtr output_gamma = std::make_shared<AbstractTensor>(input_gamma->element(), gamma_shape_ptr);
  AbstractBasePtrList rets = {output, output_gamma, output_gamma, output_gamma, output_gamma};
  return std::make_shared<AbstractTuple>(rets);
}

AbstractBasePtr InferImplFusedSparseAdam(const AnalysisEnginePtr &, const PrimitivePtr &,
                                         const AbstractBasePtrList &args_spec_list) {
  // the output is useless, so we dont have to focus on the output shape
  constexpr size_t dx_index = 1;
  constexpr size_t dscale_index = 2;
  constexpr size_t dbias_index = 3;
  MS_EXCEPTION_IF_NULL(args_spec_list[dx_index]);
  MS_EXCEPTION_IF_NULL(args_spec_list[dscale_index]);
  MS_EXCEPTION_IF_NULL(args_spec_list[dbias_index]);

  auto dx = args_spec_list[dx_index]->Broaden();
  auto dscale = args_spec_list[dscale_index]->Broaden();
  auto dbias = args_spec_list[dbias_index]->Broaden();

  AbstractBasePtrList rets = {dx, dscale, dbias};
  return std::make_shared<AbstractTuple>(rets);
}

AbstractBasePtr InferImplBiasAddGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: at least one tensor(y_backprop)
  // Outputs: dbias
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << primitive->name() << " evaluator at least has 1 parameters, while the input size is "
                      << args_spec_list.size() << ".";
  }

  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  ShapePtr shape_y = dyn_cast<Shape>(args_spec_list[0]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(shape_y);
  ShapeVector y_dims = shape_y->shape();
  if (y_dims.size() < 2) {
    MS_LOG(EXCEPTION) << primitive->name() << " input y backprop, dim should >= 2, while " << y_dims.size() << ".";
  }
  ShapeVector bias_dims = {y_dims[1]};
  ShapePtr ret_shape = std::make_shared<Shape>(bias_dims);
  AbstractBasePtr ret = args_spec_list[0]->Broaden();
  ret->set_shape(ret_shape);
  return ret;
}

AbstractBasePtr InferImplHSigmoid(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  // add check, types other than half and float are from cpu
  auto tensor = CheckArg<AbstractTensor>(primitive->name(), args_spec_list, 0);
  (void)CheckTensorDType(tensor, {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32}, "Input of HSigmoid should be %s");
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplHSigmoidGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 2);
  // add check, types other than half and float are from cpu
  auto dout = CheckArg<AbstractTensor>(primitive->name(), args_spec_list, 0);
  auto x = CheckArg<AbstractTensor>(primitive->name(), args_spec_list, 1);
  (void)CheckTensorDType(dout, {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32},
                         "Dout of HSigmoidGrad should be %s");
  (void)CheckTensorDType(x, {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32}, "X of HSigmoidGrad should be %s");
  return args_spec_list[1]->Broaden();
}

AbstractBasePtr InferImplBpropCut(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  AbstractBasePtrList args_list;
  constexpr size_t out_and_dout_size = 2;
  for (size_t i = 0; i < args_spec_list.size() - out_and_dout_size; i++) {
    args_list.push_back(args_spec_list[i]->Broaden());
  }
  return std::make_shared<AbstractTuple>(args_list);
}

AbstractBasePtr InferImplDropout(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  ShapeVector min_shape = x->shape()->min_shape();
  ShapeVector max_shape = x->shape()->max_shape();
  CheckMinMaxShape(shape, &min_shape, &max_shape);
  auto output_shape =
    std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  AbstractBasePtrList ret = {output_shape, output_shape};
  return std::make_shared<AbstractTuple>(ret);
}

AbstractBasePtr InferImplSparseApplyFtrl(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  CheckRequiredArgsSize(primitive->name(), args_spec_list, 5);
  AbstractBasePtrList elements;
  for (size_t i = 0; i < 3; ++i) {
    elements.push_back(args_spec_list[i]->Clone()->Broaden());
  }
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplSparseApplyProximalAdagrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const AbstractBasePtrList &args_spec_list) {
  CheckRequiredArgsSize(primitive->name(), args_spec_list, 7);
  AbstractBasePtrList elements;
  const size_t args_size = 2;
  for (size_t i = 0; i < args_size; ++i) {
    elements.push_back(args_spec_list[i]->Clone()->Broaden());
  }
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplSGD(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  CheckRequiredArgsSize(primitive->name(), args_spec_list, 6);
  AbstractBasePtrList elements;
  elements.push_back(args_spec_list[0]->Clone()->Broaden());
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplCTCGreedyDecoder(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list) {
  // inputs: inputs, sequence_length
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  constexpr size_t size_expected = 3;
  auto shape = input->shape();
  if (shape->shape().size() != size_expected) {
    MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 3.";
  }

  ShapeVector indices_shape = {Shape::SHP_ANY, 2};
  ShapeVector min_shape = {1, 2};
  ShapeVector max_shape = {shape->shape()[0] * shape->shape()[1], 2};
  auto decoded_indices =
    std::make_shared<AbstractTensor>(kInt64, std::make_shared<Shape>(indices_shape, min_shape, max_shape));

  ShapeVector values_shape = {Shape::SHP_ANY};
  ShapeVector values_min_shape = {1};
  ShapeVector values_max_shape = {shape->shape()[0] * shape->shape()[1]};
  ShapePtr values_shapes = std::make_shared<Shape>(values_shape, values_min_shape, values_max_shape);
  auto decoded_values = std::make_shared<AbstractTensor>(kInt64, values_shapes);

  ShapeVector decoded_shape_shape = {2};
  auto decoded_shape = std::make_shared<AbstractTensor>(kInt64, decoded_shape_shape);

  ShapeVector log_probability_shape = {shape->shape()[1], 1};
  auto log_probability =
    std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(log_probability_shape));

  // outputs: decoded_indices, decoded_values, decoded_shape, log_probability
  AbstractBasePtrList elements = {decoded_indices, decoded_values, decoded_shape, log_probability};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplPad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_shp = arg->shape()->shape();
  MS_EXCEPTION_IF_NULL(primitive);
  auto padding_attr = primitive->GetAttr("paddings");
  MS_EXCEPTION_IF_NULL(padding_attr);
  if (!padding_attr->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Paddings is not a ValueTuple";
  }
  std::vector<ValuePtr> paddings = padding_attr->cast<ValueTuplePtr>()->value();
  std::vector<std::vector<int64_t>> paddings_vec;
  for (ValuePtr paddings_elements : paddings) {
    std::vector<ValuePtr> paddings_elements_tuple = paddings_elements->cast<ValueTuplePtr>()->value();
    std::vector<int64_t> paddings_vec_item;
    (void)std::transform(std::begin(paddings_elements_tuple), std::end(paddings_elements_tuple),
                         std::back_inserter(paddings_vec_item),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
    paddings_vec.push_back(paddings_vec_item);
  }

  ShapeVector result_shp;
  size_t length = paddings_vec.size();
  for (size_t i = 0; i < length; ++i) {
    if (paddings_vec[i].size() != 2) {
      MS_LOG(EXCEPTION) << "Paddings 's second dim size is not 2";
    }
    result_shp.push_back(input_shp[i] + paddings_vec[i][0] + paddings_vec[i][1]);
  }
  return std::make_shared<AbstractTensor>(arg->element(), std::make_shared<Shape>(result_shp));
}

AbstractBasePtr InferImplComputeAccidentalHits(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list) {
  // inputs: true_classes, sampled_candidates
  const std::string op_name = primitive->name();
  constexpr size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  auto shape = input->shape();
  if (shape->shape().size() != size_expected) {
    MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 2.";
  }
  ShapeVector indices_shape = {Shape::SHP_ANY};
  ShapeVector min_shape = {1};
  ShapeVector max_shape = {shape->shape()[0] * shape->shape()[1]};

  auto indices =
    std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(indices_shape, min_shape, max_shape));

  auto weights = std::make_shared<AbstractTensor>(kFloat32, indices_shape);
  weights->set_shape(std::make_shared<Shape>(indices_shape, min_shape, max_shape));
  // outputs: indices, ids, weights
  AbstractBasePtrList elements = {indices, indices, weights};
  return std::make_shared<AbstractTuple>(elements);
}
}  // namespace abstract
}  // namespace mindspore
