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

#include "pipeline/static_analysis/prim.h"
#include "operator/ops.h"
#include "pipeline/static_analysis/utils.h"
#include "pipeline/static_analysis/param_validator.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplPooling(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input_tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  (void)CheckTensorDType(input_tensor, {kFloat16, kFloat32}, "Input 0 of Pooling should be %s");

  ShapePtr input_shape = dyn_cast<Shape>(input_tensor->GetShapeTrack());  // NCHW
  MS_EXCEPTION_IF_NULL(input_shape);
  if (input_shape->shape().size() != 4) {
    MS_LOG(EXCEPTION) << "Pooling input should be a 4-D tensor.";
  }
  int h_input = input_shape->shape()[2];
  int w_input = input_shape->shape()[3];

  int window = primitive->GetAttr("window")->cast<Int32ImmPtr>()->value();
  int stride = primitive->GetAttr("stride")->cast<Int32ImmPtr>()->value();
  int padding = primitive->GetAttr("pad")->cast<Int32ImmPtr>()->value();
  int nan_opt = primitive->GetAttr("nan_opt")->cast<Int32ImmPtr>()->value();
  int data_mode = primitive->GetAttr("data_mode")->cast<Int32ImmPtr>()->value();
  int ceil_mode = primitive->GetAttr("ceil_mode")->cast<Int32ImmPtr>()->value();

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

  std::set<std::string> available_pad_mode{"pad", "same", "valid"};
  auto pad_mode_ptr = primitive->GetAttr("pad_mode");
  if ((pad_mode_ptr != nullptr) && pad_mode_ptr->isa<StringImm>()) {
    auto pad_mode = pad_mode_ptr->cast<StringImmPtr>()->value();
    if (available_pad_mode.find(pad_mode) == available_pad_mode.end()) {
      MS_LOG(EXCEPTION) << "Unsupported pad mode: " << pad_mode << ". use pad, same, valid";
    }
    if (pad_mode == "valid") {
      padding = 0;
    } else if (pad_mode == "same") {
      padding = (window - 1) / 2;
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

  int h_out = ((h_input + 2 * padding - (window - 1) - 1) / stride) + 1;
  int w_out = ((w_input + 2 * padding - (window - 1) - 1) / stride) + 1;
  std::vector<int> shape_out = {input_shape->shape()[0], input_shape->shape()[1], h_out, w_out};
  AbstractBasePtr ret = input_tensor->Broaden();
  ret->set_shape(std::make_shared<Shape>(shape_out));
  return ret;
}

AbstractBasePtr InferImplPoolingGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors(y, dy, x).
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
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

void FusedBatchNormCheckDim(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list) {
  // check dimension, x > 1, others equal 1
  const std::string op_name = primitive->name();
  for (std::size_t i = 0; i < args_spec_list.size(); ++i) {
    AbstractTensorPtr arg = CheckArg<AbstractTensor>(op_name, args_spec_list, i);
    ShapePtr arg_shape = dyn_cast<Shape>(arg->GetShapeTrack());
    if (arg_shape == nullptr) {
      MS_LOG(EXCEPTION) << op_name << " type of args[" << i << "] should be Shape, but " << arg->ToString();
    }

    if (i == 0) {
      if (arg_shape->shape().size() < 2) {
        MS_LOG(EXCEPTION) << op_name << " shape of args[" << i
                          << "] should be TensorShape with dimension greater than 1, but shape: "
                          << arg_shape->ToString();
      }
      continue;
    }

    if (arg_shape->shape().size() != 1) {
      MS_LOG(EXCEPTION) << op_name << " shape of args[" << i
                        << "] should be TensorShape with dimension: 1, but shape: " << arg_shape->ToString();
    }
  }
}

AbstractBasePtr InferImplFusedBatchNorm(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  // Inputs: five tensors(x, gamma, beta, mean, variance).
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 5);
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  MS_LOG(DEBUG) << "InferImplFusedBatchNorm args0:" << args_spec_list[0]->ToString()
                << ", arg1:" << args_spec_list[1]->ToString();
  FusedBatchNormCheckDim(primitive, args_spec_list);

  auto input = args_spec_list[0];
  auto input_shape = dyn_cast<Shape>(input->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(input_shape);
  const auto &input_shape_list = input_shape->shape();
  if (input_shape_list.size() < 2) {
    MS_LOG(EXCEPTION) << "Input shape size should >= 2.";
  }

  for (size_t i = 1; i < args_spec_list.size(); ++i) {
    auto arg_shape = dyn_cast<Shape>(args_spec_list[i]->GetShapeTrack());
    MS_EXCEPTION_IF_NULL(arg_shape);
    const auto &arg_shape_list = arg_shape->shape();
    if (arg_shape_list.size() < 1) {
      MS_LOG(EXCEPTION) << "Arg shape size should >= 1.";
    }
    if (arg_shape_list[0] != input_shape_list[1]) {
      MS_LOG(EXCEPTION) << op_name << " size of tensor param[" << i << "](which is " << arg_shape_list[0]
                        << ") should match the second dimension of tensor"
                           " param[0](which is "
                        << input_shape_list[1] << ").";
    }
  }
  auto input_tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  (void)CheckTensorDType(input_tensor, {kFloat16, kFloat32}, "param 0 of FusedBatchNorm should be %s");

  AbstractTensorPtrList tensorPtrList = std::vector<AbstractTensorPtr>();
  for (size_t i = 1; i < args_spec_list.size(); ++i) {
    auto param = CheckArg<AbstractTensor>(op_name, args_spec_list, i);
    tensorPtrList.push_back(param);
  }
  (void)CheckTensorsDTypeSame(tensorPtrList, {kFloat16, kFloat32}, "param 1 to 4 of FusedBatchNorm should be %s");

  // check validity;
  auto epsilon_value = primitive->GetAttr("epsilon");
  auto momentum_value = primitive->GetAttr("momentum");
  MS_EXCEPTION_IF_NULL(epsilon_value);
  MS_EXCEPTION_IF_NULL(momentum_value);
  if (!epsilon_value->isa<FP32Imm>() || !momentum_value->isa<FP32Imm>()) {
    MS_LOG(EXCEPTION) << "expect epsilon and momentum be float, but: epsilon: " << epsilon_value->ToString()
                      << ", momentum: " << momentum_value->ToString();
  }

  auto epsilon = epsilon_value->cast<FP32ImmPtr>()->value();
  auto momentum = momentum_value->cast<FP32ImmPtr>()->value();

  if (epsilon > 1.0f || epsilon <= 0.0f) {
    MS_LOG(EXCEPTION) << "expect epsilon is greater than 0 and less or equal than 1, but epsilon: " << epsilon;
  }
  if (momentum > 1.0f || momentum < 0.0f) {
    MS_LOG(EXCEPTION) << "expect momentum is great or equal than 0 and less or equal than 1, but epsilon: " << momentum;
  }

  // Outputs: y, running_mean, running_variance, save_mean, save_inv_variance.
  AbstractBasePtr y = input->Broaden();
  AbstractBasePtr other = args_spec_list[1]->Broaden();
  MS_LOG(DEBUG) << "output y: " << y->ToString() << ", other: " << other->ToString();

  AbstractBasePtrList elements = {y, other, other, other, other};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplFusedBatchNormGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  // Inputs: five tensors(y_backprop, x, scale, save_mean, save_inv_variance).
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  MS_EXCEPTION_IF_NULL(args_spec_list[2]);
  MS_EXCEPTION_IF_NULL(args_spec_list[3]);

  CheckArgsSize(primitive->name(), args_spec_list, 5);
  auto dx = args_spec_list[1]->Broaden();
  auto dscale = args_spec_list[2]->Broaden();
  auto dbias = args_spec_list[3]->Broaden();

  AbstractBasePtrList rets = {dx, dscale, dbias};
  return std::make_shared<AbstractTuple>(rets);
}

AbstractBasePtr InferImplReluGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors(y_backprop, x).
  CheckArgsSize(primitive->name(), args_spec_list, 2);
  return args_spec_list[1]->Broaden();
}

AbstractBasePtr InferImplConv2DBackpropInput(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors(doutput, input, filters).
  CheckArgsSize(primitive->name(), args_spec_list, 3);
  return args_spec_list[1]->Broaden();
}

AbstractBasePtr InferImplConv2DBackpropFilter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors(inputs, filter, doutput).
  CheckArgsSize(primitive->name(), args_spec_list, 3);
  return args_spec_list[2]->Broaden();
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
  std::vector<int> y_dims = shape_y->shape();
  if (y_dims.size() < 2) {
    MS_LOG(EXCEPTION) << primitive->name() << " input y backprop, dim should >= 2, while " << y_dims.size() << ".";
  }
  std::vector<int> bias_dims = {y_dims[1]};
  ShapePtr ret_shape = std::make_shared<Shape>(bias_dims);
  AbstractBasePtr ret = args_spec_list[0]->Broaden();
  ret->set_shape(ret_shape);
  return ret;
}

AbstractBasePtr InferImplRelu(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplZerosLikeTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplFakeBprop(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplLayerNorm(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors(x, gamma, beta).
  // outputs: y, mean, variance
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_shape = input_x->shape();
  auto const &input_shape_list = input_shape->shape();
  const size_t input_rank = input_shape_list.size();
  if (input_rank == 0) {
    MS_LOG(EXCEPTION) << "input_rank should not be zero";
  }

  // begin_norm_axis and begin_params_axis should be smaller than the size of input_x and >= -1
  ValuePtr bna_ptr = primitive->GetAttr("begin_norm_axis");
  int begin_norm_axis = CheckAxis(op_name, bna_ptr, -1, SizeToInt(input_rank) - 1);

  ValuePtr bpa_ptr = primitive->GetAttr("begin_params_axis");
  int begin_params_axis = CheckAxis(op_name, bpa_ptr, -1, SizeToInt(input_rank) - 1);
  begin_params_axis = GetPositiveAxis(begin_params_axis, input_rank);

  // the beta and gama shape should be x_shape[begin_params_axis:]
  auto tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto gamma = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto beta = CheckArg<AbstractTensor>(op_name, args_spec_list, 2);
  (void)CheckTensorDType(tensor, {kFloat16, kFloat32}, "input 0 of LayerNorm should be %s");
  (void)CheckTensorDType(gamma, {kFloat16, kFloat32}, "input 1 of LayerNorm should be %s");
  (void)CheckTensorDType(beta, {kFloat16, kFloat32}, "input 2 of LayerNorm should be %s");
  auto gamma_shape = dyn_cast<Shape>(gamma->BuildShape());
  auto beta_shape = dyn_cast<Shape>(beta->BuildShape());
  MS_EXCEPTION_IF_NULL(gamma_shape);
  MS_EXCEPTION_IF_NULL(beta_shape);

  auto const &gamma_shape_list = gamma_shape->shape();
  auto const &beta_shape_list = beta_shape->shape();
  if (gamma_shape_list.empty() || beta_shape_list.empty()) {
    MS_LOG(EXCEPTION) << "LayerNorm evaluator gamma or beta is a AbstractScalar that is not support.";
  }

  size_t begin_params_axis_u = IntToSize(begin_params_axis);
  if ((begin_params_axis_u > input_shape_list.size()) ||
      (gamma_shape_list.size() + begin_params_axis_u < input_shape_list.size()) ||
      (beta_shape_list.size() + begin_params_axis_u < input_shape_list.size())) {
    MS_LOG(EXCEPTION) << "Gamma and beta shape get wrong size.";
  }
  for (size_t i = begin_params_axis_u; i < input_shape_list.size(); ++i) {
    size_t gamma_beta_shape_dim = i - begin_params_axis_u;
    if ((gamma_shape_list[gamma_beta_shape_dim] != input_shape_list[i]) ||
        (beta_shape_list[gamma_beta_shape_dim] != input_shape_list[i])) {
      MS_LOG(EXCEPTION) << "Gamma or beta shape not match input shape, input_shape=" << input_shape->ToString()
                        << ", gamma_shape=" << gamma_shape->ToString() << ", beta_shape=" << beta_shape->ToString();
    }
  }

  auto mean_var_shape_value = input_shape->shape();
  if (begin_norm_axis == -1) {
    mean_var_shape_value[input_rank - 1] = 1;
  } else {
    for (size_t i = begin_norm_axis; i < input_rank; ++i) {
      mean_var_shape_value[i] = 1;
    }
  }

  auto mean = input_x->Broaden();
  mean->set_shape(std::make_shared<Shape>(mean_var_shape_value));
  auto var = input_x->Broaden();
  var->set_shape(std::make_shared<Shape>(mean_var_shape_value));

  AbstractBasePtrList args_list({input_x->Broaden(), mean, var});
  return std::make_shared<AbstractTuple>(args_list);
}

AbstractBasePtr InferImplLayerNormGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: five tensors(y_backprob, x, variance, mean, gamma).
  // Outputs: x_backprob, gamma_backprob, beta_backprob
  CheckArgsSize(primitive->name(), args_spec_list, 5);

  auto x_backprob = args_spec_list[0]->Broaden();
  auto gamma_backprob = args_spec_list[4]->Broaden();
  auto beta_backprob = args_spec_list[4]->Broaden();

  AbstractBasePtrList args_list({x_backprob, gamma_backprob, beta_backprob});
  return std::make_shared<AbstractTuple>(args_list);
}

AbstractBasePtr InferImplDropoutGenMask(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple and a tensor.
  // Outputs: mask.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr x_shape = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  AbstractTensorPtr keep_prob = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);

  TypePtr prob_type = keep_prob->element()->BuildType();
  if ((prob_type->type_id() != kNumberTypeFloat16) && (prob_type->type_id() != kNumberTypeFloat32)) {
    MS_LOG(EXCEPTION) << op_name << " keep_prob type should be float16 or float32, but " << prob_type->ToString()
                      << ".";
  }

  auto x_shape_data = x_shape->elements();
  int count = 1;
  for (std::size_t i = 0; i < x_shape->size(); ++i) {
    auto value_track = x_shape_data[i]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    if (!value_track->isa<Int32Imm>()) {
      MS_LOG(EXCEPTION) << "DropOutGenMask input x_shape elements is not int32, but " << value_track->ToString() << ".";
    }

    int e_value = GetValue<int>(value_track);
    if (e_value <= 0) {
      MS_LOG(EXCEPTION) << "DropOutGenMask product of x_shape should be > 0";
    }
    if (std::numeric_limits<int>::max() / count / e_value < 1) {
      MS_LOG(EXCEPTION) << "integer multiply integer overflow";
    }
    count = count * e_value;
  }

  // convert to bytes(8 bits) mask, using round up
  int n128s = count / 128;
  if ((count % 128) != 0) {
    n128s++;
  }
  int bytes_count = n128s * 16;
  std::vector<int> shape_y{bytes_count};

  primitive->set_attr("T", kInt32);
  return std::make_shared<AbstractTensor>(std::make_shared<AbstractScalar>(kAnyValue, kUInt8),
                                          std::make_shared<Shape>(std::vector<int>{shape_y}));
}
}  // namespace abstract
}  // namespace mindspore
