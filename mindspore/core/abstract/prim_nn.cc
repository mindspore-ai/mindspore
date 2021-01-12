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

#include <cmath>
#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

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
  int64_t h_input = input_shape->shape()[2];
  int64_t w_input = input_shape->shape()[3];

  int64_t window = primitive->GetAttr("window")->cast<Int64ImmPtr>()->value();
  int64_t stride = primitive->GetAttr("stride")->cast<Int64ImmPtr>()->value();
  int64_t padding = primitive->GetAttr("pad")->cast<Int64ImmPtr>()->value();
  int64_t nan_opt = primitive->GetAttr("nan_opt")->cast<Int64ImmPtr>()->value();
  int64_t data_mode = primitive->GetAttr("data_mode")->cast<Int64ImmPtr>()->value();
  int64_t ceil_mode = primitive->GetAttr("ceil_mode")->cast<Int64ImmPtr>()->value();

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

  int64_t h_out = ((h_input + 2 * padding - (window - 1) - 1) / stride) + 1;
  int64_t w_out = ((w_input + 2 * padding - (window - 1) - 1) / stride) + 1;
  ShapeVector shape_out = {input_shape->shape()[0], input_shape->shape()[1], h_out, w_out};
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

AbstractBasePtr InferImplBatchNormGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: five tensors(y_backprop, x, scale, save_mean, save_inv_variance).
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  MS_EXCEPTION_IF_NULL(args_spec_list[2]);
  MS_EXCEPTION_IF_NULL(args_spec_list[3]);

  CheckArgsSize(primitive->name(), args_spec_list, 5);
  auto dx = args_spec_list[1]->Broaden();
  auto dscale = args_spec_list[2]->Broaden();
  auto dbias = args_spec_list[3]->Broaden();
  auto reserve_1 = args_spec_list[4]->Broaden();
  auto reserve_2 = args_spec_list[5]->Broaden();

  AbstractBasePtrList rets = {dx, dscale, dbias, reserve_1, reserve_2};
  return std::make_shared<AbstractTuple>(rets);
}

AbstractBasePtr InferImplReluGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors(y_backprop, x).
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto dout = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto out = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  (void)CheckDtypeSame(op_name, out, dout);
  (void)CheckShapeSame(op_name, out, dout);

  return out->Broaden();
}

AbstractBasePtr InferImplFusedSparseAdam(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  // the output is useless, so we dont have to focus on the output shape
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  MS_EXCEPTION_IF_NULL(args_spec_list[2]);
  MS_EXCEPTION_IF_NULL(args_spec_list[3]);

  auto dx = args_spec_list[1]->Broaden();
  auto dscale = args_spec_list[2]->Broaden();
  auto dbias = args_spec_list[3]->Broaden();

  AbstractBasePtrList rets = {dx, dscale, dbias};
  return std::make_shared<AbstractTuple>(rets);
}

AbstractBasePtr InferImplConv2D(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);

  AbstractTensorPtr input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_x->shape());
  ShapeVector x_shape = input_x->shape()->shape();
  ShapeVector x_min_shape = input_x->shape()->min_shape();
  ShapeVector x_max_shape = input_x->shape()->max_shape();
  (void)CheckMinMaxShape(x_shape, &x_min_shape, &x_max_shape);

  AbstractTensorPtr input_w = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(input_w);
  MS_EXCEPTION_IF_NULL(input_w->shape());
  ShapeVector w_shape = input_w->shape()->shape();
  ShapeVector w_min_shape = input_w->shape()->min_shape();
  ShapeVector w_max_shape = input_w->shape()->max_shape();
  (void)CheckMinMaxShape(w_shape, &w_min_shape, &w_max_shape);

  std::set<std::string> available_data_format{"NCHW", "NHWC"};
  int64_t n_axis = 0;
  int64_t c_axis = 1;
  int64_t h_axis = 2;
  int64_t w_axis = 3;
  auto data_format_ptr = primitive->GetAttr("format");
  std::string data_format = "NCHW";
  if ((data_format_ptr != nullptr) && data_format_ptr->isa<StringImm>()) {
    data_format = data_format_ptr->cast<StringImmPtr>()->value();
    if (available_data_format.find(data_format) == available_data_format.end()) {
      MS_LOG(EXCEPTION) << "Unsupported data format: " << data_format << ". use NCHW or NHWC";
    }
    if (data_format == "NHWC") {
      c_axis = 3;
      h_axis = 1;
      w_axis = 2;
    }
  }

  int64_t group = primitive->GetAttr("group")->cast<Int64ImmPtr>()->value();
  if (group <= 0) {
    MS_LOG(EXCEPTION) << "Invalid group value: " << group << ", should be greater then 0";
  }
  if ((x_shape[c_axis] != Shape::SHP_ANY) && (x_shape[c_axis] % group != 0)) {
    MS_LOG(EXCEPTION) << "x_shape[" << c_axis << "] = " << x_shape[c_axis]
                      << " (channels) must be divisible by group = " << group;
  }

  int64_t out_channel = primitive->GetAttr("out_channel")->cast<Int64ImmPtr>()->value();
  if (out_channel <= 0) {
    MS_LOG(EXCEPTION) << "Invalid out_channel value: " << out_channel << ", should be greater then 0";
  }
  if ((w_shape[0] != Shape::SHP_ANY) && (w_shape[0] != out_channel)) {
    MS_LOG(EXCEPTION) << "w_shape[0] = " << w_shape[0] << " must equal to = " << out_channel;
  }

  int64_t kernel_h = 0;
  int64_t kernel_w = 0;
  ValuePtr kernel_size_attr = primitive->GetAttr("kernel_size");
  MS_EXCEPTION_IF_NULL(kernel_size_attr);
  if (kernel_size_attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> kernel_size_vec = kernel_size_attr->cast<ValueTuplePtr>()->value();
    kernel_h = GetValue<int64_t>(kernel_size_vec[0]);
    kernel_w = GetValue<int64_t>(kernel_size_vec[1]);
  } else {
    int64_t kernel_size = kernel_size_attr->cast<Int64ImmPtr>()->value();
    kernel_h = kernel_size;
    kernel_w = kernel_size;
  }
  if ((w_shape[2] != Shape::SHP_ANY) && (w_shape[2] != kernel_h)) {
    MS_LOG(EXCEPTION) << "weight height, w_shape[2] = " << w_shape[2] << ", must equal to = " << kernel_h;
  }
  if ((w_shape[3] != Shape::SHP_ANY) && (w_shape[3] != kernel_w)) {
    MS_LOG(EXCEPTION) << "weight width, w_shape[3] = " << w_shape[3] << ", must equal to = " << kernel_w;
  }

  int64_t stride_h = 0;
  int64_t stride_w = 0;
  ValuePtr stride_attr = primitive->GetAttr("stride");
  MS_EXCEPTION_IF_NULL(stride_attr);
  if (stride_attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> stride_vec = stride_attr->cast<ValueTuplePtr>()->value();
    stride_h = GetValue<int64_t>(stride_vec[2]);
    stride_w = GetValue<int64_t>(stride_vec[3]);
  } else {
    int64_t stride = stride_attr->cast<Int64ImmPtr>()->value();
    stride_h = stride;
    stride_w = stride;
  }

  int64_t dilation_h = 0;
  int64_t dilation_w = 0;
  ValuePtr dilation_attr = primitive->GetAttr("dilation");
  MS_EXCEPTION_IF_NULL(dilation_attr);
  if (dilation_attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> dilation_vec = dilation_attr->cast<ValueTuplePtr>()->value();
    dilation_h = GetValue<int64_t>(dilation_vec[2]);
    dilation_w = GetValue<int64_t>(dilation_vec[3]);
  } else {
    int64_t dilation = dilation_attr->cast<Int64ImmPtr>()->value();
    dilation_h = dilation;
    dilation_w = dilation;
  }

  std::vector<int64_t> padding;
  ValuePtr padding_attr = primitive->GetAttr("pad");
  MS_EXCEPTION_IF_NULL(padding_attr);
  if (padding_attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> padding_vec = padding_attr->cast<ValueTuplePtr>()->value();
    (void)std::transform(std::begin(padding_vec), std::end(padding_vec), std::back_inserter(padding),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  } else {
    int64_t padding_val = padding_attr->cast<Int64ImmPtr>()->value();
    padding = {padding_val, padding_val, padding_val, padding_val};
  }

  std::set<std::string> available_pad_mode{"pad", "same", "valid"};
  ValuePtr pad_mode_attr = primitive->GetAttr("pad_mode");
  MS_EXCEPTION_IF_NULL(pad_mode_attr);
  auto pad_mode = pad_mode_attr->cast<StringImmPtr>()->value();
  if (available_pad_mode.find(pad_mode) == available_pad_mode.end()) {
    MS_LOG(EXCEPTION) << "Unsupported pad mode: " << pad_mode << ". use pad, same, valid";
  }

  std::function<void(int64_t, int64_t, std::vector<int64_t> &, std::vector<int64_t> &)> pad_function =
    [kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_mode, padding](
      int64_t x_h, int64_t x_w, std::vector<int64_t> &output_hw, std::vector<int64_t> &pad_list) {
      if (pad_mode == "valid") {
        output_hw.push_back(std::ceil(((x_h * 1.0) - dilation_h * (kernel_h - 1)) / stride_h));
        output_hw.push_back(std::ceil(((x_w * 1.0) - dilation_w * (kernel_w - 1)) / stride_w));
        pad_list = {0, 0, 0, 0};
      } else if (pad_mode == "same") {
        output_hw.push_back(std::ceil((x_h * 1.0) / stride_h));
        output_hw.push_back(std::ceil((x_w * 1.0) / stride_w));
        int64_t pad_needed_h = (output_hw[0] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_h;
        pad_needed_h = std::max((int64_t)0, pad_needed_h);
        pad_list.push_back(std::floor(pad_needed_h / 2));
        pad_list.push_back(pad_needed_h - pad_list[0]);
        int64_t pad_needed_w = (output_hw[1] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_w;
        pad_needed_w = std::max((int64_t)0, pad_needed_w);
        pad_list.push_back(std::floor(pad_needed_w / 2));
        pad_list.push_back(pad_needed_w - pad_list[2]);
      } else if (pad_mode == "pad") {
        pad_list = padding;
        output_hw.push_back(std::floor(
          1 + ((x_h * 1.0) + pad_list[0] + pad_list[1] - kernel_h - (kernel_h - 1) * (dilation_h - 1)) / stride_h));
        output_hw.push_back(std::floor(
          1 + ((x_w * 1.0) + pad_list[2] + pad_list[3] - kernel_w - (kernel_w - 1) * (dilation_w - 1)) / stride_w));
      }
    };

  std::vector<int64_t> output_hw;
  std::vector<int64_t> pad_list;
  std::vector<int64_t> output_hw_min;
  std::vector<int64_t> pad_list_min;
  std::vector<int64_t> output_hw_max;
  std::vector<int64_t> pad_list_max;
  pad_function(x_shape[h_axis], x_shape[w_axis], output_hw, pad_list);
  if (x_shape[h_axis] == Shape::SHP_ANY) {
    output_hw[0] = Shape::SHP_ANY;
  }
  if (x_shape[w_axis] == Shape::SHP_ANY) {
    output_hw[1] = Shape::SHP_ANY;
  }
  pad_function(x_min_shape[h_axis], x_min_shape[w_axis], output_hw_min, pad_list_min);
  pad_function(x_max_shape[h_axis], x_max_shape[w_axis], output_hw_max, pad_list_max);

  std::vector<ValuePtr> pad_list_val = {MakeValue(pad_list[0]), MakeValue(pad_list[1]), MakeValue(pad_list[2]),
                                        MakeValue(pad_list[3])};
  primitive->set_attr("pad_list", MakeValue(pad_list_val));
  ShapeVector output_shape;
  ShapeVector output_shape_min;
  ShapeVector output_shape_max;
  if (data_format == "NHWC") {
    output_shape = {x_shape[n_axis], output_hw[0], output_hw[1], out_channel};
    output_shape_min = {x_min_shape[n_axis], output_hw_min[0], output_hw_min[1], out_channel};
    output_shape_max = {x_max_shape[n_axis], output_hw_max[0], output_hw_max[1], out_channel};
  } else {
    output_shape = {x_shape[n_axis], out_channel, output_hw[0], output_hw[1]};
    output_shape_min = {x_min_shape[n_axis], out_channel, output_hw_min[0], output_hw_min[1]};
    output_shape_max = {x_max_shape[n_axis], out_channel, output_hw_max[0], output_hw_max[1]};
  }

  ShapePtr output_shape_ptr = std::make_shared<Shape>(output_shape, output_shape_min, output_shape_max);
  return std::make_shared<AbstractTensor>(input_x->element(), output_shape_ptr);
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

AbstractBasePtr InferImplRelu(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplBpropCut(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  AbstractBasePtrList args_list;
  for (size_t i = 0; i < args_spec_list.size() - 2; i++) {
    args_list.push_back(args_spec_list[i]->Broaden());
  }
  return std::make_shared<AbstractTuple>(args_list);
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
  int64_t begin_norm_axis = CheckAxis(op_name, bna_ptr, -1, SizeToLong(input_rank) - 1);

  ValuePtr bpa_ptr = primitive->GetAttr("begin_params_axis");
  int64_t begin_params_axis = CheckAxis(op_name, bpa_ptr, -1, SizeToLong(input_rank) - 1);
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

  size_t begin_params_axis_u = LongToSize(begin_params_axis);
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
  int64_t count = 1;
  for (std::size_t i = 0; i < x_shape->size(); ++i) {
    auto value_track = x_shape_data[i]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    int64_t e_value = 0;
    if (value_track->isa<Int64Imm>()) {
      e_value = GetValue<int64_t>(value_track);
    } else {
      MS_LOG(EXCEPTION) << "DropOutGenMask input x_shape elements is not int64 or int32, but "
                        << value_track->ToString() << ".";
    }

    if (e_value <= 0) {
      MS_LOG(EXCEPTION) << "DropOutGenMask product of x_shape should be > 0";
    }
    if (std::numeric_limits<int64_t>::max() / count / e_value < 1) {
      MS_LOG(EXCEPTION) << "integer multiply integer overflow";
    }
    count = count * e_value;
  }

  // convert to bytes(8 bits) mask, using round up
  int64_t n128s = count / 128;
  if ((count % 128) != 0) {
    n128s++;
  }
  int64_t bytes_count = n128s * 16;
  std::vector<int64_t> shape_y{bytes_count};

  primitive->set_attr("T", kInt32);
  return std::make_shared<AbstractTensor>(std::make_shared<AbstractScalar>(kAnyValue, kUInt8),
                                          std::make_shared<Shape>(std::vector<int64_t>{shape_y}));
}

AbstractBasePtr InferImplSparseApplyFtrl(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  CheckArgsSize(primitive->name(), args_spec_list, 5);
  AbstractBasePtrList elements;
  for (size_t i = 0; i < 3; ++i) {
    elements.push_back(args_spec_list[i]->Clone()->Broaden());
  }
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplSparseApplyProximalAdagrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const AbstractBasePtrList &args_spec_list) {
  CheckArgsSize(primitive->name(), args_spec_list, 7);
  AbstractBasePtrList elements;
  for (size_t i = 0; i < 2; ++i) {
    elements.push_back(args_spec_list[i]->Clone()->Broaden());
  }
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplSGD(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  CheckArgsSize(primitive->name(), args_spec_list, 6);
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

  auto shape = input->shape();
  if (shape->shape().size() != 3) {
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
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  auto shape = input->shape();
  if (shape->shape().size() != 2) {
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
