/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "mindspore/core/ops/core_ops.h"
#include "ir/func_graph.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "proto/onnx.pb.h"
#include "utils/check_convert_utils.h"
#include "utils/hash_map.h"
#include "utils/ms_context.h"

namespace mindspore {
const int ONNX_VERSION = 11;
const int kZeroNum = 0;
const int kOneNum = 1;
const int kTwoNum = 2;
const int kThreeNum = 3;
const int kFourNum = 4;
const int kFiveNum = 5;
const int kSixNum = 6;
const int64_t kOneNumLong = 1;
const float weight_for_mul = 0.5;
enum OpMergeMode {
  OP_MERGE_UNDEFINED = 0,            // undefined behavior
  OP_MERGE_IGNORE = 1,               // indicate an input op merged into other op in compute node list
  OP_MERGE_CONV = 2,                 // indicate `MindSpore Conv + BiasAdd` --> `ONNX Conv`
  OP_MERGE_GEMM = 3,                 // indicate `MindSpore MatMul + BiasAdd` --> `ONNX Gemm`
  OP_MERGE_BATCH_NORM = 4,           // indicate `MindSpore BatchNorm(x)[0]` --> `ONNX Batch Normalization`
  OP_MERGE_MAXPOOL_WITH_ARGMAX = 5,  // indicate `MindSpore MaxPoolWithArgmax(x)[0]` --> `ONNX MaxPool`
  OP_MERGE_LAYER_NORM = 6,           // indicate `MindSpore LayerNorm(x)[0]` --> `ONNX MeanVarianceNormalization`
  OP_MERGE_CONV2D_TRANSPOSE = 7,     // indicate `MindSpore ConvTranspose + BiasAdd` --> `ONNX ConvTranspose`
  OP_MERGE_DYNAMIC_GRU_V2 = 8,       // indicate `MindSpore DynamicGRUV2(...)[0]` --> `ONNX GRU`
};

struct OpMergedInfo {
  OpMergeMode mode = OP_MERGE_UNDEFINED;
  int referred_count = 0;
};

using GenAttrFuncType =
  std::function<void(ValuePtr, onnx::AttributeProto_AttributeType, onnx::AttributeProto *, const PrimitivePtr &)>;

bool IsIgnoredIdentityNode(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad);
}

/*
 If true, the node should not be referenced by anything and should not be contributing to any
 ref counts itself
 */
bool IsZeroRefcountNode(const AnfNodePtr &node) { return HasAbstractMonad(node) || IsIgnoredIdentityNode(node); }

// Ideally this should be applied to every node->input() call, not only inside GetNodeInputName
static AnfNodePtr GetRealInput(const AnfNodePtr &origin_input) {
  AnfNodePtr input = origin_input;
  while (IsIgnoredIdentityNode(input)) {
    input = input->cast<CNodePtr>()->inputs().at(1);
  }
  return input;
}

template <typename T, size_t rep_cnt = 0>
void SetAttrValueToProto(const ValuePtr &value, onnx::AttributeProto_AttributeType attr_type,
                         onnx::AttributeProto *const attr_proto, const PrimitivePtr &) {
  auto casted_value = dyn_cast<T>(value);
  if (casted_value == nullptr) {
    MS_LOG(EXCEPTION) << "Cast value " << value->ToString() << " to type T failed.";
  }
  auto attr_value = casted_value->value();
  switch (attr_type) {
    case onnx::AttributeProto_AttributeType_INT:
      attr_proto->set_i(static_cast<::google::protobuf::int64>(attr_value));
      break;
    case onnx::AttributeProto_AttributeType_FLOAT:
      attr_proto->set_f(static_cast<float>(attr_value));
      break;
    case onnx::AttributeProto_AttributeType_INTS:
      for (size_t i = 0; i < rep_cnt; ++i) {
        attr_proto->add_ints(static_cast<::google::protobuf::int64>(attr_value));
      }
      break;
    case onnx::AttributeProto_AttributeType_FLOATS:
      for (size_t i = 0; i < rep_cnt; ++i) {
        attr_proto->add_floats(static_cast<float>(attr_value));
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "Convert attribute fail, unexpected ONNX type " << attr_type;
  }
  attr_proto->set_type(attr_type);
}

template <size_t beg_idx = 0>
void SetAttrTupleValueToProto(const ValuePtr &value, onnx::AttributeProto_AttributeType attr_type,
                              onnx::AttributeProto *const attr_proto, const PrimitivePtr &) {
  auto tuple_ptr = dyn_cast<ValueTuple>(value);
  if (tuple_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Cast value from type " << value->type_name() << " to ValueTuple failed.";
  }
  switch (attr_type) {
    case onnx::AttributeProto_AttributeType_INTS:
      for (size_t i = beg_idx; i < tuple_ptr->size(); ++i) {
        attr_proto->add_ints(GetValue<int64_t>((*tuple_ptr)[i]));
      }
      break;
    case onnx::AttributeProto_AttributeType_INT:
      attr_proto->set_i(GetValue<int64_t>((*tuple_ptr)[beg_idx]));
      break;
    case onnx::AttributeProto_AttributeType_FLOATS:
      for (size_t i = beg_idx; i < tuple_ptr->size(); ++i) {
        attr_proto->add_floats(GetValue<float>((*tuple_ptr)[i]));
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "Convert attribute fail, unexpected ONNX type " << attr_type;
  }
  attr_proto->set_type(attr_type);
}

void SetPoolingPadMode(const ValuePtr &value, onnx::AttributeProto_AttributeType,
                       onnx::AttributeProto *const attr_proto, const PrimitivePtr &) {
  attr_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  int64_t attr_value;
  CheckAndConvertUtils::GetPadModEnumValue(value, &attr_value, true);
  if (attr_value == PadMode::VALID) {
    attr_proto->set_s("VALID");
  } else {
    attr_proto->set_s("SAME_UPPER");
  }
}

void SetConvPadding(const ValuePtr &value, onnx::AttributeProto_AttributeType, onnx::AttributeProto *const attr_proto,
                    const PrimitivePtr &prim) {
  attr_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  int64_t attr_value;
  CheckAndConvertUtils::GetPadModEnumValue(value, &attr_value);
  if (attr_value == PadMode::VALID) {
    attr_proto->set_s("VALID");
  } else if (attr_value == PadMode::SAME) {
    attr_proto->set_s("SAME_UPPER");
  } else {  // pad_mode is 'pad', use attribute 'pad_list' to fill ONNX attribute 'pads'
    attr_proto->set_name("pads");
    SetAttrTupleValueToProto(prim->GetAttr("pad_list"), onnx::AttributeProto_AttributeType_INTS, attr_proto, prim);
  }
}

void SetConvTransposePadding(const ValuePtr &value, onnx::AttributeProto_AttributeType,
                             onnx::AttributeProto *const attr_proto, const PrimitivePtr &prim) {
  attr_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  int64_t attr_value;
  CheckAndConvertUtils::GetPadModEnumValue(value, &attr_value);
  if (attr_value == PadMode::VALID) {
    attr_proto->set_s("VALID");
  } else if (attr_value == PadMode::SAME) {
    attr_proto->set_s("SAME_LOWER");
  } else {  // pad_mode is 'pad', use attribute 'pad_list' to fill ONNX attribute 'pads'
    attr_proto->set_name("pads");
    SetAttrTupleValueToProto(prim->GetAttr("pad_list"), onnx::AttributeProto_AttributeType_INTS, attr_proto, prim);
  }
}

PrimitivePtr GetPrimitive(const CNodePtr &node) {
  AnfNodePtr op = node->input(kZeroNum);
  auto op_value = dyn_cast<ValueNode>(op);
  MS_EXCEPTION_IF_NULL(op_value);
  auto prim = dyn_cast<Primitive>(op_value->value());
  MS_EXCEPTION_IF_NULL(prim);
  return prim;
}

template <typename T>
T GetOpAttribute(const CNodePtr &node, const std::string &name) {
  ValuePtr attr = GetPrimitive(node)->GetAttr(name);
  return GetValue<T>(attr);
}

template <typename T>
std::shared_ptr<T> GetOpAttributePtr(const CNodePtr &node, const std::string &name) {
  ValuePtr attr = GetPrimitive(node)->GetAttr(name);
  auto result = dyn_cast<T>(attr);
  MS_EXCEPTION_IF_NULL(result);
  return result;
}

std::string MakeOutputName(const std::string &node_name, int output_index) {
  return node_name + "_" + std::to_string(output_index);
}

int64_t RavelIndex(const std::vector<int64_t> &index, const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_CHECK_FAIL(index.size() <= shape.size(), "Index ndims must be <= shape ndims");
  int64_t result = 0;
  int64_t stride = 1;
  for (size_t i = 0; i < shape.size() - index.size(); ++i) {
    stride *= shape[shape.size() - 1 - i];
  }
  for (size_t i = 0; i < index.size(); ++i) {
    size_t rev_i = index.size() - 1 - i;
    result += index[rev_i] * stride;
    stride *= shape[rev_i];
  }
  return result;
}

namespace fp16 {
uint32_t FieldMask(unsigned int field_size) {
  const unsigned int BYTE_SIZE = 8;
  uint32_t mask = std::numeric_limits<uint32_t>::max();
  return mask >> (BYTE_SIZE * sizeof(mask) - field_size);
}

uint32_t ExponentBias(unsigned int exponent_size) { return (1U << (exponent_size - 1U)) - 1U; }

uint32_t Fp32ToFp16(float value) {
  const unsigned int FP32_M = 23;
  const unsigned int FP32_E = 32 - 1 - FP32_M;
  const unsigned int FP16_M = 10;
  const unsigned int FP16_E = 16 - 1 - FP16_M;

  uint32_t fp32_bits;
  auto ret = memcpy_s(reinterpret_cast<std::byte *>(&fp32_bits), sizeof(fp32_bits),
                      reinterpret_cast<std::byte *>(&value), sizeof(value));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Set data memcpy_s failed, ret = " << ret;
  }

  uint32_t mantissa = fp32_bits & FieldMask(FP32_M);
  uint32_t fp32_exp_mask = FieldMask(FP32_E);
  uint32_t fp32_exponent = (fp32_bits >> FP32_M) & fp32_exp_mask;
  if (fp32_exponent == fp32_exp_mask) {
    MS_LOG(EXCEPTION) << "Tried to convert inf or nan to float16: " << value;
  }
  uint32_t sign = fp32_bits >> (FP32_E + FP32_M);

  uint32_t fp16_bits = 0;
  fp16_bits |= sign << (FP16_E + FP16_M);
  uint32_t fp16_exponent = 0;
  if (fp32_exponent != 0) {
    fp16_exponent = fp32_exponent - ExponentBias(FP32_E) + ExponentBias(FP16_E);
  }
  if (fp16_exponent >= FieldMask(FP16_E)) {  // inf, nan (==), underflow, or overflow (>)
    MS_LOG(EXCEPTION) << "Conversion of " << value << " to float16 resulted in exponent overflow or underflow";
  }
  fp16_bits |= fp16_exponent << FP16_M;
  fp16_bits |= mantissa >> (FP32_M - FP16_M);

  return fp16_bits;
}
}  // namespace fp16

void AddFloatScalarInitializer(const std::string &name, float value, onnx::TensorProto_DataType type,
                               onnx::GraphProto *graph_proto) {
  onnx::TensorProto *initializer = graph_proto->add_initializer();
  initializer->set_name(name);
  if (type == onnx::TensorProto_DataType_FLOAT16) {
    uint32_t fp16 = fp16::Fp32ToFp16(value);
    initializer->add_int32_data(static_cast<int32_t>(fp16));
  } else if (type == onnx::TensorProto_DataType_FLOAT) {
    initializer->add_float_data(value);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << type;
  }
  initializer->set_data_type(type);
}

void AddInt64Tensor1DInitializer(const std::string &name, const std::vector<int64_t> &values,
                                 onnx::GraphProto *graph_proto) {
  onnx::TensorProto *initializer = graph_proto->add_initializer();
  initializer->set_name(name);
  initializer->set_data_type(onnx::TensorProto_DataType_INT64);
  initializer->add_dims(static_cast<int64_t>(values.size()));
  for (auto value : values) {
    initializer->add_int64_data(value);
  }
}

void AddFloatTensor1DInitializer(const std::string &name, const std::vector<float> &values,
                                 onnx::TensorProto_DataType type, onnx::GraphProto *graph_proto) {
  onnx::TensorProto *initializer = graph_proto->add_initializer();
  initializer->set_name(name);
  initializer->add_dims(static_cast<int64_t>(values.size()));
  if (type == onnx::TensorProto_DataType_FLOAT16) {
    for (auto value : values) {
      uint32_t fp16 = fp16::Fp32ToFp16(value);
      initializer->add_int32_data(static_cast<int32_t>(fp16));
    }
  } else if (type == onnx::TensorProto_DataType_FLOAT) {
    for (auto value : values) {
      initializer->add_float_data(value);
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << type;
  }
  initializer->set_data_type(type);
}

void AddOp(const std::string &type, const std::vector<std::string> &inputs, const std::vector<std::string> &outputs,
           onnx::GraphProto *graph_proto) {
  onnx::NodeProto *op = graph_proto->add_node();
  op->set_op_type(type);
  op->set_name(outputs.at(0) + type);
  for (const auto &input : inputs) {
    op->add_input(input);
  }
  for (const auto &output : outputs) {
    op->add_output(output);
  }
}

void AddClipOp(const std::string &input, const std::string &output, float min, float max,
               onnx::TensorProto_DataType type, onnx::GraphProto *graph_proto) {
  auto min_input_name = output + "__min_initializer";
  AddFloatScalarInitializer(min_input_name, min, type, graph_proto);

  auto max_input_name = output + "__max_initializer";
  AddFloatScalarInitializer(max_input_name, max, type, graph_proto);

  AddOp("Clip", {input, min_input_name, max_input_name}, {output}, graph_proto);
}

void AddSliceOp(const std::string &input, const std::string &output, const std::vector<int64_t> &start,
                const std::vector<int64_t> &end, const std::vector<int64_t> &axis, const std::vector<int64_t> &step,
                onnx::GraphProto *graph_proto) {
  auto starts_name = output + "__starts_initializer";
  AddInt64Tensor1DInitializer(starts_name, start, graph_proto);

  auto ends_name = output + "__ends_initializer";
  AddInt64Tensor1DInitializer(ends_name, end, graph_proto);

  auto axes_name = output + "__axes_initializer";
  AddInt64Tensor1DInitializer(axes_name, axis, graph_proto);

  auto steps_name = output + "__steps_initializer";
  AddInt64Tensor1DInitializer(steps_name, step, graph_proto);

  AddOp("Slice", {input, starts_name, ends_name, axes_name, steps_name}, {output}, graph_proto);
}

void AddSplitOp(const std::string &input, const std::vector<std::string> &outputs, const std::vector<int64_t> &split,
                int64_t axis, onnx::GraphProto *graph_proto) {
  if (outputs.size() != split.size()) {
    MS_LOG(EXCEPTION) << "Number of splits and number of outputs do not match";
  }

  onnx::NodeProto *split_proto = graph_proto->add_node();
  std::string op_type = "Split";
  split_proto->set_op_type(op_type);
  split_proto->set_name(outputs.at(0) + op_type);
  split_proto->add_input(input);
  for (const auto &output : outputs) {
    split_proto->add_output(output);
  }
  onnx::AttributeProto *axis_attr_proto = split_proto->add_attribute();
  axis_attr_proto->set_name("axis");
  axis_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  axis_attr_proto->set_i(axis);
  onnx::AttributeProto *split_attr_proto = split_proto->add_attribute();
  split_attr_proto->set_name("split");
  split_attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
  for (int64_t n : split) {
    split_attr_proto->add_ints(n);
  }
}

void AddExpandOp(const std::string &input, const std::string &output, const std::vector<int64_t> &shape,
                 onnx::GraphProto *graph_proto) {
  onnx::NodeProto *expand_node_proto = graph_proto->add_node();
  expand_node_proto->set_op_type("Expand");
  expand_node_proto->set_name(output + "_Expand");
  expand_node_proto->add_input(input);
  auto shape_name = output + "_expand_shape_initializer";
  AddInt64Tensor1DInitializer(shape_name, shape, graph_proto);
  expand_node_proto->add_input(shape_name);
  expand_node_proto->add_output(output);
}

void AddReshapeOp(const std::string &input, const std::string &output, const std::vector<int64_t> &shape,
                  onnx::GraphProto *graph_proto) {
  auto shape_name = output + "__shape_initializer";
  AddInt64Tensor1DInitializer(shape_name, shape, graph_proto);
  AddOp("Reshape", {input, shape_name}, {output}, graph_proto);
}

onnx::TensorProto *AddConstantOfShapeOp(const std::string &shape, const std::string &output,
                                        onnx::GraphProto *graph_proto) {
  onnx::NodeProto *op = graph_proto->add_node();
  std::string op_type = "ConstantOfShape";
  op->set_op_type(op_type);
  op->set_name(output + op_type);
  op->add_input(shape);
  op->add_output(output);
  onnx::AttributeProto *value_attr = op->add_attribute();
  value_attr->set_name("value");
  value_attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *value_proto = value_attr->mutable_t();
  value_proto->add_dims(1);
  return value_proto;
}

void AddCastOp(const std::string &input, const std::string &output, onnx::TensorProto_DataType target_type,
               onnx::GraphProto *graph_proto) {
  onnx::NodeProto *node_proto = graph_proto->add_node();
  std::string op_type = "Cast";
  node_proto->set_op_type(op_type);
  node_proto->set_name(output + op_type);
  node_proto->add_input(input);
  node_proto->add_output(output);

  onnx::AttributeProto *target_type_attr = node_proto->add_attribute();
  target_type_attr->set_name("to");
  target_type_attr->set_type(onnx::AttributeProto_AttributeType_INT);
  target_type_attr->set_i(target_type);
}

void AddReduceOp(const std::string &op_type, const std::string &input, const std::string &output,
                 const std::vector<int64_t> &axes, bool keepdims, onnx::GraphProto *graph_proto) {
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_name(output + op_type);
  node_proto->set_op_type(op_type);
  node_proto->add_input(input);
  node_proto->add_output(output);

  onnx::AttributeProto *keep_dims_proto = node_proto->add_attribute();
  keep_dims_proto->set_name("keepdims");
  keep_dims_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  keep_dims_proto->set_i(static_cast<int64_t>(keepdims));

  onnx::AttributeProto *axes_proto = node_proto->add_attribute();
  axes_proto->set_name("axes");
  axes_proto->set_type(onnx::AttributeProto_AttributeType_INTS);

  for (auto axis : axes) {
    axes_proto->add_ints(axis);
  }
}

void AddMeanVarianceNormalizationOp(const std::string &input, const std::string &gamma, const std::string &beta,
                                    const std::string &output, const std::vector<int64_t> &axes, float epsilon,
                                    const std::vector<int64_t> &input_shape, onnx::TensorProto_DataType input_type,
                                    onnx::GraphProto *graph_proto) {
  auto input_name = output + "_input";
  AddCastOp(input, input_name, onnx::TensorProto_DataType_FLOAT, graph_proto);
  auto gamma_name = output + "_gamma";
  AddCastOp(gamma, gamma_name, onnx::TensorProto_DataType_FLOAT, graph_proto);
  auto beta_name = output + "_beta";
  AddCastOp(beta, beta_name, onnx::TensorProto_DataType_FLOAT, graph_proto);

  // MeanVarianceNormalization is replaced with equivalent ops because it is not supported by CUDAExecutionProvider
  auto meanvariancenormal_node_name = output + "_normalized";

  auto mean_name = output + "_mean";
  AddReduceOp("ReduceMean", input_name, mean_name, axes, true, graph_proto);
  auto centered_name = output + "_centered";
  AddOp("Sub", {input_name, mean_name}, {centered_name}, graph_proto);

  auto sqsum_name = output + "_sqsum";
  AddReduceOp("ReduceSumSquare", centered_name, sqsum_name, axes, true, graph_proto);
  float reduce_size = std::accumulate(axes.begin(), axes.end(), 1.0f,
                                      [&input_shape](auto acc, auto axis) { return acc * input_shape[axis]; });
  auto reduce_size_name = output + "_reduce_size";
  AddFloatScalarInitializer(reduce_size_name, reduce_size, onnx::TensorProto_DataType_FLOAT, graph_proto);
  auto variance_name = output + "_variance";
  AddOp("Div", {sqsum_name, reduce_size_name}, {variance_name}, graph_proto);

  auto epsilon_name = output + "_epsilon";
  AddFloatScalarInitializer(epsilon_name, epsilon, onnx::TensorProto_DataType_FLOAT, graph_proto);
  auto variance_with_epsilon_name = output + "_variance_with_epsilon";
  AddOp("Add", {variance_name, epsilon_name}, {variance_with_epsilon_name}, graph_proto);
  auto std_name = output + "_std";
  AddOp("Sqrt", {variance_with_epsilon_name}, {std_name}, graph_proto);

  AddOp("Div", {centered_name, std_name}, {meanvariancenormal_node_name}, graph_proto);

  // Add mul and add node
  auto mul_node_name = output + "_rescaled";
  AddOp("Mul", {meanvariancenormal_node_name, gamma_name}, {mul_node_name}, graph_proto);

  // add beta
  auto add_node_name = output;
  if (input_type == onnx::TensorProto_DataType_FLOAT16) {
    add_node_name += "_shifted";
  }
  AddOp("Add", {mul_node_name, beta_name}, {add_node_name}, graph_proto);

  if (input_type == onnx::TensorProto_DataType_FLOAT16) {
    AddCastOp(add_node_name, output, onnx::TensorProto_DataType_FLOAT16, graph_proto);
  }
}

void AddConcatOp(const std::vector<std::string> &inputs, const std::string &output, int axis,
                 onnx::GraphProto *graph_proto) {
  onnx::NodeProto *concat_proto = graph_proto->add_node();
  auto op_type = "Concat";
  concat_proto->set_op_type(op_type);
  concat_proto->set_name(output + op_type);
  for (const auto &input : inputs) {
    concat_proto->add_input(input);
  }
  concat_proto->add_output(output);
  onnx::AttributeProto *axis_proto = concat_proto->add_attribute();
  axis_proto->set_name("axis");
  axis_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  axis_proto->set_i(axis);
}

void ConvertBoxesToXywh(const std::string &startpoints, const std::string &endpoints, const std::string &centerpoints,
                        const std::string &dimensions, onnx::TensorProto_DataType type, onnx::GraphProto *graph_proto) {
  auto coord_sums_name = centerpoints + "__to_div";
  AddOp("Add", {startpoints, endpoints}, {coord_sums_name}, graph_proto);
  auto two_name = centerpoints + "__two_initializer";
  AddFloatScalarInitializer(two_name, 2.0f, type, graph_proto);
  AddOp("Div", {coord_sums_name, two_name}, {centerpoints}, graph_proto);

  auto coord_diffs_name = dimensions + "__to_add";
  AddOp("Sub", {endpoints, startpoints}, {coord_diffs_name}, graph_proto);
  auto one_name = dimensions + "__one_initializer";
  AddFloatScalarInitializer(one_name, 1.0f, type, graph_proto);
  AddOp("Add", {coord_diffs_name, one_name}, {dimensions}, graph_proto);
}

void ConvertBoxesToXyxy(const std::string &centerpoints, const std::string &dimensions, const std::string &startpoints,
                        const std::string &endpoints, onnx::TensorProto_DataType type, onnx::GraphProto *graph_proto) {
  auto half_name = startpoints + "__half_initializer";
  AddFloatScalarInitializer(half_name, 0.5f, type, graph_proto);

  auto half_dim_name = startpoints + "__half_dim";
  auto half_dim_to_sub_name = startpoints + "__to_sub";
  AddOp("Mul", {dimensions, half_name}, {half_dim_to_sub_name}, graph_proto);
  AddOp("Sub", {half_dim_to_sub_name, half_name}, {half_dim_name}, graph_proto);

  AddOp("Sub", {centerpoints, half_dim_name}, {startpoints}, graph_proto);
  AddOp("Add", {centerpoints, half_dim_name}, {endpoints}, graph_proto);
}

void ClipPointsComponent(const std::string &points, const std::string &clipped, float max, int64_t component_idx,
                         onnx::TensorProto_DataType type, onnx::GraphProto *graph_proto) {
  auto res_to_clip_name = clipped + "__clip";
  AddSliceOp(points, res_to_clip_name, {component_idx}, {component_idx + 1}, {1}, {1}, graph_proto);
  AddClipOp(res_to_clip_name, clipped, 0.0f, max, type, graph_proto);
}

// check AnfNode data type is float or not.
bool IsFloatDataType(const AnfNodePtr &node) {
  auto dtype = node->Type();
  auto elem_type = dyn_cast<TensorType>(dtype)->element()->type_id();
  switch (elem_type) {
    case (kNumberTypeFloat):
    case (kNumberTypeFloat16):
    case (kNumberTypeFloat32):
    case (kNumberTypeFloat64):
      return True;
    default:
      return False;
  }
}

namespace while_loop_export {
namespace {
const char CONTROL_PATTERN[] = "\u21B5";
const char LOOP_BODY_PATTERN[] = "\u21BB";
const char AFTER_LOOP_PATTERN[] = "\u2193";

const size_t LOOP_BODY_INPUT = 2;
const size_t AFTER_LOOP_INPUT = 3;

bool IsSubgraphNameCorrect(const FuncGraphPtr &func_graph, const std::string &part_pattern) {
  auto name = func_graph->ToString();
  return name.find("construct") != std::string::npos && name.find(part_pattern) != std::string::npos;
}

template <typename T>
const std::shared_ptr<T> GetNodeInput(const CNodePtr &node, size_t i) {
  auto input = GetRealInput(node->input(i));
  auto result = dyn_cast<T>(input);
  if (result == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get input " << i << " of node " << node->DebugString();
  }
  return result;
}

template <typename T>
const std::shared_ptr<T> GetNodeInputValue(const CNodePtr &node, size_t i) {
  auto input = GetNodeInput<ValueNode>(node, i);
  auto result = dyn_cast<T>(input->value());
  if (result == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get a value from input " << i << " of node " << node->DebugString();
  }
  return result;
}

CNodePtr FindLoopSwitchNode(const FuncGraphPtr &control_subgraph) {
  if (!IsSubgraphNameCorrect(control_subgraph, CONTROL_PATTERN)) {
    MS_LOG(EXCEPTION) << "Expected a loop control structure";
  }
  auto lazy_call_node = GetNodeInput<CNode>(control_subgraph->get_return(), kOneNum);
  if (lazy_call_node->inputs().size() != kOneNum || !lazy_call_node->input(kZeroNum)->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Expected a lazy call node";
  }
  auto switch_node = GetNodeInput<CNode>(lazy_call_node, kZeroNum);
  if (!switch_node->IsApply(prim::kPrimSwitch)) {
    MS_LOG(EXCEPTION) << "Expected a switch node";
  }
  return switch_node;
}

FuncGraphPtr GetSubgraph(const CNodePtr &switch_node, size_t input_index, const std::string &name_pattern) {
  auto input_node = GetNodeInput<CNode>(switch_node, input_index);
  if (!input_node->IsApply(prim::kPrimPartial)) {
    MS_LOG(EXCEPTION) << "Expected a partial node";
  }

  auto subgraph = GetNodeInputValue<FuncGraph>(input_node, kOneNum);
  if (!IsSubgraphNameCorrect(subgraph, name_pattern)) {
    MS_LOG(EXCEPTION) << "Expected a loop part: " << name_pattern;
  }

  return subgraph;
}

// The inputs of this node are the outputs of ONNX Loop
CNodePtr FindLoopRepeatNode(const FuncGraphPtr &loop_subgraph, const FuncGraphPtr &control_subgraph) {
  auto repeat_node = GetNodeInput<CNode>(loop_subgraph->return_node(), kOneNum);
  auto maybe_control_graph = GetNodeInputValue<FuncGraph>(repeat_node, kZeroNum);
  MS_EXCEPTION_IF_CHECK_FAIL(maybe_control_graph == control_subgraph, "Loop matching failed");
  return repeat_node;
}

struct LoopConditionInfo {
  int64_t begin;
  int64_t end;
  int64_t step;
};

/*
  NOTE: loop support is currently very limited, because proper condition export requires more graph surgery (copying
  condition expression before and inside Loop subgraph)
  The only while loop form supported currently is the one used in GNMT v2's Beam Search. Python example:
    i = begin
    while i < end
        ...
        i += step
  To enable proper support for arbitrary while loop contitions, condition calculation should be duplicated inside the
  Loop supgraph. But exporting the same ops twice with different names is not currently supported.
 */
LoopConditionInfo TraceLoopConditionInfo(const CNodePtr &start_node, const CNodePtr &cond_node,
                                         const FuncGraphPtr &control_subgraph, const CNodePtr &loop_repeat_node) {
  MS_EXCEPTION_IF_CHECK_FAIL(cond_node->IsApply(prim::kPrimLess), "Expected Less node");

  auto counter = GetNodeInput<Parameter>(cond_node, kOneNum);
  auto end_tensor = GetNodeInputValue<tensor::Tensor>(cond_node, kTwoNum);
  MS_EXCEPTION_IF_CHECK_FAIL(end_tensor->shape_c().empty(), "Expected a scalar tensor");
  auto end = *reinterpret_cast<const int32_t *>(end_tensor->data_c());

  const auto &subgraph_args = control_subgraph->parameters();
  auto counter_input_pos = std::find(subgraph_args.begin(), subgraph_args.end(), counter) - subgraph_args.begin();

  auto begin_tensor = GetNodeInputValue<tensor::Tensor>(start_node, 1UL + static_cast<size_t>(counter_input_pos));
  MS_EXCEPTION_IF_CHECK_FAIL(begin_tensor->shape_c().empty(), "Expected a scalar tensor");
  auto begin = *reinterpret_cast<const int32_t *>(begin_tensor->data_c());

  auto increment_node = GetNodeInput<CNode>(loop_repeat_node, 1UL + static_cast<size_t>(counter_input_pos));
  MS_EXCEPTION_IF_CHECK_FAIL(increment_node->IsApply(prim::kPrimAdd), "Expected Add node");
  auto step_tensor = GetNodeInputValue<tensor::Tensor>(increment_node, kTwoNum);
  MS_EXCEPTION_IF_CHECK_FAIL(step_tensor->shape_c().empty(), "Expected a scalar tensor");
  auto step = *reinterpret_cast<const int32_t *>(step_tensor->data_c());

  return LoopConditionInfo{begin, end, step};
}

// result[i] is which control subgraph input should be taken for pos i to match the order of loop subgraph inputs
std::vector<size_t> TraceLoopToControlMap(const FuncGraphPtr &control_subgraph) {
  std::vector<size_t> result;

  auto switch_node = FindLoopSwitchNode(control_subgraph);
  auto loop_partial_node = GetNodeInput<CNode>(switch_node, kTwoNum);
  const auto &control_params = control_subgraph->parameters();
  int64_t auxiliary_inputs_num = 2;
  for (size_t i = static_cast<size_t>(auxiliary_inputs_num); i < loop_partial_node->inputs().size(); ++i) {
    auto loop_param = GetNodeInput<Parameter>(loop_partial_node, i);
    auto control_param_pos =
      std::find(control_params.begin(), control_params.end(), loop_param) - control_params.begin();
    result.push_back(control_param_pos);
  }

  return result;
}

std::vector<size_t> TraceAfterToLoopMap(const FuncGraphPtr &control_subgraph) {
  std::vector<size_t> result;

  auto switch_node = FindLoopSwitchNode(control_subgraph);
  auto loop_partial_node = GetNodeInput<CNode>(switch_node, kTwoNum);
  auto after_partial_node = GetNodeInput<CNode>(switch_node, kThreeNum);
  const auto &loop_params = loop_partial_node->inputs();
  int64_t auxiliary_inputs_num = 2;
  for (size_t i = static_cast<size_t>(auxiliary_inputs_num); i < after_partial_node->inputs().size(); ++i) {
    auto after_param = GetNodeInput<Parameter>(after_partial_node, i);
    auto after_param_pos = std::find(loop_params.begin(), loop_params.end(), after_param) - loop_params.begin();
    result.push_back(after_param_pos - auxiliary_inputs_num);
  }

  return result;
}

std::vector<bool> TraceIgnoredLoopParams(const CNodePtr &start_node, const std::vector<size_t> &loop_to_control_map) {
  auto inputs_num = start_node->inputs().size() - 1;
  std::vector<bool> result(inputs_num);
  for (size_t loop_i = 0; loop_i < inputs_num; ++loop_i) {
    auto control_i = loop_to_control_map.at(loop_i);
    const auto &input = start_node->input(control_i + 1);
    if ((input->isa<Parameter>() && input->cast<ParameterPtr>()->has_default()) || HasAbstractMonad(input)) {
      result.at(loop_i) = true;
    }
  }
  return result;
}
}  // namespace

bool IsControlSubgraph(const ValuePtr &func_graph_node) {
  auto func_graph = dyn_cast<FuncGraph>(func_graph_node);
  return func_graph != nullptr && IsSubgraphNameCorrect(func_graph, CONTROL_PATTERN);
}

bool IsLoopBodyReturnNode(const CNodePtr &node, const FuncGraphPtr &func_graph) {
  return IsSubgraphNameCorrect(func_graph, LOOP_BODY_PATTERN) && node == func_graph->get_return();
}

bool IsAfterLoopReturnNode(const CNodePtr &node, const FuncGraphPtr &func_graph) {
  return IsSubgraphNameCorrect(func_graph, AFTER_LOOP_PATTERN) && node == func_graph->get_return();
}

struct LoopParts {
  LoopConditionInfo loop_condition_info;
  std::vector<std::pair<size_t, size_t>> after_param_to_output_indices;
  std::vector<size_t> ignored_loop_param_indices;
  std::vector<std::pair<size_t, size_t>> used_loop_to_control_param_indices;
  CNodePtr repeat_node;
  FuncGraphPtr loop_subgraph;
  FuncGraphPtr after_loop_subgraph;
};

LoopParts MatchGraph(const CNodePtr &start_node) {
  LoopParts result;

  auto control_subgraph_value = dyn_cast<ValueNode>(start_node->input(0));
  MS_EXCEPTION_IF_NULL(control_subgraph_value);
  auto control_subgraph = dyn_cast<FuncGraph>(control_subgraph_value->value());
  MS_EXCEPTION_IF_NULL(control_subgraph);

  auto switch_node = FindLoopSwitchNode(control_subgraph);
  auto cond_node = GetNodeInput<CNode>(switch_node, kOneNum);

  result.loop_subgraph = GetSubgraph(switch_node, LOOP_BODY_INPUT, LOOP_BODY_PATTERN);

  result.repeat_node = FindLoopRepeatNode(result.loop_subgraph, control_subgraph);
  result.loop_condition_info = TraceLoopConditionInfo(start_node, cond_node, control_subgraph, result.repeat_node);

  result.after_loop_subgraph = GetSubgraph(switch_node, AFTER_LOOP_INPUT, AFTER_LOOP_PATTERN);

  auto loop_to_control_order_map = TraceLoopToControlMap(control_subgraph);
  auto ignored_loop_params_mask = TraceIgnoredLoopParams(start_node, loop_to_control_order_map);
  auto loop_inputs_num = start_node->inputs().size() - 1;
  for (size_t i = 0; i < loop_inputs_num; ++i) {
    if (ignored_loop_params_mask.at(i)) {
      result.ignored_loop_param_indices.push_back(i);
    } else {
      result.used_loop_to_control_param_indices.push_back(std::make_pair(i, loop_to_control_order_map.at(i)));
    }
  }

  auto after_to_loop_order_map = TraceAfterToLoopMap(control_subgraph);
  for (size_t after_i = 0; after_i < result.after_loop_subgraph->parameters().size(); ++after_i) {
    auto loop_i = after_to_loop_order_map.at(after_i);
    if (!ignored_loop_params_mask.at(loop_i)) {
      auto output_i = loop_i;
      for (size_t i = 0; i < loop_i; ++i) {
        output_i -= static_cast<size_t>(ignored_loop_params_mask.at(i));
      }
      result.after_param_to_output_indices.push_back(std::make_pair(after_i, output_i));
    }
  }

  return result;
}
}  // namespace while_loop_export

class OpAttrInfo {
 public:
  OpAttrInfo(const std::string &attr_name, const string &onnx_attr_name,
             onnx::AttributeProto_AttributeType onnx_attr_type, const GenAttrFuncType &fn_gen_attr)
      : attr_name_(attr_name),
        onnx_attr_name_(onnx_attr_name),
        onnx_attr_type_(onnx_attr_type),
        fn_gen_attr_(fn_gen_attr) {}
  ~OpAttrInfo() {}

  const std::string &attr_name() const { return attr_name_; }
  const std::string &onnx_attr_name() const { return onnx_attr_name_; }
  onnx::AttributeProto_AttributeType onnx_attr_type() const { return onnx_attr_type_; }
  GenAttrFuncType fn_gen_attr() const { return fn_gen_attr_; }

 private:
  std::string attr_name_;                              // attribute name of MindSpore
  std::string onnx_attr_name_;                         // corresponding attribute name of ONNX
  onnx::AttributeProto_AttributeType onnx_attr_type_;  // corresponding attribute type of ONNX
  GenAttrFuncType fn_gen_attr_;                        // function used convert
};

struct InputConversion {
  int input_index;
  onnx::TensorProto_DataType input_type;
  onnx::TensorProto_DataType target_type;
};

struct OutputConversion {
  int output_index;
  enum class Mode { FIXED, INPUT } mode;
  union {
    onnx::TensorProto_DataType target_type;
    int input_with_matching_type;
  };
};

class OpNameInfo {
 public:
  OpNameInfo &set_op_type(const std::string &op_type) {
    op_type_ = op_type;
    return *this;
  }

  const std::string &op_type() const { return op_type_; }

  OpNameInfo &set_onnx_type(const std::string &onnx_type) {
    onnx_type_ = onnx_type;
    return *this;
  }

  const std::string &onnx_type() const { return onnx_type_; }

  OpNameInfo &Attr(const std::string &attr_name, const std::string &onnx_attr_name,
                   onnx::AttributeProto_AttributeType onnx_attr_type, const GenAttrFuncType &fn_gen_attr) {
    (void)op_attrs_.emplace_back(OpAttrInfo(attr_name, onnx_attr_name, onnx_attr_type, fn_gen_attr));
    return *this;
  }

  const std::vector<OpAttrInfo> &op_attrs() const { return op_attrs_; }

  const std::vector<InputConversion> &input_casts() const { return input_casts_; }

  OpNameInfo &CastInput(int input_index, onnx::TensorProto_DataType input_type,
                        onnx::TensorProto_DataType target_type) {
    input_casts_.push_back({input_index, input_type, target_type});
    return *this;
  }

  const std::vector<OutputConversion> &output_casts() const { return output_casts_; }

  OpNameInfo &CastOutputToFixedType(onnx::TensorProto_DataType type, int output_index = 0) {
    output_casts_.push_back({output_index, OutputConversion::Mode::FIXED, {type}});
    return *this;
  }

  OpNameInfo &CastOutputToInputType(int input_index, int output_index = 0) {
    auto rule = OutputConversion{output_index, OutputConversion::Mode::INPUT};
    rule.input_with_matching_type = input_index;
    output_casts_.push_back(rule);
    return *this;
  }

  int num_outputs() const { return num_outputs_; }

  OpNameInfo &set_num_outputs(int n) {
    num_outputs_ = n;
    return *this;
  }

 private:
  std::string op_type_;                         // operator type of MindSpore
  std::string onnx_type_;                       // corresponding ONNX operator type
  std::vector<OpAttrInfo> op_attrs_;            // operator attributes map info
  std::vector<InputConversion> input_casts_;    // if input input_index has type input_type, cast it to target_type
  std::vector<OutputConversion> output_casts_;  // cast output output_index to fixed type or input type
  int num_outputs_ = 1;
};

#define OPERATOR_ONNX_CONVERT_DEFINE(name, onnx_name, impl) \
  OpNameInfo GetOpOnnxConvertInfo_##name() { return impl.set_op_type(#name).set_onnx_type(#onnx_name); }

OPERATOR_ONNX_CONVERT_DEFINE(Mod, Mod, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Add, Add, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Mul, Mul, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Pow, Pow, OpNameInfo())

OPERATOR_ONNX_CONVERT_DEFINE(ReLU, Relu, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Sigmoid, Sigmoid, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Sin, Sin, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Round, Round, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Div, Div, OpNameInfo())

OPERATOR_ONNX_CONVERT_DEFINE(Flatten, Flatten, OpNameInfo())

OPERATOR_ONNX_CONVERT_DEFINE(
  Conv2D, Conv,
  OpNameInfo()
    .Attr("dilation", "dilations", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>)
    .Attr("group", "group", onnx::AttributeProto_AttributeType_INT, SetAttrValueToProto<Int64Imm>)
    .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<0>)
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetConvPadding)
    .Attr("stride", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>))
OPERATOR_ONNX_CONVERT_DEFINE(
  Conv3D, Conv,
  OpNameInfo()
    .Attr("dilations", "dilations", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<kTwoNum>)
    .Attr("group", "group", onnx::AttributeProto_AttributeType_INT, SetAttrValueToProto<Int64Imm>)
    .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<0>)
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetConvPadding)
    .Attr("strides", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<kTwoNum>))
OPERATOR_ONNX_CONVERT_DEFINE(
  Conv3DTranspose, ConvTranspose,
  OpNameInfo()
    .Attr("dilations", "dilations", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<kTwoNum>)
    .Attr("group", "group", onnx::AttributeProto_AttributeType_INT, SetAttrValueToProto<Int64Imm>)
    .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<0>)
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetConvTransposePadding)
    .Attr("strides", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<kTwoNum>)
    .Attr("output_padding", "output_padding", onnx::AttributeProto_AttributeType_INTS,
          SetAttrTupleValueToProto<kTwoNum>))

OPERATOR_ONNX_CONVERT_DEFINE(DepthToSpace, DepthToSpace,
                             OpNameInfo().Attr("block_size", "blocksize", onnx::AttributeProto_AttributeType_INT,
                                               SetAttrValueToProto<Int64Imm>))

OPERATOR_ONNX_CONVERT_DEFINE(BiasAdd, Add, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(MatMul, Gemm,
                             OpNameInfo()
                               .Attr("transpose_a", "transA", onnx::AttributeProto_AttributeType_INT,
                                     SetAttrValueToProto<BoolImm>)
                               .Attr("transpose_b", "transB", onnx::AttributeProto_AttributeType_INT,
                                     SetAttrValueToProto<BoolImm>))

OPERATOR_ONNX_CONVERT_DEFINE(BatchNorm, BatchNormalization,
                             OpNameInfo()
                               .Attr("epsilon", "epsilon", onnx::AttributeProto_AttributeType_FLOAT,
                                     SetAttrValueToProto<FP32Imm>)
                               .CastInput(0, onnx::TensorProto_DataType_FLOAT16, onnx::TensorProto_DataType_FLOAT)
                               .CastOutputToInputType(0))

OPERATOR_ONNX_CONVERT_DEFINE(Reshape, Reshape, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Cast, Cast, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(PReLU, PRelu, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Argmax, ArgMax,
                             OpNameInfo()
                               .Attr("axis", "axis", onnx::AttributeProto_AttributeType_INT,
                                     SetAttrValueToProto<Int64Imm>)
                               .Attr("", "keepdims", onnx::AttributeProto_AttributeType_INT,
                                     [](ValuePtr, onnx::AttributeProto_AttributeType,
                                        onnx::AttributeProto *const attr_proto, const PrimitivePtr &) {
                                       attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
                                       attr_proto->set_i(0);
                                     })
                               .CastOutputToFixedType(onnx::TensorProto_DataType_INT32))

OPERATOR_ONNX_CONVERT_DEFINE(SimpleMean, AveragePool, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(
  MaxPool, MaxPool,
  OpNameInfo()
    .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>)
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetPoolingPadMode)
    .Attr("strides", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>))

OPERATOR_ONNX_CONVERT_DEFINE(
  MaxPool3D, MaxPool,
  OpNameInfo()
    .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>)
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetPoolingPadMode)
    .Attr("strides", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>))

OPERATOR_ONNX_CONVERT_DEFINE(
  MaxPoolWithArgmax, MaxPool,
  OpNameInfo()
    .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>)
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetPoolingPadMode)
    .Attr("strides", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>))

OPERATOR_ONNX_CONVERT_DEFINE(
  AvgPool, AveragePool,
  OpNameInfo()
    .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>)
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetPoolingPadMode)
    .Attr("strides", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>))

OPERATOR_ONNX_CONVERT_DEFINE(Gather, Gather, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(MakeTuple, SequenceConstruct, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(RealDiv, Div, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Sub, Sub, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Neg, Neg, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Maximum, Max,
                             OpNameInfo()
                               .CastInput(0, onnx::TensorProto_DataType_INT32, onnx::TensorProto_DataType_FLOAT)
                               .CastInput(1, onnx::TensorProto_DataType_INT32, onnx::TensorProto_DataType_FLOAT)
                               .CastOutputToInputType(0))
OPERATOR_ONNX_CONVERT_DEFINE(Minimum, Min,
                             OpNameInfo()
                               .CastInput(0, onnx::TensorProto_DataType_INT32, onnx::TensorProto_DataType_FLOAT)
                               .CastInput(1, onnx::TensorProto_DataType_INT32, onnx::TensorProto_DataType_FLOAT)
                               .CastOutputToInputType(0))
OPERATOR_ONNX_CONVERT_DEFINE(Transpose, Transpose, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Exp, Exp, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Softplus, Softplus, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Tanh, Tanh, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Abs, Abs, OpNameInfo())

// MindSpore Softmax axis(int, Tuple)
OPERATOR_ONNX_CONVERT_DEFINE(Softmax, Softmax,
                             OpNameInfo().Attr("axis", "axis", onnx::AttributeProto_AttributeType_INT,
                                               SetAttrTupleValueToProto<0>))

// MindSpore LogSoftmax axis(int)
OPERATOR_ONNX_CONVERT_DEFINE(LogSoftmax, LogSoftmax,
                             OpNameInfo().Attr("axis", "axis", onnx::AttributeProto_AttributeType_INT,
                                               SetAttrValueToProto<Int64Imm>))

OPERATOR_ONNX_CONVERT_DEFINE(Softsign, Softsign, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Sqrt, Sqrt, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Equal, Equal, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Floor, Floor, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(ACos, Acos, OpNameInfo())

OPERATOR_ONNX_CONVERT_DEFINE(GatherNd, GatherND,
                             OpNameInfo().CastInput(1, onnx::TensorProto_DataType_INT32,
                                                    onnx::TensorProto_DataType_INT64))
OPERATOR_ONNX_CONVERT_DEFINE(Select, Where, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Log, Log, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Greater, Greater, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(LogicalAnd, And, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(LogicalOr, Or, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(ReverseSequence, ReverseSequence,
                             OpNameInfo()
                               .Attr("seq_dim", "time_axis", onnx::AttributeProto_AttributeType_INT,
                                     SetAttrValueToProto<Int64Imm>)
                               .Attr("batch_dim", "batch_axis", onnx::AttributeProto_AttributeType_INT,
                                     SetAttrValueToProto<Int64Imm>)
                               .CastInput(1, onnx::TensorProto_DataType_INT32, onnx::TensorProto_DataType_INT64))
OPERATOR_ONNX_CONVERT_DEFINE(Less, Less, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(TensorScatterUpdate, ScatterND,
                             OpNameInfo().CastInput(1, onnx::TensorProto_DataType_INT32,
                                                    onnx::TensorProto_DataType_INT64))
OPERATOR_ONNX_CONVERT_DEFINE(Cos, Cos, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Atan2, Atan2, OpNameInfo())

#define OP_CONVERT_FUNCTION_NAME(name) GetOpOnnxConvertInfo_##name

void RegisterOpConverters(const std::function<void(OpNameInfo &&)> &fn) {
  fn(OP_CONVERT_FUNCTION_NAME(Mod)());
  fn(OP_CONVERT_FUNCTION_NAME(DepthToSpace)());
  fn(OP_CONVERT_FUNCTION_NAME(Add)());
  fn(OP_CONVERT_FUNCTION_NAME(Mul)());
  fn(OP_CONVERT_FUNCTION_NAME(Pow)());
  fn(OP_CONVERT_FUNCTION_NAME(ReLU)());
  fn(OP_CONVERT_FUNCTION_NAME(Sigmoid)());
  fn(OP_CONVERT_FUNCTION_NAME(Conv2D)());
  fn(OP_CONVERT_FUNCTION_NAME(Conv3D)());
  fn(OP_CONVERT_FUNCTION_NAME(Conv3DTranspose)());
  fn(OP_CONVERT_FUNCTION_NAME(Argmax)());
  fn(OP_CONVERT_FUNCTION_NAME(Flatten)());
  fn(OP_CONVERT_FUNCTION_NAME(MaxPool)());
  fn(OP_CONVERT_FUNCTION_NAME(MaxPool3D)());
  fn(OP_CONVERT_FUNCTION_NAME(MaxPoolWithArgmax)());
  fn(OP_CONVERT_FUNCTION_NAME(AvgPool)());

  fn(OP_CONVERT_FUNCTION_NAME(BatchNorm)());
  fn(OP_CONVERT_FUNCTION_NAME(MatMul)());
  fn(OP_CONVERT_FUNCTION_NAME(MakeTuple)());
  fn(OP_CONVERT_FUNCTION_NAME(RealDiv)());
  fn(OP_CONVERT_FUNCTION_NAME(BiasAdd)());
  fn(OP_CONVERT_FUNCTION_NAME(Sub)());
  fn(OP_CONVERT_FUNCTION_NAME(Neg)());
  fn(OP_CONVERT_FUNCTION_NAME(Maximum)());
  fn(OP_CONVERT_FUNCTION_NAME(Minimum)());
  fn(OP_CONVERT_FUNCTION_NAME(Exp)());

  fn(OP_CONVERT_FUNCTION_NAME(Softplus)());
  fn(OP_CONVERT_FUNCTION_NAME(Tanh)());
  fn(OP_CONVERT_FUNCTION_NAME(Softmax)());
  fn(OP_CONVERT_FUNCTION_NAME(LogSoftmax)());
  fn(OP_CONVERT_FUNCTION_NAME(Abs)());
  fn(OP_CONVERT_FUNCTION_NAME(Softsign)());
  fn(OP_CONVERT_FUNCTION_NAME(Sqrt)());
  fn(OP_CONVERT_FUNCTION_NAME(Equal)());
  fn(OP_CONVERT_FUNCTION_NAME(Floor)());
  fn(OP_CONVERT_FUNCTION_NAME(ACos)());

  fn(OP_CONVERT_FUNCTION_NAME(GatherNd)());
  fn(OP_CONVERT_FUNCTION_NAME(Select)());
  fn(OP_CONVERT_FUNCTION_NAME(Log)());
  fn(OP_CONVERT_FUNCTION_NAME(Less)());
  fn(OP_CONVERT_FUNCTION_NAME(Greater)());
  fn(OP_CONVERT_FUNCTION_NAME(LogicalAnd)());
  fn(OP_CONVERT_FUNCTION_NAME(LogicalOr)());
  fn(OP_CONVERT_FUNCTION_NAME(ReverseSequence)());
  fn(OP_CONVERT_FUNCTION_NAME(TensorScatterUpdate)());

  fn(OP_CONVERT_FUNCTION_NAME(Sin)());
  fn(OP_CONVERT_FUNCTION_NAME(Cos)());
  fn(OP_CONVERT_FUNCTION_NAME(Atan2)());
  fn(OP_CONVERT_FUNCTION_NAME(Round)());
  fn(OP_CONVERT_FUNCTION_NAME(Div)());
}

class OpConvertRegistry {
 public:
  ~OpConvertRegistry() { Clear(); }

  static void RegisterOneOpConverter(OpNameInfo &&op_info) { GetSingleton().op_map_[op_info.op_type()] = op_info; }

  static void RegisterAllOpConverters() { RegisterOpConverters(RegisterOneOpConverter); }

  static OpConvertRegistry &GetSingleton() {
    static OpConvertRegistry registry = OpConvertRegistry();
    return registry;
  }

  static const mindspore::HashMap<std::string, OpNameInfo> &GetOpConvertMap() { return GetSingleton().op_map_; }

  void Clear() noexcept { op_map_.clear(); }

 private:
  OpConvertRegistry() {}

  mindspore::HashMap<std::string, OpNameInfo> op_map_;
};

class OnnxExporter {
 public:
  OnnxExporter() {}
  ~OnnxExporter() {}

  std::string GetOnnxProtoString(const FuncGraphPtr &func_graph);

 private:
  void InitModelInfo();

  void ExportFuncGraph(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, std::string> *node_map_ptr,
                       onnx::GraphProto *graph_proto, bool export_inputs = true);
  void ExportInputs(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, std::string> *node_map_ptr,
                    onnx::GraphProto *graph_proto);

  std::string ExportPrimitive(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, std::string> *node_map_ptr,
                              const PrimitivePtr &prim, const std::vector<AnfNodePtr> &inputs,
                              onnx::GraphProto *graph_proto);

  static onnx::TensorProto_DataType GetOnnxDataType(TypeId type_id);
  static onnx::TensorProto_DataType GetOutputType(const AnfNodePtr &node, int64_t output_index = -1);
  void SetValueInfoType(const AnfNodePtr &node, onnx::ValueInfoProto *value_proto, int64_t output_index = -1) const;

  void MatchAndMark(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                    mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) const;
  void MatchAndMarkCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                         mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) const;
  void IgnoreMakeTuple(const AnfNodePtr &node, mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) const;

  void ExportNodes(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, std::string> *node_map_ptr,
                   onnx::GraphProto *graph_proto);

  void ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node,
                   std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportWhileLoop(const CNodePtr &start_node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                       onnx::GraphProto *graph_proto);

  void ExportPrimReshape(const FuncGraphPtr &func_graph, const CNodePtr &node,
                         std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimReduce(const FuncGraphPtr &func_graph, const CNodePtr &node,
                        std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimReduceAnyOrAll(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTranspose(const FuncGraphPtr &func_graph, const CNodePtr &node,
                           std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &node,
                              std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  onnx::NodeProto *PrimResizeExportHelper(const FuncGraphPtr &, const CNodePtr &node,
                                          std::map<AnfNodePtr, std::string> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto);
  void ExportPrimResizeNearestNeighbor(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                       std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimResizeBilinear(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimExpandDims(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimGatherD(const FuncGraphPtr &func_graph, const CNodePtr &node,
                         std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimPad(const FuncGraphPtr &func_graph, const CNodePtr &node,
                     std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimBatchMatMul(const FuncGraphPtr &func_graph, const CNodePtr &node,
                             std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimBroadcastTo(const FuncGraphPtr &func_graph, const CNodePtr &node,
                             std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimAddN(const FuncGraphPtr &func_graph, const CNodePtr &node,
                      std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimGeLU(const FuncGraphPtr &func_graph, const CNodePtr &node,
                      std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimConcat(const FuncGraphPtr &func_graph, const CNodePtr &node,
                        std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimCast(const FuncGraphPtr &func_graph, const CNodePtr &node,
                      std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimPReLU(const FuncGraphPtr &func_graph, const CNodePtr &node,
                       std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimReLU6(const FuncGraphPtr &func_graph, const CNodePtr &node,
                       std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimDepthwiseConv2d(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                 std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTile(const FuncGraphPtr &func_graph, const CNodePtr &node,
                      std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimSquare(const FuncGraphPtr &func_graph, const CNodePtr &node,
                        std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimGatherV2(const FuncGraphPtr &func_graph, const CNodePtr &node,
                          std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTupleGetItem(const FuncGraphPtr &func_graph, const CNodePtr &node,
                              std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTopK(const FuncGraphPtr &func_graph, const CNodePtr &node,
                      std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimBoundingBoxDecode(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimNMSWithMask(const FuncGraphPtr &func_graph, const CNodePtr &node,
                             std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimSplit(const FuncGraphPtr &func_graph, const CNodePtr &node,
                       std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimROIAlign(const FuncGraphPtr &func_graph, const CNodePtr &node,
                          std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimSlice(const FuncGraphPtr &func_graph, const CNodePtr &node,
                       std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimOnesLike(const FuncGraphPtr &func_graph, const CNodePtr &node,
                          std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimScatterNd(const FuncGraphPtr &func_graph, const CNodePtr &node,
                           std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimArgMaxWithValue(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                 std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimArgMinWithValue(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                 std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimOneHot(const FuncGraphPtr &func_graph, const CNodePtr &node,
                        std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void PrimConv2DTransposeExportHelper(const CNodePtr &conv_node, const CNodePtr &bias_add_node,
                                       std::map<AnfNodePtr, std::string> *node_map_ptr,
                                       onnx::GraphProto *const graph_proto);
  void ExportPrimConv2DTranspose(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                 std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimGreaterEqual(const FuncGraphPtr &func_graph, const CNodePtr &node,
                              std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimLessEqual(const FuncGraphPtr &func_graph, const CNodePtr &node,
                           std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimSqueeze(const FuncGraphPtr &func_graph, const CNodePtr &node,
                         std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimDynamicRNN(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *const graph_proto);
  void ExportPrimLSTM(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                      onnx::GraphProto *graph_proto);
  void ExportPrimReverseV2(const FuncGraphPtr &func_graph, const CNodePtr &node,
                           std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTensorCopySlices(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimStack(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportPrimAtan2(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportPrimFloorDiv(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                          onnx::GraphProto *graph_proto);
  void ExportPrimFloorMod(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                          onnx::GraphProto *graph_proto);
  void ExportPrimSort(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                      onnx::GraphProto *graph_proto);
  void ExportPrimCustom(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                        onnx::GraphProto *graph_proto);
  void ExportMergeConv(const FuncGraphPtr &func_graph, const CNodePtr &node,
                       std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeGemm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                       std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeBatchNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeMaxPoolWithArgmax(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                    std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeLayerNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeConv2DTranspose(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeDynamicGRUV2(const FuncGraphPtr &, const CNodePtr &node,
                               std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *const graph_proto);
  void ExportOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &return_arg,
                    std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto);
  std::string GetNodeInputName(const AnfNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                               onnx::GraphProto *const);

  void ConvertTupleToTensor(const ValuePtr &value, onnx::TensorProto *tensor_proto) const;
  void SetTensorData(const ValuePtr &value, onnx::TensorProto *tensor_proto);

  void AddOutputWithCast(onnx::NodeProto *node_proto, const std::string &output_name,
                         onnx::TensorProto_DataType target_type, onnx::GraphProto *graph_proto) const;

  std::string GenerateUniqueName() { return std::to_string(++onnx_node_index_); }
  std::string RegisterNodeWithUniqueName(const AnfNodePtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr) {
    auto name = GenerateUniqueName();
    (*node_map_ptr)[node] = name;
    return name;
  }
  std::string GenerateUniqueParameterName(const ParameterPtr &node, std::map<AnfNodePtr, std::string> *node_map_ptr) {
    auto node_name = node->ToString();
    MS_EXCEPTION_IF_CHECK_FAIL(node_name != "", "Cannot get the name of an ignored parameter");
    auto dup_iter = std::find_if(node_map_ptr->begin(), node_map_ptr->end(),
                                 [&node_name](const auto &pair) { return pair.second == node_name; });
    if (dup_iter != node_map_ptr->end()) {
      node_name = GenerateUniqueName() + node_name;
    }
    return node_name;
  }

  void ResetNodeIndex() { onnx_node_index_ = 0; }

  static int64_t GetInt64Value(const AnfNodePtr &node) {
    auto value_node_ptr = dyn_cast<ValueNode>(node);
    MS_EXCEPTION_IF_NULL(value_node_ptr);
    return GetValue<int64_t>(value_node_ptr->value());
  }

  onnx::ModelProto model_;

  size_t onnx_node_index_ = 0;

  std::map<AnfNodePtr, std::string> renamed_node_map_;
};

std::string OnnxExporter::GetOnnxProtoString(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return "";
  }
  ResetNodeIndex();
  OpConvertRegistry::GetSingleton().Clear();
  OpConvertRegistry::RegisterAllOpConverters();
  InitModelInfo();
  onnx::GraphProto *graph_proto = model_.mutable_graph();
  std::map<AnfNodePtr, std::string> node_map;
  ExportFuncGraph(func_graph, &node_map, graph_proto);
  return model_.SerializeAsString();
}

void OnnxExporter::InitModelInfo() {
  model_.set_ir_version(onnx::IR_VERSION_2019_1_22);
  model_.set_producer_name("MindSpore");
  model_.set_producer_version("1.0");
  onnx::OperatorSetIdProto *opset_proto = model_.add_opset_import();
  opset_proto->set_version(ONNX_VERSION);
}

void OnnxExporter::ExportFuncGraph(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *const graph_proto, bool export_inputs) {
  MS_LOG(INFO) << "Begin exporting onnx model for graph " << func_graph->ToString();

  // set graph name
  graph_proto->set_name(func_graph->ToString());

  // export inputs if graph is not inlined
  if (export_inputs) {
    ExportInputs(func_graph, node_map_ptr, graph_proto);
  }

  // export computational nodes and output nodes
  ExportNodes(func_graph, node_map_ptr, graph_proto);

  // add names for easier debugging
  for (auto &node : *graph_proto->mutable_node()) {
    if (!node.has_name()) {
      node.set_name(node.output(0) + node.op_type());
    }
  }

  MS_LOG(INFO) << "End exporting onnx model for graph " << func_graph->ToString();
}

void OnnxExporter::ExportInputs(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, std::string> *node_map_ptr,
                                onnx::GraphProto *const graph_proto) {
  for (auto &param : func_graph->parameters()) {
    const ParameterPtr param_ptr = dyn_cast<Parameter>(param);
    if (param_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Parameter '" << param->ToString() << "' could not cast to parameter.";
    }

    if (param_ptr->has_default()) {
      continue;
    }

    // set onnx input.
    std::string name;
    auto renamed_iter = renamed_node_map_.find(param_ptr);
    if (renamed_iter != renamed_node_map_.end()) {
      name = renamed_iter->second;
      if (name == "") {
        continue;
      }
    } else {
      name = GenerateUniqueParameterName(param_ptr, node_map_ptr);
      (*node_map_ptr)[param_ptr] = name;
    }

    onnx::ValueInfoProto *input_proto = graph_proto->add_input();
    input_proto->set_name(name);
    SetValueInfoType(param_ptr, input_proto);
  }
}

onnx::TensorProto_DataType OnnxExporter::GetOnnxDataType(TypeId type_id) {
  // clang-format off
  static mindspore::HashMap<int, onnx::TensorProto_DataType> type_map = {
    {kNumberTypeBool, onnx::TensorProto_DataType_BOOL},
    {kNumberTypeInt8, onnx::TensorProto_DataType_INT8},
    {kNumberTypeInt16, onnx::TensorProto_DataType_INT16},
    {kNumberTypeInt32, onnx::TensorProto_DataType_INT32},
    {kNumberTypeInt64, onnx::TensorProto_DataType_INT64},
    {kNumberTypeUInt8, onnx::TensorProto_DataType_UINT8},
    {kNumberTypeUInt16, onnx::TensorProto_DataType_UINT16},
    {kNumberTypeUInt32, onnx::TensorProto_DataType_UINT32},
    {kNumberTypeUInt64, onnx::TensorProto_DataType_UINT64},
    {kNumberTypeFloat16, onnx::TensorProto_DataType_FLOAT16},
    {kNumberTypeFloat32, onnx::TensorProto_DataType_FLOAT},
    {kNumberTypeFloat64, onnx::TensorProto_DataType_DOUBLE},
  };
  // clang-format on

  auto iter = type_map.find(type_id);
  if (iter == type_map.end()) {
    MS_LOG(EXCEPTION) << "Convert type error, unsupported type " << type_id;
  }

  return iter->second;
}

void OnnxExporter::SetValueInfoType(const AnfNodePtr &node, onnx::ValueInfoProto *const value_proto,
                                    int64_t output_index) const {
  auto dtype = GetOutputType(node, output_index);
  auto shape = node->Shape();

  abstract::ShapePtr output_shape;
  if (shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = dyn_cast<abstract::TupleShape>(shape);
    auto base_shape = tuple_shape->shape().at(static_cast<size_t>(output_index));
    output_shape = dyn_cast<abstract::Shape>(base_shape);
    if (output_shape == nullptr) {
      MS_LOG(EXCEPTION) << "Expected " << node->ToString() << " to output a tuple of tensors. Instead got "
                        << base_shape->ToString() << " from output " << output_index;
    }
  } else if (shape->isa<abstract::Shape>()) {
    output_shape = dyn_cast<abstract::Shape>(shape);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported shape: " << shape->ToString();
  }

  auto *type_proto = value_proto->mutable_type();
  type_proto->mutable_tensor_type()->set_elem_type(dtype);
  auto *shape_proto = type_proto->mutable_tensor_type()->mutable_shape();

  for (const auto dim : output_shape->shape()) {
    shape_proto->add_dim()->set_dim_value(dim);
  }
}

void OnnxExporter::MatchAndMark(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                                mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) const {
  auto &op_merged_infos = *op_merged_infos_ptr;

  for (auto &node : nodes) {
    if (!node->isa<CNode>() || IsZeroRefcountNode(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == func_graph->get_return()) {
      // if the key `input` does not exist, just create a new one
      op_merged_infos[cnode].referred_count += 1;
    }
    for (auto &orig_input : cnode->inputs()) {
      auto input = GetRealInput(orig_input);
      if (!input->isa<CNode>() || IsZeroRefcountNode(input)) {
        continue;
      }
      // if the key `input` does not exist, just create a new one
      op_merged_infos[input].referred_count += 1;
    }
    MatchAndMarkCNode(func_graph, cnode, op_merged_infos_ptr);
  }
}

struct MergeRule {
  PrimitivePtr node_type;
  PrimitivePtr prev_type;
  OpMergeMode merge_mode;
};

void OnnxExporter::MatchAndMarkCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                     mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) const {
  MS_EXCEPTION_IF_NULL(op_merged_infos_ptr);
  auto &op_merged_infos = *op_merged_infos_ptr;
  const auto ignore = [&op_merged_infos](const AnfNodePtr &node) {
    op_merged_infos[node].mode = OP_MERGE_IGNORE;
    op_merged_infos[node].referred_count -= 1;
  };

  const std::vector<MergeRule> first_input_merge_rules = {
    {prim::kPrimBiasAdd, prim::kPrimConv2D, OP_MERGE_CONV},
    {prim::kPrimBiasAdd, prim::kPrimConv2DTranspose, OP_MERGE_CONV2D_TRANSPOSE},
    {prim::kPrimBiasAdd, prim::kPrimConv3D, OP_MERGE_CONV},
    {prim::kPrimBiasAdd, prim::kPrimConv3DTranspose, OP_MERGE_CONV},
    {prim::kPrimBiasAdd, prim::kPrimMatMul, OP_MERGE_GEMM},
    {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, OP_MERGE_BATCH_NORM},
    {prim::kPrimTupleGetItem, prim::kPrimMaxPoolWithArgmax, OP_MERGE_MAXPOOL_WITH_ARGMAX},
    {prim::kPrimTupleGetItem, prim::kPrimLayerNorm, OP_MERGE_LAYER_NORM},
    {prim::kPrimTupleGetItem, prim::kPrimDynamicGRUV2, OP_MERGE_DYNAMIC_GRU_V2},
  };

  auto rule = std::find_if(first_input_merge_rules.begin(), first_input_merge_rules.end(), [&cnode](const auto &rule) {
    return cnode->IsApply(rule.node_type) && IsPrimitiveCNode(cnode->input(1), rule.prev_type);
  });
  if (rule != first_input_merge_rules.end()) {
    if (cnode->IsApply(prim::kPrimTupleGetItem) && GetInt64Value(cnode->input(kTwoNum)) != 0) {
      MS_LOG(EXCEPTION) << "Multiple outputs for node \"" << cnode->input(1)->ToString() << "\" are not supported";
    }
    op_merged_infos[cnode].mode = rule->merge_mode;
    ignore(cnode->input(1));
  } else if (while_loop_export::IsLoopBodyReturnNode(cnode, func_graph)) {
    // Ignore to replace with other outputs
    ignore(cnode);
    auto repeat_node = dyn_cast<CNode>(GetRealInput(cnode->input(1)));
    MS_EXCEPTION_IF_NULL(repeat_node);
    ignore(repeat_node);
  } else if (while_loop_export::IsAfterLoopReturnNode(cnode, func_graph)) {
    // Ignore to inline after-loop subgraph in main graph
    ignore(cnode);
    auto first_input = GetRealInput(cnode->input(1));
    if (IsPrimitiveCNode(first_input, prim::kPrimMakeTuple)) {
      ignore(first_input);
    }
  } else if (cnode == func_graph->get_return()) {
    auto first_input = GetRealInput(cnode->input(1));  // Unpack Depend
    // Ignore MakeTuple output node to avoid exporting it to SequenceConstruct
    // and handle multiple outputs in ExportOutput
    IgnoreMakeTuple(first_input, op_merged_infos_ptr);
  } else if (cnode->IsApply(prim::kPrimConcat) && IsPrimitiveCNode(cnode->input(1), prim::kPrimMakeTuple)) {
    // Ignore MakeTuple to handle it in ExportPrimConcat
    ignore(cnode->input(1));
  }
}

void OnnxExporter::IgnoreMakeTuple(const AnfNodePtr &node,
                                   mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) const {
  MS_EXCEPTION_IF_NULL(op_merged_infos_ptr);
  auto &op_merged_infos = *op_merged_infos_ptr;
  const auto ignore = [&op_merged_infos](const AnfNodePtr &node) {
    op_merged_infos[node].mode = OP_MERGE_IGNORE;
    op_merged_infos[node].referred_count -= 1;
  };
  if (node == nullptr) {
    return;
  }

  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    ignore(node);
    auto cnode = dyn_cast<CNode>(node);
    if (cnode != nullptr) {
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        auto real_input = GetRealInput(cnode->input(i));
        IgnoreMakeTuple(real_input, op_merged_infos_ptr);
      }
    }
  }
}

/**
 * AnfNode
 * +-- CNode
 * +-- ANode
 * |   +-- Parameter
 * |   `-- ValueNode
 */
void OnnxExporter::ExportNodes(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, std::string> *node_map_ptr,
                               onnx::GraphProto *const graph_proto) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);

  mindspore::HashMap<AnfNodePtr, OpMergedInfo> op_merged_infos;
  MatchAndMark(func_graph, nodes, &op_merged_infos);
  for (const AnfNodePtr &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    auto iter = op_merged_infos.find(cnode);
    // the node is not referenced by any other nodes, skip it
    if (iter == op_merged_infos.end()) {
      continue;
    }
    auto merged_info = iter->second;
    // the op node is merged with other node and not used any more, skip it
    if (merged_info.mode == OP_MERGE_IGNORE && merged_info.referred_count == 0) {
      continue;
    }
    if (cnode == func_graph->get_return()) {
      ExportOutput(func_graph, cnode->input(kOneNum), node_map_ptr, graph_proto);
      continue;
    }
    switch (merged_info.mode) {
      case OP_MERGE_CONV:
        ExportMergeConv(func_graph, cnode, node_map_ptr, graph_proto);
        break;
      case OP_MERGE_GEMM:
        ExportMergeGemm(func_graph, cnode, node_map_ptr, graph_proto);
        break;
      case OP_MERGE_BATCH_NORM:
        ExportMergeBatchNorm(func_graph, cnode, node_map_ptr, graph_proto);
        break;
      case OP_MERGE_MAXPOOL_WITH_ARGMAX:
        ExportMergeMaxPoolWithArgmax(func_graph, cnode, node_map_ptr, graph_proto);
        break;
      case OP_MERGE_LAYER_NORM:
        ExportMergeLayerNorm(func_graph, cnode, node_map_ptr, graph_proto);
        break;
      case OP_MERGE_CONV2D_TRANSPOSE:
        ExportMergeConv2DTranspose(func_graph, cnode, node_map_ptr, graph_proto);
        break;
      case OP_MERGE_DYNAMIC_GRU_V2:
        ExportMergeDynamicGRUV2(func_graph, cnode, node_map_ptr, graph_proto);
        break;
      default:
        ExportCNode(func_graph, cnode, node_map_ptr, graph_proto);
        break;
    }
  }
}

void OnnxExporter::ExportPrimReshape(const FuncGraphPtr &, const CNodePtr &node,
                                     std::map<AnfNodePtr, std::string> *node_map_ptr,
                                     onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_shape = node->input(kTwoNum);
  std::string name_shape;
  if (input_shape->isa<ValueNode>()) {
    name_shape = RegisterNodeWithUniqueName(input_shape, node_map_ptr);
    onnx::NodeProto *node_proto = graph_proto->add_node();
    auto name = prim::kPrimReshape->name();

    node_proto->set_name(name_shape + name);
    node_proto->add_output(name_shape);
    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("value");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    ConvertTupleToTensor(dyn_cast<ValueNode>(input_shape)->value(), attr_proto->mutable_t());
  } else {
    name_shape = GetNodeInputName(input_shape, node_map_ptr, graph_proto);
    MS_LOG(EXCEPTION) << "Need to insert op convert variable from tuple to tensor for Reshape.";
  }

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type(prim::kPrimReshape->name());
  node_proto->add_output(node_name);
  node_proto->add_input(name_x);
  node_proto->add_input(name_shape);
}

void OnnxExporter::ExportPrimReduce(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, std::string> *node_map_ptr,
                                    onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_axis = node->input(kTwoNum);
  auto keep_dims = GetOpAttribute<bool>(node, "keep_dims");

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  std::string name;
  if (node->IsApply(prim::kPrimReduceSum)) {
    name = "ReduceSum";
  } else if (node->IsApply(prim::kPrimReduceMean)) {
    name = "ReduceMean";
  } else if (node->IsApply(prim::kPrimReduceMax)) {
    name = "ReduceMax";
  } else {
    MS_LOG(EXCEPTION) << "Unsupported reduce op: " << node->ToString();
  }

  std::vector<int64_t> axes;
  if (input_axis->isa<ValueNode>()) {
    auto axis_value = dyn_cast<ValueNode>(input_axis)->value();
    if (axis_value->isa<Int32Imm>()) {
      auto int_ptr = dyn_cast<Int32Imm>(axis_value);
      axes.push_back(int_ptr->value());
    } else if (axis_value->isa<Int64Imm>()) {
      auto int_ptr = dyn_cast<Int64Imm>(axis_value);
      axes.push_back(int_ptr->value());
    } else if (axis_value->isa<ValueTuple>()) {
      auto tuple_ptr = dyn_cast<ValueTuple>(axis_value);
      axes = GetValue<std::vector<int64_t>>(tuple_ptr);
    } else {
      MS_LOG(EXCEPTION) << "Cannot convert value " << axis_value->ToString() << " of type "
                        << axis_value->type()->ToString() << " for \"axes\" attribute of " << name;
    }
  } else {
    MS_LOG(EXCEPTION) << "Need to insert op convert variable from tuple to attributes for " << name;
  }

  AddReduceOp(name, input_data, node_name, axes, keep_dims, graph_proto);
}

void OnnxExporter::ExportPrimReduceAnyOrAll(const FuncGraphPtr &, const CNodePtr &node,
                                            std::map<AnfNodePtr, std::string> *node_map_ptr,
                                            onnx::GraphProto *const graph_proto) {
  auto input_data_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_axis = node->input(kTwoNum);
  auto keep_dims = GetOpAttribute<bool>(node, "keep_dims");
  auto reduce_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  std::string target_node_name = "";
  if (node->IsApply(prim::kPrimReduceAny)) {
    target_node_name = "ReduceSum";
  } else if (node->IsApply(prim::kPrimReduceAll)) {
    target_node_name = "ReduceMin";
  } else {
    MS_LOG(EXCEPTION) << "Unsupported reduce op: " << node->ToString();
  }

  std::string cast_name = GenerateUniqueName();  // Insert cast op
  onnx::NodeProto *cast_proto = graph_proto->add_node();
  cast_proto->add_input(input_data_name);
  cast_proto->add_output(cast_name);
  cast_proto->set_op_type(prim::kPrimCast->name());
  onnx::AttributeProto *attr_proto = cast_proto->add_attribute();
  attr_proto->set_name("to");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto->set_i(GetOnnxDataType(TypeId::kNumberTypeFloat32));

  std::vector<int64_t> axes;
  if (input_axis->isa<ValueNode>()) {
    auto axis_value = dyn_cast<ValueNode>(input_axis)->value();
    if (axis_value->isa<Int32Imm>()) {
      auto int_ptr = dyn_cast<Int32Imm>(axis_value);
      axes.push_back(int_ptr->value());
    } else if (axis_value->isa<Int64Imm>()) {
      auto int_ptr = dyn_cast<Int64Imm>(axis_value);
      axes.push_back(int_ptr->value());
    } else if (axis_value->isa<ValueTuple>()) {
      auto tuple_ptr = dyn_cast<ValueTuple>(axis_value);
      axes = GetValue<std::vector<int64_t>>(tuple_ptr);
      if (axes.empty()) {
        const auto &x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape())->shape();
        for (size_t i = 0; i < x_shape.size(); ++i) {
          axes.push_back(static_cast<int64_t>(i));
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "Cannot convert value " << axis_value->ToString() << " of type "
                        << axis_value->type()->ToString() << " for \"axes\" attribute of " << target_node_name;
    }
  } else {
    MS_LOG(EXCEPTION) << "Need to insert op convert variable from tuple to attributes for " << target_node_name;
  }

  std::string greater_name = GenerateUniqueName();
  onnx::TensorProto *zero_initializer_proto = graph_proto->add_initializer();
  auto zero_input_name = greater_name + "_zero";
  zero_initializer_proto->set_name(zero_input_name);
  zero_initializer_proto->set_data_type(GetOnnxDataType(kNumberTypeFloat32));
  zero_initializer_proto->add_float_data(0);

  AddReduceOp(target_node_name, cast_name, greater_name, axes, keep_dims, graph_proto);

  onnx::NodeProto *greater_node_proto = graph_proto->add_node();  // Insert greater op
  greater_node_proto->add_input(greater_name);
  greater_node_proto->add_input(zero_input_name);
  greater_node_proto->add_output(reduce_name);
  greater_node_proto->set_op_type(prim::kPrimGreater->name());
}

void OnnxExporter::ExportPrimTranspose(const FuncGraphPtr &, const CNodePtr &node,
                                       std::map<AnfNodePtr, std::string> *node_map_ptr,
                                       onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_perm = node->input(kTwoNum);
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  auto name = prim::kPrimTranspose->name();

  node_proto->set_name(node_name + name);
  node_proto->set_op_type(name);
  node_proto->add_output(node_name);
  node_proto->add_input(input_data);

  if (input_perm->isa<ValueNode>()) {
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("perm");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    auto perm_value = dyn_cast<ValueNode>(input_perm)->value();
    auto int_ptr = dyn_cast<Int32Imm>(perm_value);
    if (int_ptr == nullptr) {
      auto tuple_ptr = dyn_cast<ValueTuple>(perm_value);
      MS_EXCEPTION_IF_NULL(tuple_ptr);
      for (size_t i = 0; i < tuple_ptr->size(); ++i) {
        attr_proto->add_ints(GetValue<int64_t>((*tuple_ptr)[i]));
      }
    } else {
      attr_proto->add_ints(int_ptr->value());
    }
  } else {
    MS_LOG(EXCEPTION) << "The input input_perm of Transpose is not a ValueNode! "
                      << "Need to insert op convert variable from tuple to attributes for " << name;
  }
}

/*
  See:
    - mindspore/ccsrc/backend/kernel_compiler/cpu/stridedslice_cpu_kernel.cc
    - mindspore/ccsrc/backend/kernel_compiler/common_utils.cc
 */
void OnnxExporter::ExportPrimStridedSlice(const FuncGraphPtr &, const CNodePtr &node,
                                          std::map<AnfNodePtr, std::string> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto name = node_name + prim::kPrimStridedSlice->name();

  auto begin = node->input(kTwoNum);
  if (!begin->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The input begin of StridedSlice is not a ValueNode! "
                      << "Need to insert op convert variable from tuple to tensor for " << name;
  }
  auto begin_value_node = dyn_cast<ValueNode>(begin);
  auto begin_value = GetValue<std::vector<int64_t>>(begin_value_node->value());
  auto begin_ignore_mask = GetOpAttribute<int64_t>(node, "begin_mask");
  for (size_t i = 0; i < begin_value.size(); ++i) {
    if ((static_cast<uint64_t>(begin_ignore_mask) & (1UL << i)) != 0) {
      begin_value[i] = 0;
    }
  }

  auto end = node->input(kThreeNum);
  if (!end->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The input end of StridedSlice is not a ValueNode! "
                      << "Need to insert op convert variable from tuple to tensor for " << name;
  }
  auto end_value_node = dyn_cast<ValueNode>(end);
  auto end_value = GetValue<std::vector<int64_t>>(end_value_node->value());
  const auto &x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape())->shape();
  auto end_ignore_mask = GetOpAttribute<int64_t>(node, "end_mask");
  for (size_t i = 0; i < end_value.size(); ++i) {
    if ((static_cast<uint64_t>(end_ignore_mask) & (1UL << i)) != 0) {
      end_value[i] = x_shape[i];
    }
  }

  std::vector<int64_t> axes_value;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    axes_value.push_back(static_cast<int64_t>(i));
  }

  auto strides = node->input(kFourNum);
  if (!strides->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The input strides of StridedSlice is not a ValueNode! "
                      << "Need to insert op convert variable from tuple to tensor for " << name;
  }
  auto strides_value_node = dyn_cast<ValueNode>(strides);
  auto strides_value = GetValue<std::vector<int64_t>>(strides_value_node->value());

  auto shrink_axis_mask = GetOpAttribute<int64_t>(node, "shrink_axis_mask");
  for (size_t i = 0; i < end_value.size(); ++i) {
    if ((static_cast<uint64_t>(shrink_axis_mask) & (1UL << i)) != 0) {
      strides_value[i] = end_value[i] > begin_value[i] ? 1 : -1;
      end_value[i] = begin_value[i] + strides_value[i];
    }
  }

  auto slice_name = node_name;
  if (shrink_axis_mask != 0) {
    slice_name = node_name + "__reshape";
  }

  AddSliceOp(input_data, slice_name, begin_value, end_value, axes_value, strides_value, graph_proto);

  if (shrink_axis_mask != 0) {
    onnx::NodeProto *squeeze_op = graph_proto->add_node();
    squeeze_op->set_op_type("Squeeze");
    squeeze_op->add_input(slice_name);
    squeeze_op->add_output(node_name);
    onnx::AttributeProto *axes_attr = squeeze_op->add_attribute();
    axes_attr->set_name("axes");
    axes_attr->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (size_t i = 0; i < x_shape.size(); ++i) {
      if ((static_cast<uint64_t>(shrink_axis_mask) & (1UL << i)) != 0) {
        axes_attr->add_ints(static_cast<int64_t>(i));
      }
    }
  }
}

onnx::NodeProto *OnnxExporter::PrimResizeExportHelper(const FuncGraphPtr &, const CNodePtr &node,
                                                      std::map<AnfNodePtr, std::string> *node_map_ptr,
                                                      onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape());

  AnfNodePtr op = node->input(kZeroNum);
  auto op_value = dyn_cast<ValueNode>(op);
  auto prim = dyn_cast<Primitive>(op_value->value());
  std::vector<int64_t> resize_size;

  auto tuple_ptr = dyn_cast<ValueSequence>(prim->GetAttr("size"));  // size may be Tuple or List
  if (tuple_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Got null pointer, currently the " << prim->name()
                      << " operator in your model is not support for exporting onnx.";
  }

  for (size_t i = 0; i < x_shape->shape().size() - kTwoNum; i++) {
    resize_size.push_back(x_shape->shape()[i]);
  }
  for (size_t i = 0; i < tuple_ptr->size(); i++) {
    ValuePtr elem = (*tuple_ptr)[i];
    resize_size.push_back(dyn_cast<Int64Imm>(elem)->value());
  }
  auto resize_size_ptr = MakeValue<std::vector<int64_t>>(resize_size);
  auto size = NewValueNode(resize_size_ptr)->cast<AnfNodePtr>();

  auto name_size = RegisterNodeWithUniqueName(size, node_map_ptr);
  onnx::NodeProto *node_proto_size = graph_proto->add_node();
  node_proto_size->add_output(name_size);
  node_proto_size->set_op_type("Constant");
  onnx::AttributeProto *attr_proto = node_proto_size->add_attribute();
  attr_proto->set_name("value");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  ConvertTupleToTensor(resize_size_ptr, attr_proto->mutable_t());

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  onnx::TensorProto *roi_initializer_proto = graph_proto->add_initializer();
  auto roi_name = node_name + "roi_initializer";
  roi_initializer_proto->set_name(roi_name);
  roi_initializer_proto->set_data_type(GetOnnxDataType(kNumberTypeFloat32));
  roi_initializer_proto->add_dims(0);

  onnx::TensorProto *scales_initializer_proto = graph_proto->add_initializer();
  auto scales_name = node_name + "scales_initializer";
  scales_initializer_proto->set_name(scales_name);
  scales_initializer_proto->set_data_type(GetOnnxDataType(kNumberTypeFloat32));
  scales_initializer_proto->add_dims(0);

  onnx::NodeProto *node_proto = graph_proto->add_node();

  node_proto->set_op_type("Resize");
  node_proto->add_output(node_name);
  node_proto->add_input(input_data);
  node_proto->add_input(roi_name);
  node_proto->add_input(scales_name);
  node_proto->add_input(name_size);

  return node_proto;
}

void OnnxExporter::ExportPrimResizeNearestNeighbor(const FuncGraphPtr &graph, const CNodePtr &node,
                                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                                   onnx::GraphProto *const graph_proto) {
  onnx::NodeProto *node_proto = PrimResizeExportHelper(graph, node, node_map_ptr, graph_proto);

  auto align_corners = GetOpAttribute<bool>(node, "align_corners");
  std::string coordinate_transformation_mode = align_corners ? "align_corners" : "asymmetric";
  // `nearest_mode` is based on ResizeNearestNeighborCPUKernel::LaunchKernel in
  // mindspore/ccsrc/backend/kernel_compiler/cpu/resize_nearest_neighbor_cpu_kernel.cc
  std::string nearest_mode = align_corners ? "round_prefer_ceil" : "floor";

  onnx::AttributeProto *coordinate_mode_proto = node_proto->add_attribute();
  coordinate_mode_proto->set_name("coordinate_transformation_mode");
  coordinate_mode_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  coordinate_mode_proto->set_s(coordinate_transformation_mode);

  onnx::AttributeProto *nearest_mode_proto = node_proto->add_attribute();
  nearest_mode_proto->set_name("nearest_mode");
  nearest_mode_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  nearest_mode_proto->set_s(nearest_mode);
}

void OnnxExporter::ExportPrimResizeBilinear(const FuncGraphPtr &graph, const CNodePtr &node,
                                            std::map<AnfNodePtr, std::string> *node_map_ptr,
                                            onnx::GraphProto *const graph_proto) {
  onnx::NodeProto *node_proto = PrimResizeExportHelper(graph, node, node_map_ptr, graph_proto);

  auto align_corners = GetOpAttribute<bool>(node, "align_corners");
  std::string coordinate_transformation_mode = align_corners ? "align_corners" : "asymmetric";

  onnx::AttributeProto *coordinate_mode_proto = node_proto->add_attribute();
  coordinate_mode_proto->set_name("coordinate_transformation_mode");
  coordinate_mode_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  coordinate_mode_proto->set_s(coordinate_transformation_mode);

  onnx::AttributeProto *mode_proto = node_proto->add_attribute();
  mode_proto->set_name("mode");
  mode_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  mode_proto->set_s("linear");
}

// MindSpore ExpandDims -> ONNX Reshape
void OnnxExporter::ExportPrimExpandDims(const FuncGraphPtr &, const CNodePtr &node,
                                        std::map<AnfNodePtr, std::string> *node_map_ptr,
                                        onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto axis = GetInt64Value(node->input(kTwoNum));
  auto x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape());
  auto name = prim::kPrimExpandDims->name();

  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < x_shape->shape().size(); i++) {
    new_shape.push_back(x_shape->shape()[i]);
  }
  if (axis < 0) {
    axis = axis + kOneNumLong + SizeToLong(x_shape->shape().size());
  }
  (void)new_shape.insert(new_shape.begin() + axis, kOneNum);
  auto new_shape_value = MakeValue<std::vector<int64_t>>(new_shape);
  auto shape = NewValueNode(new_shape_value)->cast<AnfNodePtr>();
  std::string name_shape;

  if (shape->isa<ValueNode>()) {
    name_shape = RegisterNodeWithUniqueName(shape, node_map_ptr);
    onnx::NodeProto *node_proto = graph_proto->add_node();
    node_proto->add_output(name_shape);
    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("value");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    ConvertTupleToTensor(dyn_cast<ValueNode>(shape)->value(), attr_proto->mutable_t());
  } else {
    name_shape = GetNodeInputName(shape, node_map_ptr, graph_proto);
    MS_LOG(EXCEPTION) << "Need to insert op convert variable from tuple to tensor for " << name;
  }

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Reshape");
  node_proto->add_output(node_name);
  node_proto->add_input(input_x);
  node_proto->add_input(name_shape);
}

// MindSpore GatherD -> ONNX GatherElements
void OnnxExporter::ExportPrimGatherD(const FuncGraphPtr &, const CNodePtr &node,
                                     std::map<AnfNodePtr, std::string> *node_map_ptr,
                                     onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto axis = GetInt64Value(node->input(kTwoNum));
  auto input_indices = GetNodeInputName(node->input(kThreeNum), node_map_ptr, graph_proto);
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("GatherElements");
  node_proto->add_output(node_name);
  node_proto->add_input(input_x);
  node_proto->add_input(input_indices);
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_name("axis");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto->set_i(static_cast<::google::protobuf::int64>(axis));
}

// MindSpore Pad -> ONNX Pad
void OnnxExporter::ExportPrimPad(const FuncGraphPtr &, const CNodePtr &node,
                                 std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  auto paddings = GetOpAttributePtr<ValueTuple>(node, "paddings");
  std::vector<std::vector<int64_t>> paddings_values = GetValue<std::vector<std::vector<int64_t>>>(paddings);
  std::vector<int64_t> pads_sequence;
  for (size_t i = 0; i < paddings_values.size(); ++i) {
    pads_sequence.push_back(paddings_values[i][0]);
  }
  for (size_t j = 0; j < paddings_values.size(); ++j) {
    pads_sequence.push_back(paddings_values[j][1]);
  }
  auto pads_ptr = MakeValue<std::vector<int64_t>>(pads_sequence);
  auto pads = NewValueNode(pads_ptr)->cast<AnfNodePtr>();

  auto pads_name = RegisterNodeWithUniqueName(pads, node_map_ptr);
  onnx::NodeProto *pads_node = graph_proto->add_node();
  pads_node->add_output(pads_name);
  pads_node->set_op_type("Constant");
  onnx::AttributeProto *pads_attr_proto = pads_node->add_attribute();
  pads_attr_proto->set_name("value");
  pads_attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  ConvertTupleToTensor(pads_ptr, pads_attr_proto->mutable_t());

  auto ms_pad_node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *onnx_pad_node = graph_proto->add_node();
  onnx_pad_node->set_op_type("Pad");
  onnx_pad_node->add_output(ms_pad_node_name);
  onnx_pad_node->add_input(x_name);
  onnx_pad_node->add_input(pads_name);
}

// MindSpore BatchMatMul -> ONNX Transpose + MatMul
void OnnxExporter::ExportPrimBatchMatMul(const FuncGraphPtr &, const CNodePtr &node,
                                         std::map<AnfNodePtr, std::string> *node_map_ptr,
                                         onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_y = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);

  AnfNodePtr batchmatmul_op = node->input(kZeroNum);
  auto op_value = dyn_cast<ValueNode>(batchmatmul_op);
  auto prim = dyn_cast<Primitive>(op_value->value());
  auto transpose_a = GetValue<bool>(prim->GetAttr("transpose_a"));
  auto transpose_b = GetValue<bool>(prim->GetAttr("transpose_b"));
  std::string transpose_input_x_name = "";
  std::string transpose_input_y_name = "";

  if (transpose_a) {
    auto input_x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape());
    // Add Transpose node after input_x of BatchMatMul
    transpose_input_x_name = GenerateUniqueName();
    onnx::NodeProto *transpose_inputx_node_proto = graph_proto->add_node();
    transpose_inputx_node_proto->add_input(input_x);
    transpose_inputx_node_proto->add_output(transpose_input_x_name);
    transpose_inputx_node_proto->set_op_type(prim::kPrimTranspose->name());
    onnx::AttributeProto *attr_proto = transpose_inputx_node_proto->add_attribute();
    attr_proto->set_name("perm");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (size_t i = 0; i < input_x_shape->shape().size() - kTwoNum; i++) {
      attr_proto->add_ints(SizeToLong(i));
    }
    attr_proto->add_ints(SizeToLong(input_x_shape->shape().size()) - IntToLong(kOneNum));
    attr_proto->add_ints(SizeToLong(input_x_shape->shape().size()) - IntToLong(kTwoNum));
  }
  if (transpose_b) {
    auto input_y_shape = dyn_cast<abstract::Shape>(node->input(kTwoNum)->Shape());
    // Add Transpose node after input_y of BatchMatMul
    transpose_input_y_name = GenerateUniqueName();
    onnx::NodeProto *transpose_inputy_node_proto = graph_proto->add_node();
    transpose_inputy_node_proto->add_input(input_y);
    transpose_inputy_node_proto->add_output(transpose_input_y_name);
    transpose_inputy_node_proto->set_op_type(prim::kPrimTranspose->name());
    onnx::AttributeProto *attr_proto = transpose_inputy_node_proto->add_attribute();
    attr_proto->set_name("perm");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (size_t i = 0; i < input_y_shape->shape().size() - kTwoNum; i++) {
      attr_proto->add_ints(SizeToLong(i));
    }
    attr_proto->add_ints(SizeToLong(input_y_shape->shape().size()) - IntToLong(kOneNum));
    attr_proto->add_ints(SizeToLong(input_y_shape->shape().size()) - IntToLong(kTwoNum));
  }

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("MatMul");
  node_proto->add_output(node_name);
  node_proto->set_name(node_name + "MatMul");
  if (transpose_a) {
    node_proto->add_input(transpose_input_x_name);
  } else {
    node_proto->add_input(input_x);
  }
  if (transpose_b) {
    node_proto->add_input(transpose_input_y_name);
  } else {
    node_proto->add_input(input_y);
  }
}

// MindSpore BroadcastTo -> ONNX Expand
void OnnxExporter::ExportPrimBroadcastTo(const FuncGraphPtr &, const CNodePtr &node,
                                         std::map<AnfNodePtr, std::string> *node_map_ptr,
                                         onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape());
  auto name = prim::kPrimBroadcastTo->name();

  auto shape_ptr = GetOpAttributePtr<ValueSequeue>(node, "shape");
  auto shape_vec = GetValue<std::vector<int64_t>>(shape_ptr);
  size_t n_shape = shape_vec.size();

  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < n_shape; i++) {
    if (shape_vec[i] == -kOneNum) {
      size_t ids = i + x_shape->shape().size() - n_shape;
      new_shape.push_back(x_shape->shape()[ids]);
    } else {
      new_shape.push_back(shape_vec[i]);
    }
  }

  auto new_shape_value = MakeValue<std::vector<int64_t>>(new_shape);
  auto shape = NewValueNode(new_shape_value)->cast<AnfNodePtr>();
  std::string name_shape;

  if (shape->isa<ValueNode>()) {
    name_shape = RegisterNodeWithUniqueName(shape, node_map_ptr);
    onnx::NodeProto *node_proto = graph_proto->add_node();
    node_proto->add_output(name_shape);
    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("value");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    ConvertTupleToTensor(dyn_cast<ValueNode>(shape)->value(), attr_proto->mutable_t());
  } else {
    name_shape = GetNodeInputName(shape, node_map_ptr, graph_proto);
    MS_LOG(EXCEPTION) << "Need to insert op convert variable from tuple to tensor for " << name;
  }

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Expand");
  node_proto->add_output(node_name);
  node_proto->add_input(input_x);
  node_proto->add_input(name_shape);
}

// MindSpore AddN -> ONNX Add
void OnnxExporter::ExportPrimAddN(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr,
                                  onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto input_node = node->input(kOneNum)->cast<CNodePtr>();
  auto last_input_name = GetNodeInputName(input_node->input(kOneNum), node_map_ptr, graph_proto);
  for (size_t i = kTwoNum; i < input_node->inputs().size() - 1; ++i) {
    auto input_name = GetNodeInputName(input_node->input(i), node_map_ptr, graph_proto);
    auto tmp_end_name = node_name + "ADD_" + std::to_string(i);
    AddOp("Add", {last_input_name, input_name}, {tmp_end_name}, graph_proto);
    last_input_name = tmp_end_name;
  }
  auto input_end_name = GetNodeInputName(input_node->input(input_node->inputs().size() - 1), node_map_ptr, graph_proto);
  AddOp("Add", {last_input_name, input_end_name}, {node_name}, graph_proto);
}

// MindSpore GeLU -> ONNX 0.5 * X * (1.0 + tanh((sqrt(2/pi) * (x + 0.044715 * pow(x, 3)))))
void OnnxExporter::ExportPrimGeLU(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr,
                                  onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto onnx_type = GetOutputType(node->input(kOneNum));

  // Add pow node
  auto pow_name = GenerateUniqueName();
  auto exp_node_name = pow_name + "exponent_initializer";
  AddFloatTensor1DInitializer(exp_node_name, {3.0}, onnx_type, graph_proto);
  AddOp("Pow", {input_x, exp_node_name}, {pow_name}, graph_proto);

  // Add first Mul Node
  auto fmul_name = GenerateUniqueName();
  auto fmul_input_node_name = fmul_name + "input_y_for_mul_initializer";
  AddFloatTensor1DInitializer(fmul_input_node_name, {0.044715}, onnx_type, graph_proto);
  AddOp("Mul", {pow_name, fmul_input_node_name}, {fmul_name}, graph_proto);

  // Add first Add node
  auto fadd_name = GenerateUniqueName();
  AddOp("Add", {input_x, fmul_name}, {fadd_name}, graph_proto);

  // Add second Mul Node
  auto smul_name = GenerateUniqueName();
  auto smul_input_node_name = smul_name + "input_y_for_smul_initializer";
  AddFloatTensor1DInitializer(smul_input_node_name, {0.7978845608}, onnx_type, graph_proto);
  AddOp("Mul", {fadd_name, smul_input_node_name}, {smul_name}, graph_proto);

  // Add tanh node
  auto tanh_name = GenerateUniqueName();
  AddOp("Tanh", {smul_name}, {tanh_name}, graph_proto);

  // Add second Add node
  auto sadd_name = GenerateUniqueName();
  auto sadd_input_node_name = sadd_name + "input_y_for_sadd_initializer";
  AddFloatTensor1DInitializer(sadd_input_node_name, {1.0}, onnx_type, graph_proto);
  AddOp("Add", {tanh_name, sadd_input_node_name}, {sadd_name}, graph_proto);

  // Add third Mul Node
  auto tmul_name = GenerateUniqueName();
  auto tmul_input_node_name = tmul_name + "input_y_for_tmul_initializer";
  AddFloatTensor1DInitializer(tmul_input_node_name, {0.5}, onnx_type, graph_proto);
  AddOp("Mul", {sadd_name, tmul_input_node_name}, {tmul_name}, graph_proto);

  // Add fourth Mul Node
  auto fomul_node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  AddOp("Mul", {input_x, tmul_name}, {fomul_node_name}, graph_proto);
}

void OnnxExporter::ExportPrimConcat(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, std::string> *node_map_ptr,
                                    onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  // Get inputs first: otherwise if an input is a constant, topological order will break
  auto input_node = node->input(kOneNum)->cast<CNodePtr>();
  std::vector<std::string> input_names;
  if (input_node->IsApply(prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < input_node->inputs().size(); ++i) {
      auto input_name = GetNodeInputName(input_node->input(i), node_map_ptr, graph_proto);
      input_names.push_back(input_name);
    }
  } else {
    auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
    input_names.push_back(input_data);
  }

  AddConcatOp(input_names, node_name, GetOpAttribute<int64_t>(node, "axis"), graph_proto);
}

void OnnxExporter::ExportPrimCast(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr,
                                  onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_type = node->input(kTwoNum);

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type(prim::kPrimCast->name());
  node_proto->add_output(node_name);
  node_proto->add_input(input_data);

  if (input_type->isa<ValueNode>()) {
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("to");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
    auto type_value = dyn_cast<ValueNode>(input_type)->value();
    auto type_ptr = dyn_cast<Type>(type_value);
    MS_EXCEPTION_IF_NULL(type_ptr);
    attr_proto->set_i(GetOnnxDataType(type_ptr->type_id()));
  } else {
    MS_LOG(EXCEPTION) << "Need to convert MindSpore Cast input(1) to ONNX Cast to attribute.";
  }
}

void OnnxExporter::ExportPrimPReLU(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_slope = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);

  auto x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape());
  auto slope_shape = dyn_cast<abstract::Shape>(node->input(kTwoNum)->Shape());
  MS_EXCEPTION_IF_NULL(x_shape);
  MS_EXCEPTION_IF_NULL(slope_shape);

  // format of x is NCHW, input format is NCHW, if length of input_slope is 1, insert Unsqueeze [1,2]
  if (x_shape->shape().size() == kFourNum && slope_shape->shape().size() == kOneNum) {
    auto node_name = GenerateUniqueName();
    onnx::NodeProto *node_proto = graph_proto->add_node();
    node_proto->set_op_type("Unsqueeze");
    node_proto->add_output(node_name);

    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    attr_proto->set_name("axes");
    attr_proto->add_ints(kOneNum);
    attr_proto->add_ints(kTwoNum);

    node_proto->add_input(input_slope);
    input_slope = node_name;
  }

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("PRelu");
  node_proto->add_output(node_name);
  node_proto->add_input(input_x);
  node_proto->add_input(input_slope);
}

void OnnxExporter::ExportPrimReLU6(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto onnx_input_type = GetOutputType(node->input(kOneNum));
  AddClipOp(input_x_name, node_name, 0.0f, 6.0f, onnx_input_type, graph_proto);
}

void OnnxExporter::ExportPrimDepthwiseConv2d(const FuncGraphPtr &, const CNodePtr &node,
                                             std::map<AnfNodePtr, std::string> *node_map_ptr,
                                             onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_w = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape());
  auto w_shape = dyn_cast<abstract::Shape>(node->input(kTwoNum)->Shape());
  MS_EXCEPTION_IF_NULL(x_shape);
  MS_EXCEPTION_IF_NULL(w_shape);
  if (x_shape->shape().size() != kFourNum || w_shape->shape().size() != kFourNum) {
    MS_LOG(EXCEPTION) << "DepthwiseConv2d input shape should be 4d.";
  }
  if (w_shape->shape()[kZeroNum] != kOneNum && w_shape->shape()[kOneNum] != kOneNum) {
    MS_LOG(EXCEPTION) << "DepthwiseConv2d weight shape[0] != 1 and shape[1] != 1, cannot reshape";
  }
  // create w_shape constant node
  auto node_name = GenerateUniqueName();
  onnx::NodeProto *node_proto = graph_proto->add_node();
  auto name_w_shape = node_name;
  node_proto->add_output(name_w_shape);
  node_proto->set_op_type("Constant");
  // create Value Tensor
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_name("value");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  tensor_proto->add_dims(static_cast<::google::protobuf::int64>(w_shape->shape().size()));
  tensor_proto->set_data_type(onnx::TensorProto_DataType_INT64);
  // reshape
  tensor_proto->add_int64_data(w_shape->shape()[kOneNum]);
  tensor_proto->add_int64_data(w_shape->shape()[kZeroNum]);
  tensor_proto->add_int64_data(w_shape->shape()[kTwoNum]);
  tensor_proto->add_int64_data(w_shape->shape()[kThreeNum]);

  // add reshape node
  node_name = GenerateUniqueName();
  node_proto = graph_proto->add_node();
  node_proto->set_op_type(prim::kPrimReshape->name());
  node_proto->add_input(input_w);
  node_proto->add_input(name_w_shape);
  input_w = node_name;
  node_proto->add_output(input_w);

  // add conv node
  node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  node_proto = graph_proto->add_node();
  node_proto->set_op_type("Conv");
  node_proto->add_input(input_x);
  node_proto->add_input(input_w);
  node_proto->add_output(node_name);
  // set attributes
  AnfNodePtr op = node->input(0);
  auto op_value = dyn_cast<ValueNode>(op);
  auto prim = dyn_cast<Primitive>(op_value->value());
  // set dilations
  onnx::AttributeProto *onnx_attr_proto = node_proto->add_attribute();
  onnx_attr_proto->set_name("dilations");
  SetAttrTupleValueToProto<2>(prim->GetAttr("dilation"), onnx::AttributeProto_AttributeType_INTS, onnx_attr_proto,
                              prim);
  // set group
  onnx_attr_proto = node_proto->add_attribute();
  onnx_attr_proto->set_name("group");
  onnx_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  onnx_attr_proto->set_i(x_shape->shape()[1]);
  // set kernel_shape
  onnx_attr_proto = node_proto->add_attribute();
  onnx_attr_proto->set_name("kernel_shape");
  SetAttrTupleValueToProto<0>(prim->GetAttr("kernel_size"), onnx::AttributeProto_AttributeType_INTS, onnx_attr_proto,
                              prim);

  // set pad
  onnx_attr_proto = node_proto->add_attribute();
  int64_t attr_value;
  CheckAndConvertUtils::GetPadModEnumValue(prim->GetAttr("pad_mode"), &attr_value);
  onnx_attr_proto->set_name("auto_pad");
  onnx_attr_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  if (attr_value == PadMode::VALID) {
    onnx_attr_proto->set_s("VALID");
  } else if (attr_value == PadMode::SAME) {
    onnx_attr_proto->set_s("SAME_UPPER");
  } else {
    onnx_attr_proto->set_name("pads");
    SetAttrTupleValueToProto(prim->GetAttr("pad_list"), onnx::AttributeProto_AttributeType_INTS, onnx_attr_proto, prim);
  }
  // set strides
  onnx_attr_proto = node_proto->add_attribute();
  onnx_attr_proto->set_name("strides");
  SetAttrTupleValueToProto<2>(prim->GetAttr("stride"), onnx::AttributeProto_AttributeType_INTS, onnx_attr_proto, prim);
}

void OnnxExporter::ExportPrimTile(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr,
                                  onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto multiples = node->input(kTwoNum);
  std::string name_multiples;
  if (multiples->isa<ValueNode>()) {
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_multiples = RegisterNodeWithUniqueName(multiples, node_map_ptr);
    node_proto->add_output(name_multiples);
    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("value");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    ConvertTupleToTensor(dyn_cast<ValueNode>(multiples)->value(), attr_proto->mutable_t());
  } else {
    name_multiples = GetNodeInputName(multiples, node_map_ptr, graph_proto);
    MS_LOG(EXCEPTION) << "Need to insert op convert variable from tuple to tensor for Tile.";
  }

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Tile");
  node_proto->add_output(node_name);
  node_proto->add_input(name_x);
  node_proto->add_input(name_multiples);
}

void OnnxExporter::ExportPrimSquare(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, std::string> *node_map_ptr,
                                    onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto name_exponent = GenerateUniqueName();
  onnx::NodeProto *node_proto_exp = graph_proto->add_node();
  node_proto_exp->add_output(name_exponent);

  node_proto_exp->set_op_type("Constant");
  onnx::AttributeProto *attr_proto = node_proto_exp->add_attribute();
  attr_proto->set_name("value");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  const float exponent_value = 2.0;
  tensor_proto->set_name("exponent");
  tensor_proto->add_dims(static_cast<::google::protobuf::int64>(1));
  tensor_proto->set_data_type(GetOnnxDataType(kNumberTypeFloat32));
  tensor_proto->add_float_data(exponent_value);

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Pow");
  node_proto->add_output(node_name);
  node_proto->add_input(name_x);
  node_proto->add_input(name_exponent);
}

void OnnxExporter::ExportPrimGatherV2(const FuncGraphPtr &, const CNodePtr &node,
                                      std::map<AnfNodePtr, std::string> *node_map_ptr,
                                      onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto name_indices = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto axis = node->input(kThreeNum)->cast<ValueNodePtr>()->value();
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Gather");
  node_proto->add_output(node_name);
  node_proto->add_input(name_x);
  node_proto->add_input(name_indices);
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_name("axis");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto->set_i(static_cast<::google::protobuf::int64>(dyn_cast<Int64Imm>(axis)->value()));
}

/*
  This is a workaround for nodes with several outputs used at once
  MatchAndMark cannot help here, because it only supports a single output
  Proposed convention:
    * Nodes with several outputs are registered as
      `(*node_map_ptr)[node] = node_idx;`, just like nodes with a single output
    * Their outputs are named "{node_idx}_{output_idx}"
    * TupleGetItem automatically passes the outputs to the next nodes
  See OnnxExporter::ExportPrimTopK for a usage example
*/
void OnnxExporter::ExportPrimTupleGetItem(const FuncGraphPtr &, const CNodePtr &node,
                                          std::map<AnfNodePtr, std::string> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto) {
  auto index = GetInt64Value(node->input(kTwoNum));
  auto input_node_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_name = MakeOutputName(input_node_name, index);

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Identity");
  node_proto->add_input(input_name);
  node_proto->add_output(node_name);
}

void OnnxExporter::ExportPrimTopK(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr,
                                  onnx::GraphProto *const graph_proto) {
  auto x_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto k_input_name = node_name + "k_initializer";
  auto k = GetInt64Value(node->input(kTwoNum));
  AddInt64Tensor1DInitializer(k_input_name, {k}, graph_proto);

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("TopK");
  node_proto->add_input(x_input_name);
  node_proto->add_input(k_input_name);
  node_proto->add_output(MakeOutputName(node_name, kZeroNum));  // Values
  auto indices_name = MakeOutputName(node_name, kOneNum);
  auto indices_cast_name = indices_name + "_cast";
  node_proto->add_output(indices_cast_name);

  onnx::AttributeProto *sorted_attr_proto = node_proto->add_attribute();
  sorted_attr_proto->set_name("sorted");
  sorted_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  auto sorted = GetOpAttribute<bool>(node, "sorted");
  sorted_attr_proto->set_i(sorted);
  AddCastOp(indices_cast_name, indices_name, onnx::TensorProto_DataType_INT32, graph_proto);
}

// Based on mindspore/ccsrc/backend/kernel_compiler/cpu/boundingbox_decode_cpu_kernel.cc
void OnnxExporter::ExportPrimBoundingBoxDecode(const FuncGraphPtr &, const CNodePtr &node,
                                               std::map<AnfNodePtr, std::string> *node_map_ptr,
                                               onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto anchor_bbox_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto deltas_input_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto onnx_input_type = GetOutputType(node->input(kOneNum));

  auto means = GetOpAttributePtr<ValueTuple>(node, "means");
  std::vector<float> mean_values = GetValue<std::vector<float>>(means);
  auto means_name = node_name + "means_initializer";
  AddFloatTensor1DInitializer(means_name, mean_values, onnx_input_type, graph_proto);

  auto stds = GetOpAttributePtr<ValueTuple>(node, "stds");
  std::vector<float> std_values = GetValue<std::vector<float>>(stds);
  auto stds_name = node_name + "stds_initializer";
  AddFloatTensor1DInitializer(stds_name, std_values, onnx_input_type, graph_proto);

  auto wh_ratio_clip = GetOpAttribute<float>(node, "wh_ratio_clip");
  auto max_ratio = static_cast<float>(std::abs(std::log(wh_ratio_clip)));

  auto unstd_deltas_name = node_name + "unstd_deltas";
  auto sd_to_add_name = unstd_deltas_name + "__add";
  AddOp("Mul", {deltas_input_name, stds_name}, {sd_to_add_name}, graph_proto);
  AddOp("Add", {sd_to_add_name, means_name}, {unstd_deltas_name}, graph_proto);

  auto center_deltas_name = node_name + "center_deltas";
  auto log_scale_deltas_name = node_name + "log_scale_deltas";
  auto lsd_to_clip_name = log_scale_deltas_name + "__clip";
  AddSplitOp(unstd_deltas_name, {center_deltas_name, lsd_to_clip_name}, {kTwoNum, kTwoNum}, 1, graph_proto);
  AddClipOp(lsd_to_clip_name, log_scale_deltas_name, -max_ratio, max_ratio, onnx_input_type, graph_proto);

  auto anchor_starts_name = node_name + "anchor_starts";
  auto anchor_ends_name = node_name + "anchor_ends";
  AddSplitOp(anchor_bbox_input_name, {anchor_starts_name, anchor_ends_name}, {kTwoNum, kTwoNum}, 1, graph_proto);

  auto anchor_centers_name = node_name + "anchor_centers";
  auto anchor_dimensions_name = node_name + "anchor_dimensions";
  ConvertBoxesToXywh(anchor_starts_name, anchor_ends_name, anchor_centers_name, anchor_dimensions_name, onnx_input_type,
                     graph_proto);

  auto anchor_shifts_name = node_name + "anchor_shifts";
  AddOp("Mul", {anchor_dimensions_name, center_deltas_name}, {anchor_shifts_name}, graph_proto);
  auto result_centers_name = node_name + "result_centers";
  AddOp("Add", {anchor_centers_name, anchor_shifts_name}, {result_centers_name}, graph_proto);

  auto anchor_scales_name = node_name + "anchor_scales";
  AddOp("Exp", {log_scale_deltas_name}, {anchor_scales_name}, graph_proto);
  auto result_dimensions_name = node_name + "result_dimensions";
  AddOp("Mul", {anchor_dimensions_name, anchor_scales_name}, {result_dimensions_name}, graph_proto);

  auto result_starts_to_clip_name = node_name + "result_starts_to_clip";
  auto result_ends_to_clip_name = node_name + "result_ends_to_clip";
  ConvertBoxesToXyxy(result_centers_name, result_dimensions_name, result_starts_to_clip_name, result_ends_to_clip_name,
                     onnx_input_type, graph_proto);

  auto max_shape = GetOpAttributePtr<ValueTuple>(node, "max_shape");
  auto max_y = GetValue<int64_t>((*max_shape)[0]);
  auto max_x = GetValue<int64_t>((*max_shape)[1]);
  auto result_start_xs_name = node_name + "result_start_x";
  auto result_start_ys_name = node_name + "result_start_y";
  auto result_end_xs_name = node_name + "result_end_x";
  auto result_end_ys_name = node_name + "result_end_y";
  ClipPointsComponent(result_starts_to_clip_name, result_start_xs_name, static_cast<float>(max_x), 0, onnx_input_type,
                      graph_proto);
  ClipPointsComponent(result_starts_to_clip_name, result_start_ys_name, static_cast<float>(max_y), 1, onnx_input_type,
                      graph_proto);
  ClipPointsComponent(result_ends_to_clip_name, result_end_xs_name, static_cast<float>(max_x), 0, onnx_input_type,
                      graph_proto);
  ClipPointsComponent(result_ends_to_clip_name, result_end_ys_name, static_cast<float>(max_y), 1, onnx_input_type,
                      graph_proto);

  AddConcatOp({result_start_xs_name, result_start_ys_name, result_end_xs_name, result_end_ys_name}, node_name, kOneNum,
              graph_proto);
}

void OnnxExporter::ExportPrimNMSWithMask(const FuncGraphPtr &, const CNodePtr &node,
                                         std::map<AnfNodePtr, std::string> *node_map_ptr,
                                         onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto bboxes_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto iou_threshold = GetOpAttribute<float>(node, "iou_threshold");
  auto selected_boxes_output_name = MakeOutputName(node_name, kZeroNum);
  auto selected_idx_output_name = MakeOutputName(node_name, kOneNum);
  auto selected_mask_output_name = MakeOutputName(node_name, kTwoNum);
  auto onnx_input_type = GetOutputType(node->input(kOneNum));

  // Preprocessing

  auto boxes_count_name = node_name + "max_output_boxes";
  auto max_output_boxes_to_squeeze_name = boxes_count_name + "_to_reshape";
  auto input_shape_name = node_name + "input_shape";
  AddOp("Shape", {bboxes_input_name}, {input_shape_name}, graph_proto);
  AddSliceOp(input_shape_name, max_output_boxes_to_squeeze_name, {0}, {1}, {0}, {1}, graph_proto);
  AddReshapeOp(max_output_boxes_to_squeeze_name, boxes_count_name, {}, graph_proto);

  auto scores_name = node_name + "scores";
  auto flat_scores_name = scores_name + "_flat";
  auto sorted_scores_name = flat_scores_name + "_sorted";
  auto scores_to_flatten_name = scores_name + "_to_reshape";
  auto descending_order_name = node_name + "descending_indices";
  const int BBOX_NUM_EL = 4;
  AddSliceOp(bboxes_input_name, scores_to_flatten_name, {BBOX_NUM_EL}, {BBOX_NUM_EL + 1}, {1}, {1}, graph_proto);
  AddReshapeOp(scores_to_flatten_name, flat_scores_name, {-1}, graph_proto);
  AddOp("TopK", {flat_scores_name, max_output_boxes_to_squeeze_name}, {sorted_scores_name, descending_order_name},
        graph_proto);
  AddReshapeOp(sorted_scores_name, scores_name, {1, 1, -1}, graph_proto);
  auto iou_threshold_name = node_name + "iou_threshold_initializer";
  AddFloatScalarInitializer(iou_threshold_name, iou_threshold, onnx::TensorProto_DataType_FLOAT, graph_proto);

  AddOp("Gather", {bboxes_input_name, descending_order_name}, {selected_boxes_output_name},
        graph_proto);  // Output 0: boxes
  auto boxes_name = node_name + "boxes";
  auto boxes_to_reshape_name = boxes_name + "_to_reshape";
  AddSliceOp(selected_boxes_output_name, boxes_to_reshape_name, {0}, {BBOX_NUM_EL}, {1}, {1}, graph_proto);
  AddReshapeOp(boxes_to_reshape_name, boxes_name, {1, -1, BBOX_NUM_EL}, graph_proto);

  if (onnx_input_type == onnx::TensorProto_DataType_FLOAT16) {
    auto fp32_boxes_name = boxes_name + "_fp32";
    AddCastOp(boxes_name, fp32_boxes_name, onnx::TensorProto_DataType_FLOAT, graph_proto);
    boxes_name = fp32_boxes_name;

    auto fp32_scores_name = scores_name + "_fp32";
    AddCastOp(scores_name, fp32_scores_name, onnx::TensorProto_DataType_FLOAT, graph_proto);
    scores_name = fp32_scores_name;
  }

  // NMS op

  auto selected_indices_name = node_name + "selected_indices";
  AddOp("NonMaxSuppression", {boxes_name, scores_name, boxes_count_name, iou_threshold_name}, {selected_indices_name},
        graph_proto);

  // Output 1: indices

  auto flat_indices_name = node_name + "flat_indices";
  auto flat_indices_to_squeeze_name = flat_indices_name + "__reshape";
  const int BOX_INDEX_POS = 2;
  AddSliceOp(selected_indices_name, flat_indices_to_squeeze_name, {BOX_INDEX_POS}, {BOX_INDEX_POS + 1}, {1}, {1},
             graph_proto);
  AddReshapeOp(flat_indices_to_squeeze_name, flat_indices_name, {-1}, graph_proto);

  auto zero_name = node_name + "zero_initializer";
  onnx::TensorProto *zero_initializer = graph_proto->add_initializer();
  zero_initializer->set_name(zero_name);
  zero_initializer->set_data_type(onnx::TensorProto_DataType_INT32);
  zero_initializer->add_int32_data(0);
  auto one_name = node_name + "one_initializer";
  onnx::TensorProto *one_initializer = graph_proto->add_initializer();
  one_initializer->set_name(one_name);
  one_initializer->set_data_type(onnx::TensorProto_DataType_INT32);
  one_initializer->add_int32_data(1);
  auto int32_boxes_count_name = boxes_count_name + "_int32";
  AddCastOp(boxes_count_name, int32_boxes_count_name, onnx::TensorProto_DataType_INT32, graph_proto);
  AddOp("Range", {zero_name, int32_boxes_count_name, one_name}, {selected_idx_output_name}, graph_proto);

  // Output 2: mask

  auto empty_mask_name = selected_mask_output_name + "__scatter";
  onnx::TensorProto *empty_mask_value_proto =
    AddConstantOfShapeOp(max_output_boxes_to_squeeze_name, empty_mask_name, graph_proto);
  empty_mask_value_proto->set_data_type(onnx::TensorProto_DataType_BOOL);
  empty_mask_value_proto->add_int32_data(0);

  auto true_elements_name = node_name + "true";
  auto true_elements_shape_name = true_elements_name + "_shape";
  AddOp("Shape", {flat_indices_name}, {true_elements_shape_name}, graph_proto);
  onnx::TensorProto *true_elements_value_proto =
    AddConstantOfShapeOp(true_elements_shape_name, true_elements_name, graph_proto);
  true_elements_value_proto->set_data_type(onnx::TensorProto_DataType_BOOL);
  true_elements_value_proto->add_int32_data(1);

  AddOp("ScatterElements", {empty_mask_name, flat_indices_name, true_elements_name}, {selected_mask_output_name},
        graph_proto);
}

void OnnxExporter::ExportPrimSplit(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  auto axis = GetOpAttribute<int64_t>(node, "axis");
  auto output_num = GetOpAttribute<int64_t>(node, "output_num");
  if (output_num == 0) {
    MS_LOG(EXCEPTION) << "output_num must be > 0";
  }
  const auto &input_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape())->shape();

  if (axis < 0 || static_cast<size_t>(axis) >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "`axis` is out of range";
  }
  if (input_shape[static_cast<size_t>(axis)] % output_num != 0) {
    MS_LOG(EXCEPTION) << "Input dim is not divisible by `output_num`";
  }

  onnx::NodeProto *split_proto = graph_proto->add_node();
  split_proto->set_op_type("Split");
  split_proto->add_input(input_name);
  for (int64_t i = 0; i < output_num; ++i) {
    split_proto->add_output(MakeOutputName(node_name, i));
  }

  onnx::AttributeProto *axis_attr_proto = split_proto->add_attribute();
  axis_attr_proto->set_name("axis");
  axis_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  axis_attr_proto->set_i(axis);

  onnx::AttributeProto *split_attr_proto = split_proto->add_attribute();
  split_attr_proto->set_name("split");
  split_attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
  for (int64_t i = 0; i < output_num; ++i) {
    split_attr_proto->add_ints(input_shape[static_cast<size_t>(axis)] / output_num);
  }
}

/*
  Based on mindspore-project/mindspore/ccsrc/backend/kernel_compiler/cpu/roi_align_cpu_kernel.cc
  Notes:
    * MS version uses avg pool, leaving corresponding ONNX attr as is
    * MS has two ROI end modes, implemented with pre-processing
 */
void OnnxExporter::ExportPrimROIAlign(const FuncGraphPtr &, const CNodePtr &node,
                                      std::map<AnfNodePtr, std::string> *node_map_ptr,
                                      onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto features_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto rois_input_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto onnx_input_type = GetOutputType(node->input(kOneNum));

  auto roi_indices_name = node_name + "roi_indices";
  auto roi_indices_column_name = roi_indices_name + "_column";
  auto roi_starts_name = node_name + "roi_starts";
  auto roi_ends_name = node_name + "roi_ends";
  AddSplitOp(rois_input_name, {roi_indices_column_name, roi_starts_name, roi_ends_name}, {1, kTwoNum, kTwoNum}, 1,
             graph_proto);

  // Indices transformation

  auto flat_roi_indices_name = roi_indices_name + "_flat";
  AddReshapeOp(roi_indices_column_name, flat_roi_indices_name, {-1}, graph_proto);
  auto int_roi_indices_name = roi_indices_name + "_int";
  // This should be fine if indices are whole numbers less than 2^23
  AddCastOp(flat_roi_indices_name, int_roi_indices_name, onnx::TensorProto_DataType_INT64, graph_proto);

  // ROI end mode

  auto roi_end_mode = GetOpAttribute<int64_t>(node, "roi_end_mode");
  auto roi_end_mode_name = node_name + "roi_end_mode_initializer";
  AddFloatScalarInitializer(roi_end_mode_name, static_cast<float>(roi_end_mode), onnx_input_type, graph_proto);

  auto corrected_roi_ends_name = roi_ends_name + "_corrected";
  AddOp("Add", {roi_ends_name, roi_end_mode_name}, {corrected_roi_ends_name}, graph_proto);

  // Contatenate ROIs

  auto corrected_rois_name = node_name + "corrected_rois";
  AddConcatOp({roi_starts_name, corrected_roi_ends_name}, corrected_rois_name, kOneNum, graph_proto);

  // RoiAlign op

  onnx::NodeProto *roi_align_proto = graph_proto->add_node();
  roi_align_proto->set_op_type("RoiAlign");
  roi_align_proto->add_input(features_input_name);
  roi_align_proto->add_input(corrected_rois_name);
  roi_align_proto->add_input(int_roi_indices_name);
  roi_align_proto->add_output(node_name);
  onnx::AttributeProto *height_attr_proto = roi_align_proto->add_attribute();
  height_attr_proto->set_name("output_height");
  height_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  height_attr_proto->set_i(GetOpAttribute<int64_t>(node, "pooled_height"));
  onnx::AttributeProto *width_attr_proto = roi_align_proto->add_attribute();
  width_attr_proto->set_name("output_width");
  width_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  width_attr_proto->set_i(GetOpAttribute<int64_t>(node, "pooled_width"));
  onnx::AttributeProto *scale_attr_proto = roi_align_proto->add_attribute();
  scale_attr_proto->set_name("spatial_scale");
  scale_attr_proto->set_type(onnx::AttributeProto_AttributeType_FLOAT);
  scale_attr_proto->set_f(GetOpAttribute<float>(node, "spatial_scale"));
  onnx::AttributeProto *sampling_ratio_attr_proto = roi_align_proto->add_attribute();
  sampling_ratio_attr_proto->set_name("sampling_ratio");
  sampling_ratio_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  sampling_ratio_attr_proto->set_i(GetOpAttribute<int64_t>(node, "sample_num"));
}

void OnnxExporter::ExportPrimSlice(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto begin_input_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto size_input_name = GetNodeInputName(node->input(kThreeNum), node_map_ptr, graph_proto);

  auto end_name = node_name + "end";
  AddOp("Add", {begin_input_name, size_input_name}, {end_name}, graph_proto);
  AddOp("Slice", {input_x_name, begin_input_name, end_name}, {node_name}, graph_proto);
}

void OnnxExporter::ExportPrimOnesLike(const FuncGraphPtr &, const CNodePtr &node,
                                      std::map<AnfNodePtr, std::string> *node_map_ptr,
                                      onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  auto shape_name = node_name + "shape";
  AddOp("Shape", {input_x_name}, {shape_name}, graph_proto);

  auto dtype = node->input(kOneNum)->Type();
  auto elem_type = dyn_cast<TensorType>(dtype)->element()->type_id();

  onnx::TensorProto *one_proto = AddConstantOfShapeOp(shape_name, node_name, graph_proto);
  switch (elem_type) {
    case kNumberTypeInt32:
      one_proto->set_data_type(onnx::TensorProto_DataType_INT32);
      one_proto->add_int32_data(1);
      break;
    case kNumberTypeInt64:
      one_proto->set_data_type(onnx::TensorProto_DataType_INT64);
      one_proto->add_int64_data(1);
      break;
    case kNumberTypeFloat32:
      one_proto->set_data_type(onnx::TensorProto_DataType_FLOAT);
      one_proto->add_float_data(1.0f);
      break;
    case kNumberTypeFloat64:
      one_proto->set_data_type(onnx::TensorProto_DataType_DOUBLE);
      one_proto->add_double_data(1.0);
      break;
    default:
      MS_LOG(EXCEPTION) << "Unsupported dtype: " << elem_type;
  }
}

void OnnxExporter::ExportPrimScatterNd(const FuncGraphPtr &, const CNodePtr &node,
                                       std::map<AnfNodePtr, std::string> *node_map_ptr,
                                       onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input_indices_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_update_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto input_shape_name = GetNodeInputName(node->input(kThreeNum), node_map_ptr, graph_proto);
  auto node_zero_tensor_name = node_name + "_zero";
  auto dtype = node->input(kTwoNum)->Type();
  auto elem_type = dyn_cast<TensorType>(dtype)->element()->type_id();

  onnx::TensorProto *zero_proto = AddConstantOfShapeOp(input_shape_name, node_zero_tensor_name, graph_proto);
  switch (elem_type) {
    case kNumberTypeInt32:
      zero_proto->set_data_type(onnx::TensorProto_DataType_INT32);
      zero_proto->add_int32_data(0);
      break;
    case kNumberTypeInt64:
      zero_proto->set_data_type(onnx::TensorProto_DataType_INT64);
      zero_proto->add_int64_data(0);
      break;
    case kNumberTypeFloat32:
      zero_proto->set_data_type(onnx::TensorProto_DataType_FLOAT);
      zero_proto->add_float_data(0.0f);
      break;
    case kNumberTypeFloat64:
      zero_proto->set_data_type(onnx::TensorProto_DataType_DOUBLE);
      zero_proto->add_double_data(0.0);
      break;
    default:
      MS_LOG(EXCEPTION) << "Unsupported dtype: " << elem_type;
  }
  auto int64_indices_name = input_indices_name + "_int64";
  AddCastOp(input_indices_name, int64_indices_name, onnx::TensorProto_DataType_INT64, graph_proto);

  // Create ScatterND node
  onnx::NodeProto *scatternd_proto = graph_proto->add_node();
  scatternd_proto->set_op_type("ScatterND");
  scatternd_proto->add_input(node_zero_tensor_name);
  scatternd_proto->add_input(int64_indices_name);
  scatternd_proto->add_input(input_update_name);
  scatternd_proto->add_output(node_name);
}

void OnnxExporter::ExportPrimArgMaxWithValue(const FuncGraphPtr &, const CNodePtr &node,
                                             std::map<AnfNodePtr, std::string> *node_map_ptr,
                                             onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto axis = GetOpAttribute<int64_t>(node, "axis");
  auto keep_dims = GetOpAttribute<bool>(node, "keep_dims");

  auto indices_output_name = MakeOutputName(node_name, kZeroNum);
  auto indices_cast_name = indices_output_name + "_cast";

  onnx::NodeProto *argmax_proto = graph_proto->add_node();
  argmax_proto->set_op_type("ArgMax");
  argmax_proto->add_input(input_x_name);
  argmax_proto->add_output(indices_cast_name);
  onnx::AttributeProto *argmax_axis_attr_proto = argmax_proto->add_attribute();
  argmax_axis_attr_proto->set_name("axis");
  argmax_axis_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  argmax_axis_attr_proto->set_i(axis);
  onnx::AttributeProto *argmax_keepdims_attr_proto = argmax_proto->add_attribute();
  argmax_keepdims_attr_proto->set_name("keepdims");
  argmax_keepdims_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  argmax_keepdims_attr_proto->set_i(keep_dims);

  AddCastOp(indices_cast_name, indices_output_name, onnx::TensorProto_DataType_INT32, graph_proto);

  auto max_output_name = MakeOutputName(node_name, kOneNum);
  AddReduceOp("ReduceMax", input_x_name, max_output_name, {axis}, keep_dims, graph_proto);
}

void OnnxExporter::ExportPrimArgMinWithValue(const FuncGraphPtr &, const CNodePtr &node,
                                             std::map<AnfNodePtr, std::string> *node_map_ptr,
                                             onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto axis = GetOpAttribute<int64_t>(node, "axis");
  auto keep_dims = GetOpAttribute<bool>(node, "keep_dims");

  auto indices_output_name = MakeOutputName(node_name, kZeroNum);
  auto indices_cast_name = indices_output_name + "_cast";

  onnx::NodeProto *argmax_proto = graph_proto->add_node();
  argmax_proto->set_op_type("ArgMin");
  argmax_proto->add_input(input_x_name);
  argmax_proto->add_output(indices_cast_name);
  onnx::AttributeProto *argmax_axis_attr_proto = argmax_proto->add_attribute();
  argmax_axis_attr_proto->set_name("axis");
  argmax_axis_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  argmax_axis_attr_proto->set_i(axis);
  onnx::AttributeProto *argmax_keepdims_attr_proto = argmax_proto->add_attribute();
  argmax_keepdims_attr_proto->set_name("keepdims");
  argmax_keepdims_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  argmax_keepdims_attr_proto->set_i(keep_dims);

  AddCastOp(indices_cast_name, indices_output_name, onnx::TensorProto_DataType_INT32, graph_proto);

  auto max_output_name = MakeOutputName(node_name, kOneNum);
  AddReduceOp("ReduceMin", input_x_name, max_output_name, {axis}, keep_dims, graph_proto);
}

void OnnxExporter::ExportPrimOneHot(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, std::string> *node_map_ptr,
                                    onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto indices_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto depth_input_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto on_input_name = GetNodeInputName(node->input(kThreeNum), node_map_ptr, graph_proto);
  auto off_input_name = GetNodeInputName(node->input(kFourNum), node_map_ptr, graph_proto);
  auto axis = GetOpAttribute<int64_t>(node, "axis");

  if (GetOutputType(node->input(kOneNum)) == onnx::TensorProto_DataType_INT32) {
    auto indices_cast_name = node_name + "_indices_as_int32";
    AddCastOp(indices_input_name, indices_cast_name, onnx::TensorProto_DataType_INT64, graph_proto);
    indices_input_name = indices_cast_name;
  }

  auto on_1d_name = node_name + "on_1d";
  AddReshapeOp(on_input_name, on_1d_name, {-1}, graph_proto);
  auto off_1d_name = node_name + "off_1d";
  AddReshapeOp(off_input_name, off_1d_name, {-1}, graph_proto);

  auto on_off_name = node_name + "on_off";
  AddConcatOp({off_1d_name, on_1d_name}, on_off_name, kZeroNum, graph_proto);

  onnx::NodeProto *one_hot_proto = graph_proto->add_node();
  one_hot_proto->set_op_type("OneHot");
  one_hot_proto->add_input(indices_input_name);
  one_hot_proto->add_input(depth_input_name);
  one_hot_proto->add_input(on_off_name);
  one_hot_proto->add_output(node_name);
  onnx::AttributeProto *one_hot_axis_attr_proto = one_hot_proto->add_attribute();
  one_hot_axis_attr_proto->set_name("axis");
  one_hot_axis_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  one_hot_axis_attr_proto->set_i(axis);
}

/*
  Based on nn.Conv2dTranspose
  Warning: `output_shape` is an input in MS and an attribute in ONNX. Hence
           it is not possible to change the output shape in runtime
 */
void OnnxExporter::PrimConv2DTransposeExportHelper(const CNodePtr &conv_node, const CNodePtr &bias_add_node,
                                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                                   onnx::GraphProto *const graph_proto) {
  std::string node_name;

  std::vector<AnfNodePtr> inputs{conv_node->input(kOneNum), conv_node->input(kTwoNum)};
  if (bias_add_node != nullptr) {
    inputs.push_back(bias_add_node->input(kTwoNum));
    node_name = RegisterNodeWithUniqueName(bias_add_node, node_map_ptr);
  } else {
    node_name = RegisterNodeWithUniqueName(conv_node, node_map_ptr);
  }

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("ConvTranspose");
  for (const auto &input : inputs) {
    node_proto->add_input(GetNodeInputName(input, node_map_ptr, graph_proto));
  }
  node_proto->add_output(node_name);

  auto prim = GetPrimitive(conv_node);
  auto attrs_convert_info =
    OpNameInfo()
      .Attr("dilation", "dilations", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<kTwoNum>)
      .Attr("group", "group", onnx::AttributeProto_AttributeType_INT, SetAttrValueToProto<Int64Imm>)
      .Attr("kernel_size", "kernel_shape", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<0>)
      .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING, SetConvTransposePadding)
      .Attr("stride", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<kTwoNum>);
  for (const auto &attr_info : attrs_convert_info.op_attrs()) {
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name(attr_info.onnx_attr_name());
    auto ms_attr = GetOpAttributePtr<Value>(conv_node, attr_info.attr_name());
    MS_EXCEPTION_IF_NULL(ms_attr);
    attr_info.fn_gen_attr()(ms_attr, attr_info.onnx_attr_type(), attr_proto, prim);
  }

  // Set output shape

  auto input_shape_node = GetRealInput(conv_node->input(kThreeNum));
  if (!input_shape_node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "For ONNX export third argument must be constant "
                         "(Python tuple). Instead got "
                      << input_shape_node->ToString();
  }
  auto input_shape_value_ptr = input_shape_node->cast<ValueNodePtr>()->value();
  if (!input_shape_value_ptr->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Expected ValueTuple, got " << input_shape_value_ptr->ToString() << " of type "
                      << input_shape_value_ptr->type()->ToString();
  }

  onnx::AttributeProto *output_shape_attr_proto = node_proto->add_attribute();
  output_shape_attr_proto->set_name("output_shape");
  SetAttrTupleValueToProto<0>(input_shape_value_ptr, onnx::AttributeProto_AttributeType_INTS, output_shape_attr_proto,
                              prim);
}

void OnnxExporter::ExportPrimConv2DTranspose(const FuncGraphPtr &, const CNodePtr &node,
                                             std::map<AnfNodePtr, std::string> *node_map_ptr,
                                             onnx::GraphProto *graph_proto) {
  PrimConv2DTransposeExportHelper(node, nullptr, node_map_ptr, graph_proto);
}

void OnnxExporter::ExportPrimGreaterEqual(const FuncGraphPtr &, const CNodePtr &node,
                                          std::map<AnfNodePtr, std::string> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_y_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto less_name = node_name + "less";

  AddOp("Less", {input_x_name, input_y_name}, {less_name}, graph_proto);
  AddOp("Not", {less_name}, {node_name}, graph_proto);
}

void OnnxExporter::ExportPrimLessEqual(const FuncGraphPtr &, const CNodePtr &node,
                                       std::map<AnfNodePtr, std::string> *node_map_ptr,
                                       onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_y_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto greater_name = node_name + "greater";

  AddOp("Greater", {input_x_name, input_y_name}, {greater_name}, graph_proto);
  AddOp("Not", {greater_name}, {node_name}, graph_proto);
}

void OnnxExporter::ExportPrimSqueeze(const FuncGraphPtr &, const CNodePtr &node,
                                     std::map<AnfNodePtr, std::string> *node_map_ptr,
                                     onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Squeeze");
  node_proto->add_input(input_name);
  node_proto->add_output(node_name);

  auto axes = GetOpAttributePtr<ValueSequence>(node, "axis");
  auto axes_value = GetValue<std::vector<int64_t>>(axes);
  if (!axes_value.empty()) {
    onnx::AttributeProto *axes_proto = node_proto->add_attribute();
    axes_proto->set_name("axes");
    axes_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (auto axis : axes_value) {
      axes_proto->add_ints(axis);
    }
  }
}

void MakeLSTMWeight(const std::string &input, const std::string &output, const std::vector<int64_t> &output_shape,
                    onnx::GraphProto *graph_proto) {
  auto reshaped_name = output + "__split";
  AddReshapeOp(input, reshaped_name, output_shape, graph_proto);

  auto split_i_name = output + "__concat_i";
  auto split_o_name = output + "__concat_o";
  auto split_f_name = output + "__concat_f";
  auto split_c_name = output + "__concat_c";
  int64_t hidden_size = output_shape[kOneNum] / kFourNum;
  AddSplitOp(reshaped_name, {split_i_name, split_f_name, split_c_name, split_o_name},
             {hidden_size, hidden_size, hidden_size, hidden_size}, 1, graph_proto);

  AddConcatOp({split_i_name, split_o_name, split_f_name, split_c_name}, output, 1, graph_proto);
}

void MakeLSTMWeight2(const std::string &input, const std::string &output, const std::vector<int64_t> &output_shape,
                     onnx::GraphProto *graph_proto) {
  auto reshaped_name = output + "__split";
  AddReshapeOp(input, reshaped_name, output_shape, graph_proto);

  auto split_i_name = output + "__concat_i";
  auto split_o_name = output + "__concat_o";
  auto split_f_name = output + "__concat_f";
  auto split_c_name = output + "__concat_c";
  int64_t hidden_size = output_shape[kOneNum] / kFourNum;
  AddSplitOp(reshaped_name, {split_i_name, split_c_name, split_f_name, split_o_name},
             {hidden_size, hidden_size, hidden_size, hidden_size}, 1, graph_proto);

  AddConcatOp({split_i_name, split_o_name, split_f_name, split_c_name}, output, 1, graph_proto);
}

void OnnxExporter::ExportPrimDynamicRNN(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                        std::map<AnfNodePtr, std::string> *node_map_ptr,
                                        onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto x_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto weight_input_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto bias_input_name = GetNodeInputName(node->input(kThreeNum), node_map_ptr, graph_proto);
  auto init_h_input_name = GetNodeInputName(node->input(kFiveNum), node_map_ptr, graph_proto);
  auto init_c_input_name = GetNodeInputName(node->input(kSixNum), node_map_ptr, graph_proto);

  auto hidden_size = GetOpAttribute<int64_t>(node, "hidden_size");
  auto direction_input = GetOpAttribute<std::string>(node, "direction");
  auto x_input_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape())->shape();
  auto seq_len = x_input_shape[0];
  auto batch_size = x_input_shape[1];
  auto num_dir = direction_input == "UNIDIRECTIONAL" ? 1 : 2;
  auto input_size = x_input_shape[kTwoNum];

  auto onnx_input_weights_name = node_name + "_onnx_input_weights";
  auto onnx_hidden_weights_name = node_name + "_onnx_hidden_weights";
  auto onnx_bias_name = node_name + "_onnx_bias";

  const int num_gates = 4;
  auto gate_size = num_gates * hidden_size;

  auto weight_input_name_reshape = weight_input_name + "_reshape";
  AddReshapeOp(weight_input_name, weight_input_name_reshape, {(input_size * gate_size) + (hidden_size * gate_size)},
               graph_proto);

  auto input_weights_name = node_name + "_input_weights";
  auto hidden_weights_name = node_name + "_hidden_weights";
  std::vector<int64_t> split_sizes = {input_size * gate_size, hidden_size * gate_size};
  std::vector<std::string> split_outputs = {input_weights_name, hidden_weights_name};

  AddSplitOp(weight_input_name_reshape, split_outputs, split_sizes, 0, graph_proto);

  auto input_weights_name_reshape = input_weights_name + "_reshape";
  auto hidden_weights_name_reshape = hidden_weights_name + "_reshape";
  AddReshapeOp(input_weights_name, input_weights_name_reshape, {num_dir, input_size, gate_size}, graph_proto);
  AddReshapeOp(hidden_weights_name, hidden_weights_name_reshape, {num_dir, hidden_size, gate_size}, graph_proto);

  // Transpose input_weights_name
  onnx::NodeProto *transpose_node_proto_1 = graph_proto->add_node();
  auto input_weights_name_reshape_transposed = input_weights_name_reshape + "_transposed";
  transpose_node_proto_1->set_name(input_weights_name_reshape_transposed);
  transpose_node_proto_1->set_op_type("Transpose");
  transpose_node_proto_1->add_input(input_weights_name_reshape);
  transpose_node_proto_1->add_output(input_weights_name_reshape_transposed);

  onnx::AttributeProto *perm_proto_1 = transpose_node_proto_1->add_attribute();
  perm_proto_1->set_name("perm");
  perm_proto_1->set_type(onnx::AttributeProto_AttributeType_INTS);
  perm_proto_1->add_ints(kZeroNum);
  perm_proto_1->add_ints(kTwoNum);
  perm_proto_1->add_ints(kOneNum);

  // Transpose  hidden_weights_name
  onnx::NodeProto *transpose_node_proto_2 = graph_proto->add_node();
  auto hidden_weights_name_reshape_transposed = hidden_weights_name_reshape + "_transposed";
  transpose_node_proto_2->set_name(hidden_weights_name_reshape_transposed);
  transpose_node_proto_2->set_op_type("Transpose");
  transpose_node_proto_2->add_input(hidden_weights_name_reshape);
  transpose_node_proto_2->add_output(hidden_weights_name_reshape_transposed);

  onnx::AttributeProto *perm_proto_2 = transpose_node_proto_2->add_attribute();
  perm_proto_2->set_name("perm");
  perm_proto_2->set_type(onnx::AttributeProto_AttributeType_INTS);
  perm_proto_2->add_ints(kZeroNum);
  perm_proto_2->add_ints(kTwoNum);
  perm_proto_2->add_ints(kOneNum);

  MakeLSTMWeight2(input_weights_name_reshape_transposed, onnx_input_weights_name, {num_dir, gate_size, input_size},
                  graph_proto);
  MakeLSTMWeight2(hidden_weights_name_reshape_transposed, onnx_hidden_weights_name, {num_dir, gate_size, hidden_size},
                  graph_proto);

  auto bias_input_name_reshape = bias_input_name + "_reshape";
  AddReshapeOp(bias_input_name, bias_input_name_reshape, {num_dir, gate_size}, graph_proto);

  auto bias_output_name = node_name + "_bias_output_name";
  MakeLSTMWeight2(bias_input_name_reshape, bias_output_name, {num_dir, gate_size}, graph_proto);

  auto bias_concat = bias_output_name + "_concat";
  std::vector<std::string> concat_inputs = {bias_output_name, bias_output_name};
  AddConcatOp(concat_inputs, bias_concat, 1, graph_proto);

  auto div_second_operand_name = node_name + "_div_second_operand";
  const float div_second_operand = 2.0;
  AddFloatScalarInitializer(div_second_operand_name, div_second_operand, onnx::TensorProto_DataType_FLOAT16,
                            graph_proto);

  AddOp("Div", {bias_concat, div_second_operand_name}, {onnx_bias_name}, graph_proto);

  // Create LSTM node
  onnx::NodeProto *lstm_node_proto = graph_proto->add_node();
  lstm_node_proto->set_op_type("LSTM");
  lstm_node_proto->add_input(x_input_name);
  lstm_node_proto->add_input(onnx_input_weights_name);
  lstm_node_proto->add_input(onnx_hidden_weights_name);
  lstm_node_proto->add_input(onnx_bias_name);
  lstm_node_proto->add_input("");
  lstm_node_proto->add_input(init_h_input_name);
  lstm_node_proto->add_input(init_c_input_name);

  auto Y_output_name = node_name + "_Y";
  auto Y_h_output_name = node_name + "_Y_h";
  auto Y_c_output_name = node_name + "_Y_c";
  lstm_node_proto->add_output(Y_output_name);
  lstm_node_proto->add_output(Y_h_output_name);
  lstm_node_proto->add_output(Y_c_output_name);

  onnx::AttributeProto *hidden_size_proto = lstm_node_proto->add_attribute();
  hidden_size_proto->set_name("hidden_size");
  hidden_size_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  hidden_size_proto->set_i(hidden_size);

  auto output_name_Y = MakeOutputName(node_name, kZeroNum);
  auto output_name_Y_h = MakeOutputName(node_name, kOneNum);
  auto output_name_Y_c = MakeOutputName(node_name, kTwoNum);
  AddReshapeOp(Y_output_name, output_name_Y, {seq_len, batch_size, num_dir * hidden_size}, graph_proto);
  AddExpandOp(Y_h_output_name, output_name_Y_h, {seq_len, batch_size, hidden_size}, graph_proto);
  AddExpandOp(Y_c_output_name, output_name_Y_c, {seq_len, batch_size, hidden_size}, graph_proto);
}

void ExportLSTMWeights(const CNodePtr &node, const std::string &node_name, const std::string &weights_name,
                       onnx::TensorProto_DataType dtype, const std::string &onnx_input_weights_name,
                       const std::string &onnx_hidden_weights_name, const std::string &onnx_bias_name,
                       onnx::GraphProto *graph_proto) {
  auto input_size = GetOpAttribute<int64_t>(node, "input_size");
  auto hidden_size = GetOpAttribute<int64_t>(node, "hidden_size");
  auto num_layers = GetOpAttribute<int64_t>(node, "num_layers");
  auto has_bias = GetOpAttribute<bool>(node, "has_bias");
  auto bidirectional = GetOpAttribute<bool>(node, "bidirectional");
  auto num_dir = 1 + static_cast<int>(bidirectional);
  auto num_gates = 4;
  auto gate_size = num_gates * hidden_size;

  if (num_layers != 1) {
    MS_LOG(EXCEPTION) << "Converter for multilayer LSTM is not implemented";
  }
  if (bidirectional) {
    MS_LOG(EXCEPTION) << "Bidirectional mode for P.LSTM is not implemented";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto target_device = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target_device != "CPU" && target_device != "GPU") {
    MS_LOG(EXCEPTION) << "Unsupported target device: " << target_device;
  }

  auto input_weights_name = node_name + "_input_weights";
  auto hidden_weights_name = node_name + "_hidden_weights";
  auto input_bias_name = node_name + "_input_bias";
  auto hidden_bias_name = node_name + "_hidden_bias";

  std::vector<int64_t> split_sizes = {input_size * gate_size, hidden_size * gate_size};
  std::vector<std::string> split_outputs = {input_weights_name, hidden_weights_name};
  if (has_bias) {
    if (target_device == "GPU") {
      (void)split_sizes.insert(split_sizes.end(), {gate_size, gate_size});
      (void)split_outputs.insert(split_outputs.end(), {input_bias_name, hidden_bias_name});
    } else if (target_device == "CPU") {
      split_sizes.push_back(gate_size);
      split_outputs.push_back(input_bias_name);
    } else {
      MS_LOG(EXCEPTION) << "Impossible branch";
    }
  }
  AddSplitOp(weights_name, split_outputs, split_sizes, 0, graph_proto);

  MakeLSTMWeight(input_weights_name, onnx_input_weights_name, {num_dir, gate_size, input_size}, graph_proto);
  MakeLSTMWeight(hidden_weights_name, onnx_hidden_weights_name, {num_dir, gate_size, hidden_size}, graph_proto);
  if (has_bias) {
    auto onnx_input_bias_name = node_name + "_onnx_input_bias";
    auto onnx_hidden_bias_name = node_name + "_onnx_hidden_bias";
    if (target_device == "GPU") {
      MakeLSTMWeight(input_bias_name, onnx_input_bias_name, {num_dir, gate_size}, graph_proto);
      MakeLSTMWeight(hidden_bias_name, onnx_hidden_bias_name, {num_dir, gate_size}, graph_proto);
    } else if (target_device == "CPU") {
      MakeLSTMWeight(input_bias_name, onnx_input_bias_name, {num_dir, gate_size}, graph_proto);
      auto bias_shape_name = node_name + "_bias_shape";
      AddOp("Shape", {onnx_input_bias_name}, {bias_shape_name}, graph_proto);
      onnx::TensorProto *zero_padding = AddConstantOfShapeOp(bias_shape_name, onnx_hidden_bias_name, graph_proto);
      zero_padding->set_data_type(dtype);
      if (dtype == onnx::TensorProto_DataType_FLOAT16) {
        zero_padding->add_int32_data(0);  // float 0 and int 0 have identical representations
      } else if (dtype == onnx::TensorProto_DataType_FLOAT) {
        zero_padding->add_float_data(0.0f);
      } else {
        MS_LOG(EXCEPTION) << "Unsupported type: " << dtype;
      }
    } else {
      MS_LOG(EXCEPTION) << "Impossible branch";
    }
    AddConcatOp({onnx_input_bias_name, onnx_hidden_bias_name}, onnx_bias_name, 1, graph_proto);
  }
}

void OnnxExporter::ExportPrimLSTM(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr,
                                  onnx::GraphProto *const graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  // MS inputs
  auto x_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto init_h_input_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto init_c_input_name = GetNodeInputName(node->input(kThreeNum), node_map_ptr, graph_proto);

  auto hidden_size = GetOpAttribute<int64_t>(node, "hidden_size");
  auto has_bias = GetOpAttribute<bool>(node, "has_bias");
  auto bidirectional = GetOpAttribute<bool>(node, "bidirectional");
  std::string direction = bidirectional ? "bidirectional" : "forward";
  auto x_input_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape())->shape();
  auto seq_len = x_input_shape[0];
  auto batch_size = x_input_shape[1];
  auto num_dir = 1 + static_cast<int>(bidirectional);

  auto weights_name = GetNodeInputName(node->input(kFourNum), node_map_ptr, graph_proto);
  auto dtype = GetOutputType(node->input(kOneNum));
  auto onnx_input_weights_name = node_name + "_onnx_input_weights";
  auto onnx_hidden_weights_name = node_name + "_onnx_hidden_weights";
  auto onnx_bias_name = node_name + "_onnx_bias";

  ExportLSTMWeights(node, node_name, weights_name, dtype, onnx_input_weights_name, onnx_hidden_weights_name,
                    onnx_bias_name, graph_proto);

  // Create LSTM node
  onnx::NodeProto *lstm_node_proto = graph_proto->add_node();
  lstm_node_proto->set_op_type("LSTM");
  lstm_node_proto->add_input(x_input_name);
  lstm_node_proto->add_input(onnx_input_weights_name);
  lstm_node_proto->add_input(onnx_hidden_weights_name);
  lstm_node_proto->add_input(has_bias ? onnx_bias_name : "");
  lstm_node_proto->add_input("");  // seqlens
  lstm_node_proto->add_input(init_h_input_name);
  lstm_node_proto->add_input(init_c_input_name);

  auto Y_output_name = node_name + "_Y";
  lstm_node_proto->add_output(Y_output_name);
  lstm_node_proto->add_output(MakeOutputName(node_name, kOneNum));
  lstm_node_proto->add_output(MakeOutputName(node_name, kTwoNum));

  onnx::AttributeProto *hidden_size_proto = lstm_node_proto->add_attribute();
  hidden_size_proto->set_name("hidden_size");
  hidden_size_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  hidden_size_proto->set_i(hidden_size);

  onnx::AttributeProto *direction_proto = lstm_node_proto->add_attribute();
  direction_proto->set_name("direction");
  direction_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
  direction_proto->set_s(direction);

  // Transpose 1st output of the LSTM node
  onnx::NodeProto *transpose_node_proto = graph_proto->add_node();
  auto transpose_node_name = node_name + "_Y_transposed";
  transpose_node_proto->set_name(transpose_node_name);
  transpose_node_proto->set_op_type("Transpose");
  transpose_node_proto->add_input(Y_output_name);
  transpose_node_proto->add_output(transpose_node_name);

  onnx::AttributeProto *perm_proto = transpose_node_proto->add_attribute();
  perm_proto->set_name("perm");
  perm_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
  perm_proto->add_ints(kZeroNum);
  perm_proto->add_ints(kTwoNum);
  perm_proto->add_ints(kOneNum);
  perm_proto->add_ints(kThreeNum);

  // Reshape
  auto output_name = MakeOutputName(node_name, kZeroNum);
  AddReshapeOp(transpose_node_name, output_name, {seq_len, batch_size, num_dir * hidden_size}, graph_proto);
}

void OnnxExporter::ExportPrimReverseV2(const FuncGraphPtr &, const CNodePtr &node,
                                       std::map<AnfNodePtr, std::string> *node_map_ptr,
                                       onnx::GraphProto *const graph_proto) {
  auto output = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  auto axes_ptr = GetOpAttributePtr<ValueSequeue>(node, "axis");
  auto axes_vec = GetValue<std::vector<int64_t>>(axes_ptr);
  size_t n_axes = axes_vec.size();
  auto shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape())->shape();

  std::vector<int64_t> starts_vec(n_axes, -1);
  std::vector<int64_t> ends_vec(n_axes);
  (void)std::transform(axes_vec.begin(), axes_vec.end(), ends_vec.begin(),
                       [&shape](size_t ax) { return -shape.at(ax) - 1; });
  std::vector<int64_t> steps_vec(n_axes, -1);

  AddSliceOp(input, output, starts_vec, ends_vec, axes_vec, steps_vec, graph_proto);
}

void OnnxExporter::ExportPrimTensorCopySlices(const FuncGraphPtr &, const CNodePtr &node,
                                              std::map<AnfNodePtr, std::string> *node_map_ptr,
                                              onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto x_input = node->input(kOneNum);
  auto value_input = node->input(kTwoNum);

  auto x_input_name = GetNodeInputName(x_input, node_map_ptr, graph_proto);
  auto value_input_name = GetNodeInputName(value_input, node_map_ptr, graph_proto);

  const auto &x_shape = dyn_cast<abstract::Shape>(x_input->Shape())->shape();
  const auto &value_shape = dyn_cast<abstract::Shape>(value_input->Shape())->shape();

  auto begin_node = dyn_cast<ValueNode>(node->input(kThreeNum));
  MS_EXCEPTION_IF_NULL(begin_node);
  auto begin = GetValue<std::vector<int64_t>>(begin_node->value());

  auto end_node = dyn_cast<ValueNode>(node->input(kFourNum));
  MS_EXCEPTION_IF_NULL(end_node);
  auto end = GetValue<std::vector<int64_t>>(end_node->value());

  auto strides_node = dyn_cast<ValueNode>(node->input(kFiveNum));
  MS_EXCEPTION_IF_NULL(strides_node);
  auto strides = GetValue<std::vector<int64_t>>(strides_node->value());

  MS_EXCEPTION_IF_CHECK_FAIL(
    begin.size() == end.size() && end.size() == strides.size() && strides.size() <= x_shape.size(),
    "Sizes of begin, end, and strides must be equal");
  // MindSpore only allows contuguous slices of memory
  // Contiguous slice size follows the pattern: [1, ..., 1, n, :, ..., :]
  bool found_slice = false;
  for (size_t i = 0; i < begin.size(); ++i) {
    int64_t dim = end[i] - begin[i];
    if (!found_slice && dim != 1) {
      found_slice = true;
    } else if (found_slice && dim != x_shape[i]) {
      MS_LOG(EXCEPTION) << "Slice must be contiguous";
    }
  }
  for (auto stride : strides) {
    MS_EXCEPTION_IF_CHECK_FAIL(stride == 1, "Slice must be contiguous");
  }

  int64_t flat_begin_index = RavelIndex(begin, x_shape);

  std::vector<int64_t> end_inclusive;
  (void)std::transform(end.begin(), end.end(), std::back_inserter(end_inclusive), [](auto x) { return x - 1; });
  (void)std::transform(x_shape.begin() + static_cast<int64_t>(end.size()), x_shape.end(),
                       std::back_inserter(end_inclusive), [](auto x) { return x - 1; });
  int64_t flat_end_index = RavelIndex(end_inclusive, x_shape) + 1;

  int64_t x_size = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
  int64_t value_size = std::accumulate(value_shape.begin(), value_shape.end(), 1, std::multiplies<int64_t>());
  MS_EXCEPTION_IF_CHECK_FAIL(value_size == flat_end_index - flat_begin_index, "Cannot copy 'value' to target slice");

  auto flat_x_name = node_name + "_flat_x";
  AddReshapeOp(x_input_name, flat_x_name, {-1}, graph_proto);
  auto begin_slice_name = node_name + "_begin_slice";
  AddSliceOp(flat_x_name, begin_slice_name, {0}, {static_cast<int64_t>(flat_begin_index)}, {0}, {1}, graph_proto);
  auto end_slice_name = node_name + "_end_slice";
  AddSliceOp(flat_x_name, end_slice_name, {static_cast<int64_t>(flat_end_index)}, {x_size}, {0}, {1}, graph_proto);

  auto flat_value_name = node_name + "_flat_value";
  AddReshapeOp(value_input_name, flat_value_name, {-1}, graph_proto);

  auto flat_result_name = node_name + "_flat_result";
  AddConcatOp({begin_slice_name, flat_value_name, end_slice_name}, flat_result_name, 0, graph_proto);
  AddReshapeOp(flat_result_name, node_name, x_shape, graph_proto);
}

void OnnxExporter::ExportPrimStack(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_name(node_name + "Stack");
  node_proto->set_op_type("ConcatFromSequence");
  node_proto->add_input(input_name);
  node_proto->add_output(node_name);

  onnx::AttributeProto *axis_proto = node_proto->add_attribute();
  axis_proto->set_name("axis");
  axis_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  axis_proto->set_i(GetOpAttribute<int64_t>(node, "axis"));

  onnx::AttributeProto *new_axis_proto = node_proto->add_attribute();
  new_axis_proto->set_name("new_axis");
  new_axis_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  new_axis_proto->set_i(true);
}

void OnnxExporter::ExportPrimAtan2(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto input_node1_anf = node->input(kOneNum);
  auto input_node2_anf = node->input(kTwoNum);
  auto input_node1 = GetNodeInputName(input_node1_anf, node_map_ptr, graph_proto);
  auto input_node2 = GetNodeInputName(input_node2_anf, node_map_ptr, graph_proto);
  auto atan_node = "Atan2_" + node_name + "_atan";
  auto div_node = "Atan2_" + node_name + "_div";
  auto less_node = "Atan2_" + node_name + "_less";
  auto zero_value = "Atan2_" + node_name + "_zero";
  auto neg_pi_value = "Atan2_" + node_name + "_pi";
  auto minimal_value = "Atan2_" + node_name + "_minimal_val";
  auto sign_node = "Atan2_" + node_name + "_sign";
  auto mul_node = "Atan2_" + node_name + "_mul";
  auto less_where_node1 = "Atan2_" + node_name + "_less_then_else1";
  auto add_node = "Atan2_" + node_name + "_add1";
  if (!(IsFloatDataType(input_node1_anf) && IsFloatDataType(input_node2_anf))) {
    auto input_node1_cast = node_name + "_div_cast_fp32_1";
    auto input_node2_cast = node_name + "_div_cast_fp32_2";
    AddCastOp(input_node1, input_node1_cast, onnx::TensorProto_DataType_FLOAT, graph_proto);
    AddCastOp(input_node2, input_node2_cast, onnx::TensorProto_DataType_FLOAT, graph_proto);
    input_node1 = input_node1_cast;
    input_node2 = input_node2_cast;
  }
  AddFloatScalarInitializer(minimal_value, 1e-10, onnx::TensorProto_DataType_FLOAT,
                            graph_proto);  // minimal_value, avoid division by zero
  AddOp("Add", {input_node2, minimal_value}, {add_node}, graph_proto);
  AddOp("Div", {input_node1, add_node}, {div_node}, graph_proto);
  AddOp("Atan", {div_node}, {atan_node}, graph_proto);
  AddFloatScalarInitializer(zero_value, 0, onnx::TensorProto_DataType_FLOAT, graph_proto);
  AddOp("Less", {input_node2, zero_value}, {less_node}, graph_proto);
  AddFloatScalarInitializer(neg_pi_value, -acos(-1), onnx::TensorProto_DataType_FLOAT, graph_proto);  // -PI
  AddOp("Sign", {atan_node}, {sign_node}, graph_proto);
  AddOp("Mul", {neg_pi_value, sign_node}, {mul_node}, graph_proto);
  AddOp("Where", {less_node, mul_node, zero_value}, {less_where_node1}, graph_proto);
  AddOp("Add", {less_where_node1, atan_node}, {node_name}, graph_proto);
}

void OnnxExporter::ExportPrimFloorDiv(const FuncGraphPtr &, const CNodePtr &node,
                                      std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto out_name = node_name;
  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_y_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto onnx_type = GetOutputType(node->input(kOneNum));
  bool is_float = onnx_type == onnx::TensorProto_DataType_FLOAT;

  if (!is_float) {
    auto input_x_name_cast = input_x_name + "_cast";
    auto input_y_name_cast = input_y_name + "_cast";
    AddCastOp(input_x_name, input_x_name_cast, onnx::TensorProto_DataType_FLOAT, graph_proto);
    AddCastOp(input_y_name, input_y_name_cast, onnx::TensorProto_DataType_FLOAT, graph_proto);
    input_x_name = input_x_name_cast;
    input_y_name = input_y_name_cast;
    node_name = node_name + "_floor";
  }

  auto div_name = node_name + "_div";
  AddOp("Div", {input_x_name, input_y_name}, {div_name}, graph_proto);
  AddOp("Floor", {div_name}, {node_name}, graph_proto);

  if (!is_float) {
    AddCastOp(node_name, out_name, onnx_type, graph_proto);
  }
}

void OnnxExporter::ExportPrimFloorMod(const FuncGraphPtr &, const CNodePtr &node,
                                      std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto out_name = node_name;
  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_y_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto onnx_type = GetOutputType(node->input(kOneNum));
  bool is_float = onnx_type == onnx::TensorProto_DataType_FLOAT;

  if (!is_float) {
    auto input_x_name_cast = input_x_name + "_cast";
    auto input_y_name_cast = input_y_name + "_cast";
    AddCastOp(input_x_name, input_x_name_cast, onnx::TensorProto_DataType_FLOAT, graph_proto);
    AddCastOp(input_y_name, input_y_name_cast, onnx::TensorProto_DataType_FLOAT, graph_proto);
    input_x_name = input_x_name_cast;
    input_y_name = input_y_name_cast;
    node_name = node_name + "_sub";
  }

  auto div_name = node_name + "_div";
  auto mul_name = node_name + "_mul";
  auto floor_name = node_name + "_floor";
  AddOp("Div", {input_x_name, input_y_name}, {div_name}, graph_proto);
  AddOp("Floor", {div_name}, {floor_name}, graph_proto);
  AddOp("Mul", {floor_name, input_y_name}, {mul_name}, graph_proto);
  AddOp("Sub", {input_x_name, mul_name}, {node_name}, graph_proto);

  if (!is_float) {
    AddCastOp(node_name, out_name, onnx_type, graph_proto);
  }
}

void OnnxExporter::ExportPrimSort(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);

  auto x_input = node->input(kOneNum);
  auto x_input_name = GetNodeInputName(x_input, node_map_ptr, graph_proto);
  auto x_input_shape = dyn_cast<abstract::Shape>(x_input->Shape())->shape();

  auto axis_attr = GetOpAttribute<int64_t>(node, "axis");
  auto descending_attr = GetOpAttribute<bool>(node, "descending");

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_name(node_name + "TopK");
  node_proto->set_op_type("TopK");
  node_proto->add_input(x_input_name);

  onnx::TensorProto *k_initializer_proto = graph_proto->add_initializer();
  auto k_input_name = "k";
  k_initializer_proto->set_name(k_input_name);
  k_initializer_proto->add_dims(static_cast<int64_t>(1));
  k_initializer_proto->set_data_type(GetOnnxDataType(kNumberTypeInt64));
  int64_t k_index = axis_attr;
  if (axis_attr < 0) {
    k_index += SizeToLong(x_input_shape.size());
  }
  if (k_index > SizeToLong(x_input_shape.size()) - 1 || k_index < 0) {
    MS_LOG(EXCEPTION) << "Invalid axis value: " << axis_attr;
  }
  int64_t k_value = x_input_shape[k_index];
  k_initializer_proto->add_int64_data(k_value);
  node_proto->add_input(k_input_name);

  node_proto->add_output(MakeOutputName(node_name, kZeroNum));
  auto indices_output_name = MakeOutputName(node_name, kOneNum);
  auto indices_cast_name = indices_output_name + "_cast";
  node_proto->add_output(indices_cast_name);
  AddCastOp(indices_cast_name, indices_output_name, onnx::TensorProto_DataType_INT32, graph_proto);

  onnx::AttributeProto *axis_attr_proto = node_proto->add_attribute();
  axis_attr_proto->set_name("axis");
  axis_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  axis_attr_proto->set_i(axis_attr);

  onnx::AttributeProto *largest_attr_proto = node_proto->add_attribute();
  largest_attr_proto->set_name("largest");
  largest_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  if (descending_attr) {
    largest_attr_proto->set_i(kOneNum);
  } else {
    largest_attr_proto->set_i(kZeroNum);
  }

  onnx::AttributeProto *sorted_attr_proto = node_proto->add_attribute();
  sorted_attr_proto->set_name("sorted");
  sorted_attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  sorted_attr_proto->set_i(1);
}

void OnnxExporter::ExportPrimCustom(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_name("Custom_" + node_name);
  mindspore::HashSet<size_t> input_attrs;

  constexpr auto kAttrInputNames = "input_names";
  constexpr auto kAttrAttrNames = "attr_names";
  constexpr auto kAttrOutputNames = "output_names";
  auto input_names_vec = GetOpAttribute<std::vector<std::string>>(node, kAttrInputNames);
  auto primitive = GetPrimitive(node);
  auto attr_names = primitive->GetAttr(kAttrAttrNames);
  if (attr_names != nullptr) {
    auto attr_names_vec = GetValue<std::vector<std::string>>(attr_names);
    if (input_names_vec.size() >= attr_names_vec.size()) {
      size_t offset = input_names_vec.size() - attr_names_vec.size();
      for (size_t i = offset; i < input_names_vec.size(); ++i) {
        if (input_names_vec[i] != attr_names_vec[i - offset]) {
          MS_LOG(EXCEPTION) << primitive->name() << " found mismatching attr name " << input_names_vec[i]
                            << "in input_names and " << attr_names_vec[i - offset] << " in attr_names";
        }
        (void)input_attrs.insert(i);
      }
    }
  }

  auto inputs = node->inputs();
  std::vector<AnfNodePtr> real_inputs;

  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_attrs.find(i) != input_attrs.end() && input_node->isa<ValueNode>() && !HasAbstractMonad(input_node)) {
      auto value_node = input_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto attr_value = value_node->value();
      if (attr_value->isa<StringImm>()) {
        auto str_attr = GetValue<std::string>(attr_value);
        onnx::AttributeProto *str_proto = node_proto->add_attribute();
        str_proto->set_name(input_names_vec[i]);
        str_proto->set_type(onnx::AttributeProto_AttributeType_STRING);
        str_proto->set_s(str_attr);
      } else if (attr_value->isa<IntegerImm>()) {
        int64_t int64_attr = attr_value->cast<Int64ImmPtr>()->value();
        onnx::AttributeProto *int64_proto = node_proto->add_attribute();
        int64_proto->set_name(input_names_vec[i]);
        int64_proto->set_type(onnx::AttributeProto_AttributeType_INT);
        int64_proto->set_i(int64_attr);
      } else if (attr_value->isa<FloatImm>()) {
        int64_t fp32_attr = attr_value->cast<FP32ImmPtr>()->value();
        onnx::AttributeProto *fp32_proto = node_proto->add_attribute();
        fp32_proto->set_name(input_names_vec[i]);
        fp32_proto->set_type(onnx::AttributeProto_AttributeType_FLOAT);
        fp32_proto->set_i(fp32_attr);
      } else {
        MS_LOG(EXCEPTION) << "Unsupported attr input type: " << attr_value->ToString();
      }
    } else {
      real_inputs.push_back(inputs[i + 1]);
    }
  }

  for (size_t idx = 0; idx < real_inputs.size(); idx++) {
    auto input_name = GetNodeInputName(real_inputs[idx], node_map_ptr, graph_proto);
    node_proto->add_input(input_name);
  }

  node_proto->add_output(node_name);
  auto output_names_vec = GetOpAttribute<std::vector<std::string>>(node, kAttrOutputNames);
  for (size_t idx = 0; idx < output_names_vec.size(); idx++) {
    auto indices_name = MakeOutputName(node_name, idx);
    node_proto->add_output(indices_name);
  }
  node_proto->set_op_type(GetOpAttribute<std::string>(node, "reg_op_name"));
}

void OnnxExporter::ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node,
                               std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  using ExportFunc = std::function<void(OnnxExporter *, const FuncGraphPtr &, const CNodePtr &,
                                        std::map<AnfNodePtr, std::string> *, onnx::GraphProto *const)>;
  static std::vector<std::pair<PrimitivePtr, ExportFunc>> export_table = {
    {prim::kPrimReshape, &OnnxExporter::ExportPrimReshape},
    {prim::kPrimReduceMean, &OnnxExporter::ExportPrimReduce},
    {prim::kPrimReduceSum, &OnnxExporter::ExportPrimReduce},
    {prim::kPrimReduceMax, &OnnxExporter::ExportPrimReduce},
    {prim::kPrimReduceAny, &OnnxExporter::ExportPrimReduceAnyOrAll},
    {prim::kPrimReduceAll, &OnnxExporter::ExportPrimReduceAnyOrAll},
    {prim::kPrimTranspose, &OnnxExporter::ExportPrimTranspose},
    {prim::kPrimStridedSlice, &OnnxExporter::ExportPrimStridedSlice},
    {prim::kPrimResizeNearestNeighbor, &OnnxExporter::ExportPrimResizeNearestNeighbor},
    {prim::kPrimResizeBilinear, &OnnxExporter::ExportPrimResizeBilinear},
    {prim::kPrimConcat, &OnnxExporter::ExportPrimConcat},
    {prim::kPrimCast, &OnnxExporter::ExportPrimCast},
    {prim::kPrimPRelu, &OnnxExporter::ExportPrimPReLU},
    {prim::kPrimRelu6, &OnnxExporter::ExportPrimReLU6},
    {prim::kPrimDepthwiseConv2dNative, &OnnxExporter::ExportPrimDepthwiseConv2d},
    {prim::kPrimTile, &OnnxExporter::ExportPrimTile},
    {prim::kPrimSquare, &OnnxExporter::ExportPrimSquare},
    {prim::kPrimGather, &OnnxExporter::ExportPrimGatherV2},
    {prim::kPrimTupleGetItem, &OnnxExporter::ExportPrimTupleGetItem},
    {prim::kPrimTopK, &OnnxExporter::ExportPrimTopK},
    {prim::kPrimBoundingBoxDecode, &OnnxExporter::ExportPrimBoundingBoxDecode},
    {prim::kPrimNMSWithMask, &OnnxExporter::ExportPrimNMSWithMask},
    {prim::kPrimSplit, &OnnxExporter::ExportPrimSplit},
    {prim::kPrimROIAlign, &OnnxExporter::ExportPrimROIAlign},
    {prim::kPrimSlice, &OnnxExporter::ExportPrimSlice},
    {prim::kPrimOnesLike, &OnnxExporter::ExportPrimOnesLike},
    {prim::kPrimScatterNd, &OnnxExporter::ExportPrimScatterNd},
    {prim::kPrimArgMaxWithValue, &OnnxExporter::ExportPrimArgMaxWithValue},
    {prim::kPrimArgMinWithValue, &OnnxExporter::ExportPrimArgMinWithValue},
    {prim::kPrimOneHot, &OnnxExporter::ExportPrimOneHot},
    {prim::kPrimConv2DTranspose, &OnnxExporter::ExportPrimConv2DTranspose},
    {prim::kPrimGreaterEqual, &OnnxExporter::ExportPrimGreaterEqual},
    {prim::kPrimLessEqual, &OnnxExporter::ExportPrimLessEqual},
    {prim::kPrimSqueeze, &OnnxExporter::ExportPrimSqueeze},
    {prim::kPrimExpandDims, &OnnxExporter::ExportPrimExpandDims},
    {prim::kPrimGatherD, &OnnxExporter::ExportPrimGatherD},
    {prim::kPrimPad, &OnnxExporter::ExportPrimPad},
    {prim::kPrimBatchMatMul, &OnnxExporter::ExportPrimBatchMatMul},
    {prim::kPrimBroadcastTo, &OnnxExporter::ExportPrimBroadcastTo},
    {prim::kPrimAddN, &OnnxExporter::ExportPrimAddN},
    {prim::kPrimGeLU, &OnnxExporter::ExportPrimGeLU},
    {prim::kPrimLstm, &OnnxExporter::ExportPrimLSTM},
    {prim::kPrimReverseV2, &OnnxExporter::ExportPrimReverseV2},
    {prim::kPrimTensorCopySlices, &OnnxExporter::ExportPrimTensorCopySlices},
    {prim::kPrimDynamicRNN, &OnnxExporter::ExportPrimDynamicRNN},
    {prim::kPrimStack, &OnnxExporter::ExportPrimStack},
    {prim::kPrimAtan2, &OnnxExporter::ExportPrimAtan2},
    {prim::kPrimFloorDiv, &OnnxExporter::ExportPrimFloorDiv},
    {prim::kPrimFloorMod, &OnnxExporter::ExportPrimFloorMod},
    {prim::kPrimSort, &OnnxExporter::ExportPrimSort},
    {prim::kPrimCustom, &OnnxExporter::ExportPrimCustom},
  };

  auto iter = std::find_if(export_table.begin(), export_table.end(),
                           [&node](const auto &item) { return node->IsApply(item.first); });
  if (iter != export_table.end()) {
    iter->second(this, func_graph, node, node_map_ptr, graph_proto);
    return;
  }

  auto inputs = node->inputs();
  if (inputs.size() < 1) {
    MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
  }

  AnfNodePtr op = inputs[kZeroNum];
  std::vector<AnfNodePtr> op_inputs;
  // first process node input 1,2,..., since when node input is a ValueNode, here need to create a Constant Operator
  for (size_t i = 1; i < inputs.size(); i++) {
    if (!HasAbstractMonad(inputs[i])) {
      op_inputs.push_back(inputs[i]);
    }
  }

  if (!op->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "Need to support node op type " << op->type_name();
  }

  auto op_value = dyn_cast<ValueNode>(op)->value();
  if (op_value->isa<Primitive>()) {
    auto prim = dyn_cast<Primitive>(op_value);
    (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim, op_inputs, graph_proto);
  } else if (while_loop_export::IsControlSubgraph(op_value)) {
    ExportWhileLoop(node, node_map_ptr, graph_proto);
  } else {
    MS_LOG(EXCEPTION) << "Need to support node op value type " << op_value->type_name();
  }
}

void OnnxExporter::ExportWhileLoop(const CNodePtr &start_node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *graph_proto) {
  auto node_name = RegisterNodeWithUniqueName(start_node, node_map_ptr);
  auto loop_parts = while_loop_export::MatchGraph(start_node);

  // 1. Make Loop op

  onnx::NodeProto *loop_proto = graph_proto->add_node();
  loop_proto->set_op_type("Loop");

  auto loop_count_name = node_name + "_M";
  const auto &loop_counter_params = loop_parts.loop_condition_info;
  int64_t loop_count = (loop_counter_params.end - loop_counter_params.begin) / loop_counter_params.step;
  onnx::TensorProto *loop_count_proto = graph_proto->add_initializer();
  loop_count_proto->set_name(loop_count_name);
  loop_count_proto->set_data_type(onnx::TensorProto_DataType_INT64);
  loop_count_proto->add_int64_data(loop_count);

  auto loop_cond_name = node_name + "_cond";
  auto *cond_value = graph_proto->add_initializer();
  cond_value->set_name(loop_cond_name);
  cond_value->set_data_type(onnx::TensorProto_DataType_BOOL);
  cond_value->add_int32_data(true);

  loop_proto->add_input(loop_count_name);
  loop_proto->add_input(loop_cond_name);
  for (const auto &[loop_i, control_i] : loop_parts.used_loop_to_control_param_indices) {
    auto name = GetNodeInputName(start_node->input(control_i + 1), node_map_ptr, graph_proto);
    loop_proto->add_input(name);
    loop_proto->add_output(MakeOutputName(node_name + "_loop", loop_i));
  }

  onnx::AttributeProto *subgraph_attr = loop_proto->add_attribute();
  subgraph_attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
  subgraph_attr->set_name("body");
  onnx::GraphProto *loop_subgraph_proto = subgraph_attr->mutable_g();

  // 2. Create subgraph for loop body

  auto subgraph_name = loop_parts.loop_subgraph->ToString();
  auto subgraph_input_cond_name = subgraph_name + "_input_cond";

  auto *iter_num_input = loop_subgraph_proto->add_input();
  iter_num_input->set_name(subgraph_name + "_input_M");
  (void)iter_num_input->mutable_type()->mutable_tensor_type()->mutable_shape();  // side-effect: shape created
  iter_num_input->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT64);

  auto *cond_input = loop_subgraph_proto->add_input();
  cond_input->set_name(subgraph_input_cond_name);
  cond_input->mutable_type()->mutable_tensor_type()->set_elem_type(cond_value->data_type());

  auto *cond_output = loop_subgraph_proto->add_output();
  cond_output->set_name(cond_input->name());
  cond_output->mutable_type()->mutable_tensor_type()->set_elem_type(cond_value->data_type());

  MS_EXCEPTION_IF_CHECK_FAIL(renamed_node_map_.empty(), "renamed_nodes must be cleared after subgraph export");
  for (size_t i : loop_parts.ignored_loop_param_indices) {
    const auto &param = loop_parts.loop_subgraph->parameters().at(i);
    renamed_node_map_[param] = "";
  }

  // Export everything except the control call and the output (see MatchAndMark)
  ExportFuncGraph(loop_parts.loop_subgraph, node_map_ptr, loop_subgraph_proto);

  // Export outputs manually
  for (const auto &loop_to_control_i : loop_parts.used_loop_to_control_param_indices) {
    const auto &input = loop_parts.repeat_node->input(loop_to_control_i.second + 1);
    ExportOutput(loop_parts.loop_subgraph, input, node_map_ptr, loop_subgraph_proto);
  }
  renamed_node_map_.clear();

  // 3. Export part after loop

  MS_EXCEPTION_IF_CHECK_FAIL(renamed_node_map_.empty(), "renamed_nodes must be cleared after subgraph export");
  const auto &after_loop_params = loop_parts.after_loop_subgraph->parameters();
  for (const auto &[after_i, output_i] : loop_parts.after_param_to_output_indices) {
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<int>(output_i) < loop_proto->output_size(), "Output index out of bounds");
    renamed_node_map_[after_loop_params.at(after_i)] = loop_proto->output(output_i);
  }
  ExportFuncGraph(loop_parts.after_loop_subgraph, node_map_ptr, graph_proto, false);

  auto after_loop_retval = GetRealInput(loop_parts.after_loop_subgraph->get_return()->input(1));
  if (after_loop_retval->isa<CNode>() && after_loop_retval->cast<CNodePtr>()->IsApply(prim::kPrimMakeTuple)) {
    auto tuple_retval = dyn_cast<CNode>(after_loop_retval);
    for (size_t i = 1; i < tuple_retval->inputs().size(); ++i) {
      auto output_name = GetNodeInputName(tuple_retval->input(i), node_map_ptr, graph_proto);
      AddOp("Identity", {output_name}, {MakeOutputName(node_name, i - 1)}, graph_proto);
    }
  } else {
    auto output_name = GetNodeInputName(after_loop_retval, node_map_ptr, graph_proto);
    AddOp("Identity", {output_name}, {node_name}, graph_proto);
  }
  renamed_node_map_.clear();
}

onnx::TensorProto_DataType OnnxExporter::GetOutputType(const AnfNodePtr &node, int64_t output_index) {
  auto unpacked = GetRealInput(node);
  if (IsPrimitiveCNode(unpacked, prim::kPrimTupleGetItem)) {
    if (output_index != -1) {
      MS_LOG(EXCEPTION) << "Unexpected output index for TupleGetItem: " << output_index;
    }
    auto cnode = dyn_cast<CNode>(unpacked);
    unpacked = cnode->input(kOneNum);
    output_index = GetInt64Value(cnode->input(kTwoNum));
  }

  /*
    Special cases (MS and ONNX type differences) go here
    Example:
      if (IsPrimitiveCNode(unpacked, prim::kPrim<Something>) && output_index == <i>) {
        return onnx::TensorProto_DataType_<TYPE>;
      }
  */

  if (output_index == -1) {
    auto tensor = dyn_cast<TensorType>(unpacked->Type());
    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Expected output of node " << unpacked->ToString()
                        << " to be a single tensor. Instead got: " << unpacked->Type()->ToString();
    }
    return GetOnnxDataType(tensor->element()->type_id());
  } else {
    auto tuple_type = dyn_cast<Tuple>(unpacked->Type());
    if (tuple_type == nullptr) {
      MS_LOG(EXCEPTION) << "Expected output of node " << unpacked->ToString()
                        << " to be a tuple. Instead got: " << unpacked->Type()->ToString();
    }
    auto element_type = tuple_type->elements()[static_cast<size_t>(output_index)];
    MS_EXCEPTION_IF_NULL(element_type);
    auto tensor_type = dyn_cast<TensorType>(element_type);
    if (tensor_type == nullptr) {
      MS_LOG(EXCEPTION) << "Expected output " << output_index << " of node " << unpacked->ToString()
                        << " to be a tensor. Instead got: " << element_type->ToString();
    }
    return GetOnnxDataType(tensor_type->element()->type_id());
  }
}

void OnnxExporter::AddOutputWithCast(onnx::NodeProto *node_proto, const std::string &output_name,
                                     onnx::TensorProto_DataType target_type, onnx::GraphProto *graph_proto) const {
  if (target_type == onnx::TensorProto_DataType_UNDEFINED) {
    node_proto->add_output(output_name);
  } else {
    auto output_to_cast_name = output_name + "_output_to_cast";
    node_proto->add_output(output_to_cast_name);
    AddCastOp(output_to_cast_name, output_name, target_type, graph_proto);
  }
}

std::string OnnxExporter::ExportPrimitive(const FuncGraphPtr &, std::map<AnfNodePtr, std::string> *node_map_ptr,
                                          const PrimitivePtr &prim, const std::vector<AnfNodePtr> &inputs,
                                          onnx::GraphProto *const graph_proto) {
  auto op_map = OpConvertRegistry::GetOpConvertMap();
  MS_EXCEPTION_IF_NULL(prim);
  auto op_iter = op_map.find(prim->name());
  if (op_iter == op_map.end()) {
    MS_LOG(EXCEPTION) << "Can not find key " << prim->name() << " in convert map. "
                      << "Exporting " << prim->name() << " operator is not yet supported.";
  }
  // Get input first, because input maybe valuenode which need create constant node
  std::vector<std::string> input_list;
  for (const auto &input : inputs) {
    auto input_name = GetNodeInputName(input, node_map_ptr, graph_proto);
    input_list.push_back(input_name);
  }

  const OpNameInfo &op_convert_info = op_iter->second;
  auto node_name = GenerateUniqueName();

  std::vector<onnx::TensorProto_DataType> output_cast_types(op_convert_info.num_outputs(),
                                                            onnx::TensorProto_DataType_UNDEFINED);
  // Cast inputs if needed
  for (const auto &rule : op_convert_info.input_casts()) {
    auto original_type = GetOutputType(inputs[static_cast<size_t>(rule.input_index)]);
    if (original_type != rule.input_type) {
      continue;
    }

    auto cast_input_name = node_name + "cast_input_" + std::to_string(rule.input_index);
    AddCastOp(input_list[static_cast<size_t>(rule.input_index)], cast_input_name, rule.target_type, graph_proto);
    input_list[static_cast<size_t>(rule.input_index)] = cast_input_name;

    auto output_cast = std::find_if(
      op_convert_info.output_casts().begin(), op_convert_info.output_casts().end(), [&rule](const OutputConversion &x) {
        return x.mode == OutputConversion::Mode::INPUT && x.input_with_matching_type == rule.input_index;
      });
    if (output_cast != op_convert_info.output_casts().end()) {
      output_cast_types[static_cast<size_t>(output_cast->output_index)] = original_type;
    }
  }

  for (const auto &output_cast : op_convert_info.output_casts()) {
    if (output_cast.mode == OutputConversion::Mode::FIXED) {
      output_cast_types[static_cast<size_t>(output_cast.output_index)] = output_cast.target_type;
    }
  }

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_name(node_name + op_convert_info.onnx_type());
  node_proto->set_op_type(op_convert_info.onnx_type());

  // Set outputs
  if (op_convert_info.num_outputs() == 1) {
    AddOutputWithCast(node_proto, node_name, output_cast_types[0], graph_proto);
  } else {
    for (int i = 0; i < op_convert_info.num_outputs(); ++i) {
      auto output_name = MakeOutputName(node_name, i);
      AddOutputWithCast(node_proto, output_name, output_cast_types[static_cast<size_t>(i)], graph_proto);
    }
  }

  // Set inputs
  for (const auto &input_name : input_list) {
    node_proto->add_input(input_name);
  }

  // Set node attribute
  for (const OpAttrInfo &attr : op_convert_info.op_attrs()) {
    const std::string &attr_name = attr.attr_name();
    ValuePtr attr_value = nullptr;
    if (!attr_name.empty()) {
      attr_value = prim->GetAttr(attr_name);
      if (attr_value == nullptr) {
        MS_LOG(EXCEPTION) << "Primitive " << prim->name() << " does not have attribute " << attr_name;
      }
    }
    onnx::AttributeProto *onnx_attr_proto = node_proto->add_attribute();
    onnx_attr_proto->set_name(attr.onnx_attr_name());
    attr.fn_gen_attr()(attr_value, attr.onnx_attr_type(), onnx_attr_proto, prim);
  }
  return node_name;
}

void OnnxExporter::ExportMergeConv(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *const graph_proto) {
  auto conv_node = dyn_cast<CNode>(node->input(kOneNum));
  auto input_x = conv_node->input(kOneNum);  // conv input x
  auto input_w = conv_node->input(kTwoNum);  // conv weight(filter)
  auto input_b = node->input(kTwoNum);       // conv bias

  PrimitivePtr prim_conv = dyn_cast<Primitive>((dyn_cast<ValueNode>(conv_node->input(kZeroNum)))->value());
  std::vector<AnfNodePtr> inputs{input_x, input_w, input_b};
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_conv, inputs, graph_proto);
}

void OnnxExporter::ExportMergeGemm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                   std::map<AnfNodePtr, std::string> *node_map_ptr,
                                   onnx::GraphProto *const graph_proto) {
  auto matmul_node = dyn_cast<CNode>(node->input(kOneNum));
  auto input_x = matmul_node->input(kOneNum);  // matmul input x
  auto input_y = matmul_node->input(kTwoNum);  // matmul input y
  auto input_b = node->input(kTwoNum);         // matmul bias

  PrimitivePtr prim_matmul = dyn_cast<Primitive>((dyn_cast<ValueNode>(matmul_node->input(kZeroNum)))->value());
  std::vector<AnfNodePtr> inputs{input_x, input_y, input_b};
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_matmul, inputs, graph_proto);
}

void OnnxExporter::ExportMergeBatchNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                        std::map<AnfNodePtr, std::string> *node_map_ptr,
                                        onnx::GraphProto *const graph_proto) {
  auto batch_norm_node = dyn_cast<CNode>(node->input(kOneNum));

  auto is_training = GetOpAttribute<bool>(batch_norm_node, "is_training");
  if (is_training) {
    auto input_x_name = GetNodeInputName(batch_norm_node->input(kOneNum), node_map_ptr, graph_proto);
    auto scale_input_name = GetNodeInputName(batch_norm_node->input(kTwoNum), node_map_ptr, graph_proto);
    auto bias_input_name = GetNodeInputName(batch_norm_node->input(kThreeNum), node_map_ptr, graph_proto);

    auto onnx_type = GetOutputType(batch_norm_node->input(kOneNum));

    auto output_name = RegisterNodeWithUniqueName(node, node_map_ptr);

    auto input_shape_ptr = batch_norm_node->input(kOneNum)->Shape();
    auto input_shape = input_shape_ptr->cast<abstract::ShapePtr>()->shape();

    std::vector<int64_t> normalize_axes = {0};
    for (size_t i = kTwoNum; i < input_shape.size(); ++i) {
      normalize_axes.push_back(static_cast<int64_t>(i));
    }

    std::vector<int64_t> scale_bias_shape(input_shape.size(), 1);
    scale_bias_shape[1] = -1;
    auto reshaped_scale_name = output_name + "_reshaped_scale";
    AddReshapeOp(scale_input_name, reshaped_scale_name, scale_bias_shape, graph_proto);
    auto reshaped_bias_name = output_name + "_reshaped_bias";
    AddReshapeOp(bias_input_name, reshaped_bias_name, scale_bias_shape, graph_proto);
    auto epsilon = GetOpAttribute<float>(batch_norm_node, "epsilon");

    AddMeanVarianceNormalizationOp(input_x_name, reshaped_scale_name, reshaped_bias_name, output_name, normalize_axes,
                                   epsilon, input_shape, onnx_type, graph_proto);
  } else {
    PrimitivePtr prim_batch_norm = GetPrimitive(batch_norm_node);
    std::vector<AnfNodePtr> inputs;
    for (size_t i = 1; i < batch_norm_node->inputs().size(); i++) {
      inputs.push_back(batch_norm_node->input(i));
    }
    (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_batch_norm, inputs, graph_proto);
  }
}

void OnnxExporter::ExportMergeMaxPoolWithArgmax(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                                std::map<AnfNodePtr, std::string> *node_map_ptr,
                                                onnx::GraphProto *const graph_proto) {
  auto maxpool_with_argmax_node = dyn_cast<CNode>(node->input(kOneNum));

  PrimitivePtr prim_maxpool_with_argmax =
    dyn_cast<Primitive>((dyn_cast<ValueNode>(maxpool_with_argmax_node->input(kZeroNum)))->value());
  std::vector<AnfNodePtr> inputs;
  for (size_t i = 1; i < maxpool_with_argmax_node->inputs().size(); i++) {
    inputs.push_back(maxpool_with_argmax_node->input(i));
  }
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_maxpool_with_argmax, inputs, graph_proto);
}

// LayerNorm(N, C1, H, W) --> reshape(1, C2, 1, W) + MeanVarianceNormalization + reshape(N, C1, H, W)
void OnnxExporter::ExportMergeLayerNorm(const FuncGraphPtr &, const CNodePtr &node,
                                        std::map<AnfNodePtr, std::string> *node_map_ptr,
                                        onnx::GraphProto *const graph_proto) {
  auto LayerNormNode = dyn_cast<CNode>(node->input(kOneNum));
  auto layernorm_input_x = GetNodeInputName(LayerNormNode->input(kOneNum), node_map_ptr, graph_proto);
  auto layernorm_input_gamma = GetNodeInputName(LayerNormNode->input(kTwoNum), node_map_ptr, graph_proto);
  auto layernorm_input_beta = GetNodeInputName(LayerNormNode->input(kThreeNum), node_map_ptr, graph_proto);

  auto begin_norm_axis = GetOpAttribute<int64_t>(LayerNormNode, "begin_norm_axis");
  auto begin_params_axis = GetOpAttribute<int64_t>(LayerNormNode, "begin_params_axis");
  if (begin_norm_axis != -1 || begin_params_axis != -1) {
    MS_LOG(EXCEPTION) << "begin_norm_axis != -1 and begin_params_axis != -1 are not implemented";
  }

  auto onnx_type = GetOutputType(LayerNormNode->input(kOneNum));
  auto input_shape = dyn_cast<abstract::Shape>(LayerNormNode->input(kOneNum)->Shape())->shape();
  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto epsilon = GetOpAttribute<float>(LayerNormNode, "epsilon");
  std::vector<int64_t> reduce_axes = {static_cast<int64_t>(input_shape.size()) - 1};

  AddMeanVarianceNormalizationOp(layernorm_input_x, layernorm_input_gamma, layernorm_input_beta, node_name, reduce_axes,
                                 epsilon, input_shape, onnx_type, graph_proto);
}

void OnnxExporter::ExportMergeConv2DTranspose(const FuncGraphPtr &, const CNodePtr &node,
                                              std::map<AnfNodePtr, std::string> *node_map_ptr,
                                              onnx::GraphProto *const graph_proto) {
  auto conv_node = dyn_cast<CNode>(node->input(kOneNum));
  PrimConv2DTransposeExportHelper(conv_node, node, node_map_ptr, graph_proto);
}

void AddTransposeOp(const std::string &input, const std::string &output, onnx::GraphProto *graph_proto) {
  onnx::NodeProto *node_proto = graph_proto->add_node();
  std::string op_type = "Transpose";
  node_proto->set_op_type(op_type);
  node_proto->set_name(output + op_type);
  node_proto->add_input(input);
  node_proto->add_output(output);
}

void AddUnsqueezeOp(const std::string &input, const std::string &output, int64_t axis, onnx::GraphProto *graph_proto) {
  onnx::NodeProto *node_proto = graph_proto->add_node();
  std::string op_type = "Unsqueeze";
  node_proto->set_op_type(op_type);
  node_proto->set_name(output + op_type);
  node_proto->add_input(input);
  node_proto->add_output(output);

  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto->set_name("axes");
  attr_proto->add_ints(axis);
}

void AddSqueezeOp(const std::string &input, const std::string &output, int64_t axis, onnx::GraphProto *graph_proto) {
  onnx::NodeProto *node_proto = graph_proto->add_node();
  std::string op_type = "Squeeze";
  node_proto->set_op_type(op_type);
  node_proto->set_name(output + op_type);
  node_proto->add_input(input);
  node_proto->add_output(output);

  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto->set_name("axes");
  attr_proto->add_ints(axis);
}

void AddGRUOp(const std::vector<std::string> &inputs, const std::vector<std::string> &outputs, int64_t hidden_size,
              int64_t linear_before_reset, onnx::GraphProto *graph_proto) {
  onnx::NodeProto *node_proto = graph_proto->add_node();
  std::string op_type = "GRU";
  node_proto->set_op_type(op_type);
  node_proto->set_name(outputs[0] + op_type);

  for (const auto &in : inputs) {
    node_proto->add_input(in);
  }

  for (const auto &out : outputs) {
    node_proto->add_output(out);
  }

  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto->set_name("linear_before_reset");
  attr_proto->set_i(linear_before_reset);

  onnx::AttributeProto *attr2_proto = node_proto->add_attribute();
  attr2_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  attr2_proto->set_name("hidden_size");
  attr2_proto->set_i(hidden_size);
}

void UnsqueezeInputOfGRU(std::string *in_name, const std::string &node_name, const std::string &suffix, int64_t axis,
                         onnx::GraphProto *graph_proto) {
  auto out_name = node_name + suffix;
  AddUnsqueezeOp(*in_name, out_name, 0, graph_proto);
  *in_name = out_name;
}

void GruRzh2Zrh(std::string *in_name, const std::string &node_name, const std::string &mid_name,
                std::vector<std::string> tmp_out_names, const std::vector<int64_t> &hidden_sizes, int64_t axis,
                onnx::GraphProto *graph_proto) {
  const int kConcatNum = 6;
  const int kIndexBiasHiddenR = 3;
  const int kIndexBiasHiddenZ = 4;
  auto out_name = node_name + mid_name + "_zrh";

  AddSplitOp(*in_name, tmp_out_names, hidden_sizes, 0, graph_proto);
  swap(tmp_out_names[0], tmp_out_names[1]);
  if (tmp_out_names.size() == kConcatNum) {
    swap(tmp_out_names[kIndexBiasHiddenR], tmp_out_names[kIndexBiasHiddenZ]);
  }
  AddConcatOp(tmp_out_names, out_name, 0, graph_proto);
  *in_name = out_name;
}

/*
  Mapping between the inputs of MindSpore DynamicGRUV2 and ONNX GRU operator.
  +----------------------------------------------------------+----------------------------------------------+
  |                          ONNX                            |                  MindSpore                   |
  +==========================================================+==============================================+
  | X: [seq_length, batch_size, input_size]                  | x: (num_step, batch_size, input_size)        |
  +----------------------------------------------------------+----------------------------------------------+
  | W: [num_directions, 3*hidden_size, input_size]           | weight_input: (input_size, 3*hidden_size)    |
  +----------------------------------------------------------+----------------------------------------------+
  | R: [num_directions, 3*hidden_size, hidden_size]          | weight_hidden: (hidden_size, 3*hidden_size)  |
  +----------------------------------------------------------+----------------------------------------------+
  |                                                          | bias_input:  (3*hidden_size)                 |
  + B: [num_directiBBons, 6*hidden_size]                     +----------------------------------------------+
  |                                                          | bias_hidden: (3*hidden_size)                 |
  +----------------------------------------------------------+----------------------------------------------+
  | sequence_lens: [batch_size]                              | seq_length: (hidden_size)                    |
  +----------------------------------------------------------+----------------------------------------------+
  | initial_h: [num_directions, batch_size, hidden_size]     | init_h: (batch_size, hidden_size)            |
  +----------------------------------------------------------+----------------------------------------------+
  | Y:[seq_length, num_directions, batch_size, hidden_size]  | y: (num_step, batch_size, hidden_size)       |
  +----------------------------------------------------------+----------------------------------------------+
*/
void OnnxExporter::ExportMergeDynamicGRUV2(const FuncGraphPtr &, const CNodePtr &node,
                                           std::map<AnfNodePtr, std::string> *node_map_ptr,
                                           onnx::GraphProto *const graph_proto) {
  const int kInX = 1;
  const int kInWeightInput = 2;
  const int kInWeightHidden = 3;
  const int kInBiasInput = 4;
  const int kInBiasHidden = 5;
  // The 6th input 'seq_length' now only support None, so it's not used.
  const int kInInitH = 7;

  const int kWeightHiddenDim = 2;
  const int kNumberOfGates = 3;
  const std::string kDefaultDir = "UNIDIRECTIONAL";
  const std::string kDefaultAct = "tanh";
  const std::vector<std::string> kGateOrderSupported{"rzh", "zrh"};

  auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
  auto gru_node = dyn_cast<CNode>(node->input(1));

  /* Get Attributes */
  auto direction = GetOpAttribute<std::string>(gru_node, "direction");
  auto activation = GetOpAttribute<std::string>(gru_node, "activation");
  auto gate_order = GetOpAttribute<std::string>(gru_node, "gate_order");
  auto reset_after = GetOpAttribute<bool>(gru_node, "reset_after");

  int64_t linear_before_reset = reset_after ? 1 : 0;

  if (direction != kDefaultDir) {
    MS_LOG(EXCEPTION) << "'direction': " << direction << " is not in supported values[" << kDefaultDir << "]";
  }
  if (activation != kDefaultAct) {
    MS_LOG(EXCEPTION) << "'activation': " << activation << " is not in supported values[" << kDefaultAct << "]";
  }
  if (gate_order != kGateOrderSupported[0] && gate_order != kGateOrderSupported[1]) {
    std::string supported;
    for (const auto &order : gate_order) {
      supported += order;
      supported += ", ";
    }
    MS_LOG(EXCEPTION) << "'gate_order': " << gate_order << " is not in supported values[" << supported << "]";
  }

  auto x = GetNodeInputName(gru_node->input(kInX), node_map_ptr, graph_proto);
  auto weight_input = GetNodeInputName(gru_node->input(kInWeightInput), node_map_ptr, graph_proto);
  auto weight_hidden = GetNodeInputName(gru_node->input(kInWeightHidden), node_map_ptr, graph_proto);
  auto bias_input = GetNodeInputName(gru_node->input(kInBiasInput), node_map_ptr, graph_proto);
  auto bias_hidden = GetNodeInputName(gru_node->input(kInBiasHidden), node_map_ptr, graph_proto);
  auto init_h = GetNodeInputName(gru_node->input(kInInitH), node_map_ptr, graph_proto);

  auto weight_hidden_shape = dyn_cast<abstract::Shape>(gru_node->input(kInWeightHidden)->Shape())->shape();
  if (weight_hidden_shape.size() != kWeightHiddenDim) {
    MS_LOG(EXCEPTION) << "The dim of input weight_hidden must be " << kWeightHiddenDim << ".";
  }
  int64_t hidden_size = weight_hidden_shape[1] / kNumberOfGates;

  auto trans_w_i = node_name + "_trans_w_i";
  AddTransposeOp(weight_input, trans_w_i, graph_proto);
  weight_input = trans_w_i;

  auto trans_w_h = node_name + "_trans_w_h";
  AddTransposeOp(weight_hidden, trans_w_h, graph_proto);
  weight_hidden = trans_w_h;

  auto bias_i_h = node_name + "_bias_i_h";
  AddConcatOp({bias_input, bias_hidden}, bias_i_h, 0, graph_proto);

  // ONNX GRU only support "zrh"
  if (gate_order == "rzh") {
    MS_LOG(INFO) << "change gate order 'rzh' to 'zrh'.";
    std::vector<int64_t> hidden_sizes(kNumberOfGates, hidden_size);
    GruRzh2Zrh(&weight_input, node_name, "w_i", {node_name + "_w_i_r", node_name + "_w_i_z", node_name + "_w_i_h"},
               hidden_sizes, 0, graph_proto);
    GruRzh2Zrh(&weight_hidden, node_name, "w_h", {node_name + "_w_h_r", node_name + "_w_h_z", node_name + "_w_h_h"},
               hidden_sizes, 0, graph_proto);

    std::vector<int64_t> bias_hidden_sizes(kNumberOfGates + kNumberOfGates, hidden_size);
    GruRzh2Zrh(&bias_i_h, node_name, "bias",
               {node_name + "_b_i_r", node_name + "_b_i_z", node_name + "_b_i_h", node_name + "_b_h_r",
                node_name + "_b_h_z", node_name + "_b_h_h"},
               bias_hidden_sizes, 0, graph_proto);
  }

  std::vector<std::string *> input_names = {&weight_input, &weight_hidden, &bias_i_h, &init_h};
  std::vector<std::string> suffixes = {"_unsqueeze_w_i", "_unsqueeze_w_h", "_unsqueeze_bias", "_unsqueeze_init_h"};
  for (size_t i = 0; i < input_names.size(); i++) {
    UnsqueezeInputOfGRU(input_names[i], node_name, suffixes[i], 0, graph_proto);
  }

  auto y = node_name + "_Y";
  // 'seq_length' input of DynamicGRUV2 is None, so pass "" to ONNX GRU.
  std::string sequence_lens = "";
  AddGRUOp({x, weight_input, weight_hidden, bias_i_h, sequence_lens, init_h}, {y}, hidden_size, linear_before_reset,
           graph_proto);

  AddSqueezeOp(y, node_name, 1, graph_proto);
}

/*
  Kinds of return values:
  1) A single Tensor
  2) A Tuple returned by an op with multiple outputs like TopK
  3) A Tuple returned by MakeTuple. This corresponds to `return x, y`
     or equivalent in Python, where x and y are Tensors
     In this case MakeTuple itself is not exported, so this case must be handled
     separately from the previous one
  4) A constant tuple (ValueNode). Example:
        class MyCell(nn.Cell):
            def __init__(self):
                super().__init__()
                self.x = ms.Tensor(np.zeros((1, 2, 3)))

            def construct(self):
                return self.x, self.x

 */
void OnnxExporter::ExportOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &return_arg,
                                std::map<AnfNodePtr, std::string> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  AnfNodePtr arg = GetRealInput(return_arg);
  if (IsPrimitiveCNode(arg, prim::kPrimMakeTuple)) {
    auto arg_cnode = dyn_cast<CNode>(arg);
    for (size_t i = 1; i < arg_cnode->inputs().size(); ++i) {
      const auto &output = arg_cnode->input(i);
      ExportOutput(func_graph, output, node_map_ptr, graph_proto);
    }
  } else if (arg->isa<ValueNode>() && arg->cast<ValueNodePtr>()->value()->isa<ValueTuple>()) {
    // Several outputs, all constants
    auto tuple = arg->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>();
    for (size_t i = 0; i < tuple->value().size(); ++i) {
      const auto &element = tuple->value().at(i);
      std::string output_name = GenerateUniqueName();

      onnx::TensorProto *initializer = graph_proto->add_initializer();
      initializer->set_name(output_name);
      SetTensorData(element, initializer);

      onnx::ValueInfoProto *output_proto = graph_proto->add_output();
      output_proto->set_name(output_name);
      SetValueInfoType(arg, output_proto, static_cast<int64_t>(i));
    }
  } else if (arg->Type()->isa<Tuple>()) {
    auto arg_name = GetNodeInputName(arg, node_map_ptr, graph_proto);
    auto tuple = dyn_cast<Tuple>(arg->Type());

    for (size_t i = 0; i < tuple->size(); ++i) {
      auto output_name = MakeOutputName(arg_name, i);
      onnx::ValueInfoProto *output_proto = graph_proto->add_output();
      output_proto->set_name(output_name);
      SetValueInfoType(arg, output_proto, static_cast<int64_t>(i));
    }
  } else if (arg->Type()->isa<TensorType>()) {
    auto arg_name = GetNodeInputName(arg, node_map_ptr, graph_proto);
    onnx::ValueInfoProto *output_proto = graph_proto->add_output();
    output_proto->set_name(arg_name);
    SetValueInfoType(arg, output_proto);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported network output type " << arg->Type()->ToString() << " in node "
                      << arg->ToString();
  }
}

std::string OnnxExporter::GetNodeInputName(const AnfNodePtr &orig_node, std::map<AnfNodePtr, std::string> *node_map_ptr,
                                           onnx::GraphProto *const) {
  auto node = GetRealInput(orig_node);

  // if node is renamed and not ignored, use alternative name
  // if it is ignored, try to find the actual name in global map
  auto renamed_iter = renamed_node_map_.find(node);
  if (renamed_iter != renamed_node_map_.end() && renamed_iter->second != "") {
    return renamed_iter->second;
  }

  auto iter = node_map_ptr->find(node);
  if (iter != node_map_ptr->end()) {
    return iter->second;
  }

  if (node->isa<CNode>() || (node->isa<Parameter>() && !node->cast<ParameterPtr>()->has_default())) {
    MS_LOG(EXCEPTION) << "Can not find node '" << node->DebugString() << "' in node_map";
  }

  // for ValueNode or Parameter with default input, create an initializer
  // same value can be used in several subgraphs, so create initializers in root graph
  if (node->isa<ValueNode>()) {
    auto node_name = RegisterNodeWithUniqueName(node, node_map_ptr);
    auto value = node->cast<ValueNodePtr>()->value();

    onnx::TensorProto *initializer_proto = model_.mutable_graph()->add_initializer();
    initializer_proto->set_name(node_name);
    SetTensorData(value, initializer_proto);

    (*node_map_ptr)[node] = node_name;
    return node_name;
  }

  if (node->isa<Parameter>()) {
    auto param = dyn_cast<Parameter>(node);
    auto node_name = GenerateUniqueParameterName(param, node_map_ptr);

    onnx::TensorProto *initializer_proto = model_.mutable_graph()->add_initializer();
    initializer_proto->set_name(node_name);
    SetTensorData(param->default_param(), initializer_proto);

    (*node_map_ptr)[node] = node_name;
    return node_name;
  }

  MS_LOG(EXCEPTION) << "Unexpected node type " << node->type_name();
}

void OnnxExporter::ConvertTupleToTensor(const ValuePtr &value, onnx::TensorProto *const tensor_proto) const {
  auto tuple_ptr = dyn_cast<ValueTuple>(value);
  MS_EXCEPTION_IF_NULL(tuple_ptr);
  if (tuple_ptr->size() == 0) {
    MS_LOG(EXCEPTION) << "Convert tuple to tensor fail, the size of converted tuple is 0.";
  }

  ValuePtr first_element = (*tuple_ptr)[0];
  if (!first_element->isa<Scalar>()) {  // For non-scalars x->type() contains nullptr
    MS_LOG(EXCEPTION) << "Expected tuple elements to be scalars. Got: " << value->ToString();
  }
  auto type_id = first_element->type()->type_id();
  for (size_t i = 1; i < tuple_ptr->size(); ++i) {
    const auto element_type = (*tuple_ptr)[i]->type();
    if (element_type == nullptr || element_type->type_id() != type_id) {
      MS_LOG(EXCEPTION) << "Convert tuple to tensor fail, type of tuple elements is not same.";
    }
  }

  onnx::TensorProto_DataType result_type = onnx::TensorProto_DataType_UNDEFINED;
  if (first_element->isa<IntegerImm>()) {
    result_type = onnx::TensorProto_DataType_INT64;
  } else if (first_element->isa<FloatImm>()) {
    result_type = onnx::TensorProto_DataType_FLOAT;
  } else {
    MS_LOG(EXCEPTION) << "Convert tuple to tensor fail, unexpected tuple element type "
                      << first_element->type()->type_name() << ".";
  }

  tensor_proto->add_dims(static_cast<::google::protobuf::int64>(tuple_ptr->size()));
  tensor_proto->set_data_type(result_type);
  for (size_t i = 0; i < tuple_ptr->size(); ++i) {
    ValuePtr elem = (*tuple_ptr)[i];
    if (elem->isa<Int8Imm>()) {
      tensor_proto->add_int64_data(dyn_cast<Int8Imm>(elem)->value());
    } else if (elem->isa<Int16Imm>()) {
      tensor_proto->add_int64_data(dyn_cast<Int16Imm>(elem)->value());
    } else if (elem->isa<Int32Imm>()) {
      tensor_proto->add_int64_data(dyn_cast<Int32Imm>(elem)->value());
    } else if (elem->isa<Int64Imm>()) {
      tensor_proto->add_int64_data(dyn_cast<Int64Imm>(elem)->value());
    } else if (elem->isa<FP32Imm>()) {
      tensor_proto->add_float_data(dyn_cast<FP32Imm>(elem)->value());
    } else {
      MS_LOG(EXCEPTION) << "Convert tuple to tensor fail, unexpected tuple element type " << elem->type()->type_name()
                        << ".";
    }
  }
}

void OnnxExporter::SetTensorData(const ValuePtr &value, onnx::TensorProto *tensor_proto) {
  if (value->isa<Int32Imm>()) {
    auto attr_value = dyn_cast<Int32Imm>(value)->value();
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT32);
    tensor_proto->add_int32_data(attr_value);
  } else if (value->isa<Int64Imm>()) {
    auto attr_value = dyn_cast<Int64Imm>(value)->value();
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT64);
    tensor_proto->add_int64_data(attr_value);
  } else if (value->isa<tensor::Tensor>()) {
    auto data = dyn_cast<tensor::Tensor>(value);
    tensor_proto->set_raw_data(data->data_c(), static_cast<size_t>(data->data().nbytes()));
    auto dtype = data->data_type();
    auto shape = data->shape_c();

    tensor_proto->set_data_type(GetOnnxDataType(dtype));
    for (const auto dim : shape) {
      tensor_proto->add_dims(dim);
    }
  } else if (value->isa<ValueTuple>()) {  // Note: this is a tuple of primitives, not Tensors
    ConvertTupleToTensor(value, tensor_proto);
  } else {
    MS_LOG(EXCEPTION) << "Need to set value " << value->ToString() << " attribute for Constant node";
  }
}

std::string GetOnnxProtoString(const FuncGraphPtr &func_graph) {
  OnnxExporter exporter;
  return exporter.GetOnnxProtoString(func_graph);
}
}  // namespace mindspore
