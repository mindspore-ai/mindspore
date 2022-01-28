/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>

#include "utils/hash_map.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "ir/func_graph.h"
#include "base/core_ops.h"
#include "proto/onnx.pb.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
const int ONNX_VERSION = 11;
const int kZeroNum = 0;
const int kOneNum = 1;
const int kTwoNum = 2;
const int kThreeNum = 3;
const int kFourNum = 4;
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

std::string MakeOutputName(const std::string &node_name, int output_index) {
  return node_name + "_" + std::to_string(output_index);
}

namespace fp16 {
uint32_t FieldMask(unsigned int field_size) {
  const unsigned int BYTE_SIZE = 8;
  uint32_t mask = std::numeric_limits<uint32_t>::max();
  return mask >> (BYTE_SIZE * sizeof(mask) - field_size);
}

uint32_t ExponentBias(unsigned int exponent_size) { return (1 << (exponent_size - 1)) - 1; }

uint32_t Fp32ToFp16(float value) {
  const unsigned int FP32_M = 23;
  const unsigned int FP32_E = 32 - 1 - FP32_M;
  const unsigned int FP16_M = 10;
  const unsigned int FP16_E = 16 - 1 - FP16_M;

  uint32_t fp32_bits;
  std::memcpy(reinterpret_cast<std::byte *>(&fp32_bits), reinterpret_cast<std::byte *>(&value), sizeof(value));

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
  initializer->add_dims(values.size());
  for (auto value : values) {
    initializer->add_int64_data(value);
  }
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
    op_attrs_.emplace_back(OpAttrInfo(attr_name, onnx_attr_name, onnx_attr_type, fn_gen_attr));
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

OPERATOR_ONNX_CONVERT_DEFINE(Add, Add, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Mul, Mul, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Pow, Pow, OpNameInfo())

OPERATOR_ONNX_CONVERT_DEFINE(ReLU, Relu, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Sigmoid, Sigmoid, OpNameInfo())

OPERATOR_ONNX_CONVERT_DEFINE(Flatten, Flatten, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Squeeze, Squeeze,
                             OpNameInfo().Attr("axis", "axes", onnx::AttributeProto_AttributeType_INTS,
                                               SetAttrTupleValueToProto<0>))

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
OPERATOR_ONNX_CONVERT_DEFINE(StridedSlice, Slice, OpNameInfo())
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

#define OP_CONVERT_FUNCTION_NAME(name) GetOpOnnxConvertInfo_##name

void RegisterOpConverters(const std::function<void(OpNameInfo &&)> &fn) {
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
  fn(OP_CONVERT_FUNCTION_NAME(MaxPoolWithArgmax)());
  fn(OP_CONVERT_FUNCTION_NAME(AvgPool)());

  fn(OP_CONVERT_FUNCTION_NAME(Squeeze)());
  fn(OP_CONVERT_FUNCTION_NAME(BatchNorm)());
  fn(OP_CONVERT_FUNCTION_NAME(MatMul)());
  fn(OP_CONVERT_FUNCTION_NAME(MakeTuple)());
  fn(OP_CONVERT_FUNCTION_NAME(RealDiv)());
  fn(OP_CONVERT_FUNCTION_NAME(BiasAdd)());
  fn(OP_CONVERT_FUNCTION_NAME(Sub)());
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
  fn(OP_CONVERT_FUNCTION_NAME(Greater)());
  fn(OP_CONVERT_FUNCTION_NAME(LogicalAnd)());
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

  void ExportFuncGraph(const FuncGraphPtr &func_graph, onnx::GraphProto *graph_proto);
  void ExportParameters(const FuncGraphPtr &func_graph, onnx::GraphProto *graph_proto);

  size_t ExportPrimitive(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, size_t> *node_map_ptr,
                         const PrimitivePtr &prim, const std::vector<AnfNodePtr> &inputs,
                         onnx::GraphProto *graph_proto);

  static onnx::TensorProto_DataType GetOnnxDataType(TypeId type_id);
  static onnx::TensorProto_DataType GetOutputType(const AnfNodePtr &node, int64_t output_index = -1);
  void SetValueInfoType(const AnfNodePtr &node, onnx::ValueInfoProto *value_proto, bool is_output = false);
  void SetTensorProtoInfo(const ParameterPtr &param, onnx::TensorProto *tensor_proto);

  void MatchAndMark(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                    mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr);
  void MatchAndMarkCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                         mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr);
  void ExportNodes(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, size_t> *node_map_ptr,
                   onnx::GraphProto *graph_proto);

  void ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                   onnx::GraphProto *graph_proto);

  void ExportPrimReshape(const FuncGraphPtr &func_graph, const CNodePtr &node,
                         std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimReduce(const FuncGraphPtr &func_graph, const CNodePtr &node,
                        std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTranspose(const FuncGraphPtr &func_graph, const CNodePtr &node,
                           std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &node,
                              std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  onnx::NodeProto *PrimResizeExportHelper(const FuncGraphPtr &, const CNodePtr &node,
                                          std::map<AnfNodePtr, size_t> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto);
  void ExportPrimResizeNearestNeighbor(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                       std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimResizeBilinear(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimExpandDims(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimBatchMatMul(const FuncGraphPtr &func_graph, const CNodePtr &node,
                             std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimGeLU(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                      onnx::GraphProto *graph_proto);
  void ExportPrimConcat(const FuncGraphPtr &func_graph, const CNodePtr &node,
                        std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimCast(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                      onnx::GraphProto *graph_proto);
  void ExportPrimPReLU(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportPrimReLU6(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportPrimDepthwiseConv2d(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                 std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTile(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                      onnx::GraphProto *graph_proto);
  void ExportPrimSquare(const FuncGraphPtr &func_graph, const CNodePtr &node,
                        std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimGatherV2(const FuncGraphPtr &func_graph, const CNodePtr &node,
                          std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTupleGetItem(const FuncGraphPtr &func_graph, const CNodePtr &node,
                              std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimTopK(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                      onnx::GraphProto *graph_proto);
  void ExportPrimSplit(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportPrimSlice(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportPrimArgMaxWithValue(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                 std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimGreaterEqual(const FuncGraphPtr &func_graph, const CNodePtr &node,
                              std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeConv(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportMergeGemm(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportMergeBatchNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeMaxPoolWithArgmax(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                    std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeLayerNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);

  void ExportOutput(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                    onnx::GraphProto *graph_proto);
  std::string GetNodeInputName(const AnfNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                               onnx::GraphProto *const graph_proto);

  void ConvertTupleToTensor(const ValuePtr &value, onnx::TensorProto *tensor_proto);
  void SetTensorData(const ValuePtr &value, onnx::TensorProto *tensor_proto);
  void SetConstantNodeProtoInfoForGeLU(onnx::NodeProto *const node_proto, std::string output,
                                       onnx::AttributeProto *const attr_proto, onnx::TensorProto *const tensor_proto,
                                       std::string tensor_name, float float_data);
  void SetTwoInputNodeProtoInfo(onnx::NodeProto *const node_proto, std::string output, std::string op_type,
                                std::string input_x, std::string input_y);
  void SetOneInputNodeProtoInfo(onnx::NodeProto *const node_proto, std::string output, std::string op_type,
                                std::string input);

  void SetCastNodeProtoInfo(onnx::NodeProto *const node_proto, std::string output, std::string input,
                            onnx::AttributeProto *const attr_proto, onnx::TensorProto_DataType i_type);

  void AddOutputWithCast(onnx::NodeProto *node_proto, const std::string &output_name,
                         onnx::TensorProto_DataType target_type, onnx::GraphProto *graph_proto);

  size_t AllocateNodeIndex() { return ++onnx_node_index_; }

  void ResetNodeIndex() { onnx_node_index_ = 0; }

  static int64_t GetInt64Value(const AnfNodePtr &node) {
    auto value_node_ptr = dyn_cast<ValueNode>(node);
    MS_EXCEPTION_IF_NULL(value_node_ptr);
    return GetValue<int64_t>(value_node_ptr->value());
  }

  onnx::ModelProto model_;

  size_t onnx_node_index_ = 0;
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
  ExportFuncGraph(func_graph, graph_proto);
  return model_.SerializeAsString();
}

void OnnxExporter::InitModelInfo() {
  model_.set_ir_version(onnx::IR_VERSION_2019_1_22);
  model_.set_producer_name("MindSpore");
  model_.set_producer_version("1.0");
  onnx::OperatorSetIdProto *opset_proto = model_.add_opset_import();
  opset_proto->set_version(ONNX_VERSION);
}

void OnnxExporter::ExportFuncGraph(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto) {
  std::map<AnfNodePtr, size_t> node_map;

  MS_LOG(INFO) << "Begin exporting onnx model for graph " << func_graph->ToString();

  onnx_node_index_ = func_graph->parameters().size();

  // set graph name
  graph_proto->set_name(func_graph->ToString());

  // export parameters
  // 1. all parameters (with or without default value) will be mapped to ONNX parameters
  // 2. parameters with default value will mapped to ONNX initializers
  ExportParameters(func_graph, graph_proto);

  // export computational nodes and output nodes
  ExportNodes(func_graph, &node_map, graph_proto);

  MS_LOG(INFO) << "End exporting onnx model for graph " << func_graph->ToString();
}

void OnnxExporter::ExportParameters(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto) {
  for (auto &param : func_graph->parameters()) {
    const ParameterPtr param_ptr = dyn_cast<Parameter>(param);
    if (param_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Parameter '" << param->ToString() << "' could not cast to parameter.";
    }

    // set onnx input.
    if (!param_ptr->has_default()) {
      onnx::ValueInfoProto *input_proto = graph_proto->add_input();
      input_proto->set_name(param_ptr->ToString());
      SetValueInfoType(param_ptr, input_proto);
    }
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

void OnnxExporter::SetValueInfoType(const AnfNodePtr &node, onnx::ValueInfoProto *const value_proto, bool is_output) {
  auto dtype = node->Type();
  auto shape = node->Shape();
  onnx::TypeProto *type_proto = value_proto->mutable_type();
  if (dtype->isa<TensorType>() && shape->isa<abstract::Shape>()) {
    auto tensor = dyn_cast<TensorType>(dtype);
    auto elem_type = tensor->element();
    const auto &dims = dyn_cast<abstract::Shape>(shape)->shape();
    // output type of 'Argmax' of MindSpore is int32, output type of 'ArgMax' of ONNX is int64
    auto type = is_output ? onnx::TensorProto_DataType_INT64 : GetOnnxDataType(elem_type->type_id());
    type_proto->mutable_tensor_type()->set_elem_type(type);

    for (const auto &dim : dims) {
      type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }
    if (dims.empty()) {
      type_proto->mutable_tensor_type()->mutable_shape();
    }
  }
}

void OnnxExporter::SetTensorProtoInfo(const ParameterPtr &param, onnx::TensorProto *const tensor_proto) {
  auto dtype = param->Type();
  auto shape = param->Shape();
  if (!dtype->isa<TensorType>() || !shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "Parameter " << param->name() << " is not a regular tensor, with value " << param->ToString();
  }

  auto tensor = dyn_cast<TensorType>(dtype);
  auto elem_type = tensor->element();
  const auto &dims = dyn_cast<abstract::Shape>(shape)->shape();
  tensor_proto->set_data_type(GetOnnxDataType(elem_type->type_id()));
  for (const auto &dim : dims) {
    tensor_proto->add_dims(dim);
  }
}

void OnnxExporter::MatchAndMark(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                                mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) {
  auto &op_merged_infos = *op_merged_infos_ptr;

  for (auto &node : nodes) {
    if (!node->isa<CNode>() || IsIgnoredIdentityNode(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == func_graph->get_return()) {
      // if the key `input` does not exist, just create a new one
      op_merged_infos[cnode].referred_count += 1;
    }
    for (auto &orig_input : cnode->inputs()) {
      if (HasAbstractMonad(orig_input)) {
        // Skip monad inputs.
        continue;
      }
      auto input = GetRealInput(orig_input);
      if (!input->isa<CNode>()) {
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
                                     mindspore::HashMap<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) {
  const std::vector<MergeRule> first_input_merge_rules = {
    {prim::kPrimBiasAdd, prim::kPrimConv2D, OP_MERGE_CONV},
    {prim::kPrimBiasAdd, prim::kPrimConv3D, OP_MERGE_CONV},
    {prim::kPrimBiasAdd, prim::kPrimConv3DTranspose, OP_MERGE_CONV},
    {prim::kPrimBiasAdd, prim::kPrimMatMul, OP_MERGE_GEMM},
    {prim::kPrimTupleGetItem, prim::kPrimBatchNorm, OP_MERGE_BATCH_NORM},
    {prim::kPrimTupleGetItem, prim::kPrimMaxPoolWithArgmax, OP_MERGE_MAXPOOL_WITH_ARGMAX},
    {prim::kPrimTupleGetItem, prim::kPrimLayerNorm, OP_MERGE_LAYER_NORM},
  };

  auto rule = std::find_if(first_input_merge_rules.begin(), first_input_merge_rules.end(), [&cnode](const auto &rule) {
    return cnode->IsApply(rule.node_type) && IsPrimitiveCNode(cnode->input(1), rule.prev_type);
  });
  if (rule != first_input_merge_rules.end()) {
    if (cnode->IsApply(prim::kPrimTupleGetItem) && GetInt64Value(cnode->input(kTwoNum)) != 0) {
      MS_LOG(EXCEPTION) << "Multiple outputs for node \"" << cnode->input(1)->ToString() << "\" are not supported";
    }
    auto &op_merged_infos = *op_merged_infos_ptr;
    op_merged_infos[cnode].mode = rule->merge_mode;
    op_merged_infos[cnode->input(1)].mode = OP_MERGE_IGNORE;
    op_merged_infos[cnode->input(1)].referred_count -= 1;
  }
}

/**
 * AnfNode
 * +-- CNode
 * +-- ANode
 * |   +-- Parameter
 * |   `-- ValueNode
 */
void OnnxExporter::ExportNodes(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, size_t> *node_map_ptr,
                               onnx::GraphProto *const graph_proto) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);

  mindspore::HashMap<AnfNodePtr, OpMergedInfo> op_merged_infos;
  MatchAndMark(func_graph, nodes, &op_merged_infos);
  int count = -1;
  for (const AnfNodePtr &node : nodes) {
    // skip when MakeTuple + UpdateState
    count++;
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode->IsApply(prim::kPrimMakeTuple)) {
      size_t i = IntToSize(count + 1);
      while (!nodes[i]->isa<CNode>()) {
        i++;
      }
      auto nextCNode = nodes[i]->cast<CNodePtr>();
      if (nextCNode->IsApply(prim::kPrimUpdateState) &&
          IsPrimitiveCNode(nextCNode->input(kTwoNum), prim::kPrimMakeTuple)) {
        continue;
      }
    }

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
      ExportOutput(func_graph, cnode, node_map_ptr, graph_proto);
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
      default:
        ExportCNode(func_graph, cnode, node_map_ptr, graph_proto);
        break;
    }
  }
}

void OnnxExporter::ExportPrimReshape(const FuncGraphPtr &, const CNodePtr &node,
                                     std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_shape = node->input(kTwoNum);
  std::string name_shape;
  if (input_shape->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[input_shape] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_shape = std::to_string(const_node_idx);
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

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type(prim::kPrimReshape->name());
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(name_x);
  node_proto->add_input(name_shape);
}

void OnnxExporter::ExportPrimReduce(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_axis = node->input(kTwoNum);
  auto keep_dims = GetOpAttribute<bool>(node, "keep_dims");

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;

  std::string name;
  if (node->IsApply(prim::kPrimReduceSum)) {
    name = "ReduceSum";
  } else if (node->IsApply(prim::kPrimReduceMean)) {
    name = "ReduceMean";
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

  AddReduceOp(name, input_data, std::to_string(node_idx), axes, keep_dims, graph_proto);
}

void OnnxExporter::ExportPrimTranspose(const FuncGraphPtr &, const CNodePtr &node,
                                       std::map<AnfNodePtr, size_t> *node_map_ptr,
                                       onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_perm = node->input(kTwoNum);
  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  auto name = prim::kPrimTranspose->name();

  node_proto->set_name(std::to_string(node_idx) + name);
  node_proto->set_op_type(name);
  node_proto->add_output(std::to_string(node_idx));
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

void OnnxExporter::ExportPrimStridedSlice(const FuncGraphPtr &, const CNodePtr &node,
                                          std::map<AnfNodePtr, size_t> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto begin = node->input(kTwoNum);
  auto name = prim::kPrimStridedSlice->name();
  std::string name_begin;
  if (begin->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[begin] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_begin = std::to_string(const_node_idx);
    node_proto->add_output(name_begin);

    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("value");

    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    ConvertTupleToTensor(dyn_cast<ValueNode>(begin)->value(), attr_proto->mutable_t());
  } else {
    MS_LOG(EXCEPTION) << "The input begin of StridedSlice is not a ValueNode! "
                      << "Need to insert op convert variable from tuple to tensor for " << name;
  }

  auto end = node->input(kThreeNum);
  std::string name_end;
  if (end->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[end] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_end = std::to_string(const_node_idx);
    node_proto->add_output(name_end);

    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("value");

    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    ConvertTupleToTensor(dyn_cast<ValueNode>(end)->value(), attr_proto->mutable_t());
  } else {
    MS_LOG(EXCEPTION) << "The input end of StridedSlice is not a ValueNode! "
                      << "Need to insert op convert variable from tuple to tensor for " << name;
  }

  auto x_shape = dyn_cast<abstract::Shape>(node->input(1)->Shape());
  int size = SizeToInt(x_shape->shape().size());
  std::vector<int32_t> axes_value;
  ValuePtr axes_value_ptr = nullptr;
  for (int i = 0; i < size; ++i) {
    axes_value.push_back(i);
  }
  axes_value_ptr = MakeValue<std::vector<int32_t>>(axes_value);
  auto axes = NewValueNode(axes_value_ptr)->cast<AnfNodePtr>();
  std::string name_axes;
  auto const_node_idx_axes = AllocateNodeIndex();
  (*node_map_ptr)[axes] = const_node_idx_axes;
  onnx::NodeProto *node_proto_axes = graph_proto->add_node();
  name_axes = std::to_string(const_node_idx_axes);
  node_proto_axes->add_output(name_axes);
  node_proto_axes->set_op_type("Constant");
  onnx::AttributeProto *attr_proto_axes = node_proto_axes->add_attribute();
  attr_proto_axes->set_name("value");
  attr_proto_axes->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  ConvertTupleToTensor(dyn_cast<ValueNode>(axes)->value(), attr_proto_axes->mutable_t());

  auto strides = node->input(kFourNum);
  std::string name_strides;
  if (strides->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[strides] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_strides = std::to_string(const_node_idx);
    node_proto->add_output(name_strides);

    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto_steps = node_proto->add_attribute();
    attr_proto_steps->set_name("value");
    attr_proto_steps->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    ConvertTupleToTensor(dyn_cast<ValueNode>(strides)->value(), attr_proto_steps->mutable_t());
  } else {
    MS_LOG(EXCEPTION) << "The input strides of StridedSlice is not a ValueNode! "
                      << "Need to insert op convert variable from tuple to tensor for " << name;
  }

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Slice");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(input_data);
  node_proto->add_input(name_begin);
  node_proto->add_input(name_end);
  node_proto->add_input(name_axes);
  node_proto->add_input(name_strides);
}

onnx::NodeProto *OnnxExporter::PrimResizeExportHelper(const FuncGraphPtr &, const CNodePtr &node,
                                                      std::map<AnfNodePtr, size_t> *node_map_ptr,
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
  std::string name_size;

  auto const_node_idx = AllocateNodeIndex();
  (*node_map_ptr)[size] = const_node_idx;
  onnx::NodeProto *node_proto_size = graph_proto->add_node();
  name_size = std::to_string(const_node_idx);
  node_proto_size->add_output(name_size);
  node_proto_size->set_op_type("Constant");
  onnx::AttributeProto *attr_proto = node_proto_size->add_attribute();
  attr_proto->set_name("value");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  ConvertTupleToTensor(resize_size_ptr, attr_proto->mutable_t());

  auto node_idx = AllocateNodeIndex();

  onnx::TensorProto *roi_initializer_proto = graph_proto->add_initializer();
  auto roi_name = std::to_string(node_idx) + "roi_initializer";
  roi_initializer_proto->set_name(roi_name);
  roi_initializer_proto->set_data_type(GetOnnxDataType(kNumberTypeFloat32));
  roi_initializer_proto->add_dims(0);

  onnx::TensorProto *scales_initializer_proto = graph_proto->add_initializer();
  auto scales_name = std::to_string(node_idx) + "scales_initializer";
  scales_initializer_proto->set_name(scales_name);
  scales_initializer_proto->set_data_type(GetOnnxDataType(kNumberTypeFloat32));
  scales_initializer_proto->add_dims(0);

  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();

  node_proto->set_op_type("Resize");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(input_data);
  node_proto->add_input(roi_name);
  node_proto->add_input(scales_name);
  node_proto->add_input(name_size);

  return node_proto;
}

void OnnxExporter::ExportPrimResizeNearestNeighbor(const FuncGraphPtr &graph, const CNodePtr &node,
                                                   std::map<AnfNodePtr, size_t> *node_map_ptr,
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
                                            std::map<AnfNodePtr, size_t> *node_map_ptr,
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
                                        std::map<AnfNodePtr, size_t> *node_map_ptr,
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
  new_shape.insert(new_shape.begin() + axis, kOneNum);
  auto new_shape_value = MakeValue<std::vector<int64_t>>(new_shape);
  auto shape = NewValueNode(new_shape_value)->cast<AnfNodePtr>();
  std::string name_shape;

  if (shape->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[shape] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_shape = std::to_string(const_node_idx);
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

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Reshape");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(input_x);
  node_proto->add_input(name_shape);
}

// MindSpore BatchMatMul -> ONNX Transpose + MatMul
void OnnxExporter::ExportPrimBatchMatMul(const FuncGraphPtr &, const CNodePtr &node,
                                         std::map<AnfNodePtr, size_t> *node_map_ptr,
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
    auto transpose_input_x_index = AllocateNodeIndex();
    onnx::NodeProto *transpose_inputx_node_proto = graph_proto->add_node();
    transpose_inputx_node_proto->add_input(input_x);
    transpose_inputx_node_proto->add_output(std::to_string(transpose_input_x_index));
    transpose_inputx_node_proto->set_op_type(prim::kPrimTranspose->name());
    onnx::AttributeProto *attr_proto = transpose_inputx_node_proto->add_attribute();
    attr_proto->set_name("perm");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (size_t i = 0; i < input_x_shape->shape().size() - kTwoNum; i++) {
      attr_proto->add_ints(SizeToLong(i));
    }
    attr_proto->add_ints(SizeToLong(input_x_shape->shape().size()) - IntToLong(kOneNum));
    attr_proto->add_ints(SizeToLong(input_x_shape->shape().size()) - IntToLong(kTwoNum));
    transpose_input_x_name = std::to_string(transpose_input_x_index);
  }
  if (transpose_b) {
    auto input_y_shape = dyn_cast<abstract::Shape>(node->input(kTwoNum)->Shape());
    // Add Transpose node after input_y of BatchMatMul
    auto transpose_input_y_index = AllocateNodeIndex();
    onnx::NodeProto *transpose_inputy_node_proto = graph_proto->add_node();
    transpose_inputy_node_proto->add_input(input_y);
    transpose_inputy_node_proto->add_output(std::to_string(transpose_input_y_index));
    transpose_inputy_node_proto->set_op_type(prim::kPrimTranspose->name());
    onnx::AttributeProto *attr_proto = transpose_inputy_node_proto->add_attribute();
    attr_proto->set_name("perm");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (size_t i = 0; i < input_y_shape->shape().size() - kTwoNum; i++) {
      attr_proto->add_ints(SizeToLong(i));
    }
    attr_proto->add_ints(SizeToLong(input_y_shape->shape().size()) - IntToLong(kOneNum));
    attr_proto->add_ints(SizeToLong(input_y_shape->shape().size()) - IntToLong(kTwoNum));
    transpose_input_y_name = std::to_string(transpose_input_y_index);
  }

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("MatMul");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->set_name(std::to_string(node_idx) + "MatMul");
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

void OnnxExporter::SetConstantNodeProtoInfoForGeLU(onnx::NodeProto *const node_proto, const std::string output,
                                                   onnx::AttributeProto *const attr_proto,
                                                   onnx::TensorProto *const tensor_proto, const std::string tensor_name,
                                                   const float float_data) {
  node_proto->set_op_type("Constant");
  node_proto->add_output(output);

  attr_proto->set_name("value");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);

  tensor_proto->set_name(tensor_name);
  tensor_proto->add_dims(static_cast<::google::protobuf::int64>(kOneNum));
  tensor_proto->set_data_type(GetOnnxDataType(kNumberTypeFloat32));
  tensor_proto->add_float_data(float_data);
}

void OnnxExporter::SetCastNodeProtoInfo(onnx::NodeProto *const node_proto, const std::string output,
                                        const std::string input, onnx::AttributeProto *const attr_proto,
                                        onnx::TensorProto_DataType i_type) {
  node_proto->set_op_type(prim::kPrimCast->name());
  node_proto->add_output(output);
  node_proto->add_input(input);

  attr_proto->set_name("to");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto->set_i(i_type);
}

void OnnxExporter::SetTwoInputNodeProtoInfo(onnx::NodeProto *const node_proto, const std::string output,
                                            const std::string op_type, const std::string input_x,
                                            const std::string input_y) {
  node_proto->add_output(output);
  node_proto->set_op_type(op_type);
  node_proto->add_input(input_x);
  node_proto->add_input(input_y);
}

void OnnxExporter::SetOneInputNodeProtoInfo(onnx::NodeProto *const node_proto, const std::string output,
                                            const std::string op_type, const std::string input) {
  node_proto->add_output(output);
  node_proto->set_op_type(op_type);
  node_proto->add_input(input);
}

// MindSpore GeLU -> ONNX 0.5 * X * (1.0 + tanh((sqrt(2/pi) * (x + 0.044715 * pow(x, 3)))))
void OnnxExporter::ExportPrimGeLU(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_x_node = node->input(kOneNum);
  auto dtype = input_x_node->Type();
  auto elem_type = dyn_cast<TensorType>(dtype)->element()->type_id();
  size_t pre_cast_node_idx = 0;

  // if type is float16, add cast node cast float16 to float32
  if (elem_type == kNumberTypeFloat16) {
    pre_cast_node_idx = AllocateNodeIndex();
    onnx::NodeProto *pre_cast_node_proto = graph_proto->add_node();
    onnx::AttributeProto *pre_cast_attr_proto = pre_cast_node_proto->add_attribute();
    SetCastNodeProtoInfo(pre_cast_node_proto, std::to_string(pre_cast_node_idx), input_x, pre_cast_attr_proto,
                         onnx::TensorProto_DataType_FLOAT);
  }

  // Add Pow node
  // Add input exponent node for Pow node
  auto exp_node_idx = AllocateNodeIndex();
  const float exponent_for_pow = 3.0;
  onnx::NodeProto *exp_node_proto = graph_proto->add_node();
  onnx::AttributeProto *exp_attr_proto = exp_node_proto->add_attribute();
  onnx::TensorProto *exp_tensor_proto = exp_attr_proto->mutable_t();
  SetConstantNodeProtoInfoForGeLU(exp_node_proto, std::to_string(exp_node_idx), exp_attr_proto, exp_tensor_proto,
                                  "exponent", exponent_for_pow);
  // Add pow node
  auto pow_idx = AllocateNodeIndex();
  auto pow_name = std::to_string(pow_idx);
  onnx::NodeProto *pow_node_proto = graph_proto->add_node();
  pow_node_proto->set_op_type("Pow");
  pow_node_proto->add_output(pow_name);
  if (elem_type == kNumberTypeFloat16) {
    pow_node_proto->add_input(std::to_string(pre_cast_node_idx));
  } else {
    pow_node_proto->add_input(input_x);
  }
  pow_node_proto->add_input(std::to_string(exp_node_idx));

  // Add first Mul node
  // Add input node for first Mul node
  auto fmul_input_node_idx = AllocateNodeIndex();
  const float weight_for_fmul = 0.044715;
  onnx::NodeProto *fmul_input_node_proto = graph_proto->add_node();
  onnx::AttributeProto *fmul_input_attr_proto = fmul_input_node_proto->add_attribute();
  onnx::TensorProto *fmul_input_tensor_proto = fmul_input_attr_proto->mutable_t();
  SetConstantNodeProtoInfoForGeLU(fmul_input_node_proto, std::to_string(fmul_input_node_idx), fmul_input_attr_proto,
                                  fmul_input_tensor_proto, "input_y_for_mul", weight_for_fmul);
  // Add first Mul Node
  auto fmul_name = std::to_string(AllocateNodeIndex());
  onnx::NodeProto *fmul_node_proto = graph_proto->add_node();
  SetTwoInputNodeProtoInfo(fmul_node_proto, fmul_name, "Mul", pow_name, std::to_string(fmul_input_node_idx));

  // Add first Add node
  auto fadd_name = std::to_string(AllocateNodeIndex());
  onnx::NodeProto *fadd_node_proto = graph_proto->add_node();
  if (elem_type == kNumberTypeFloat16) {
    fadd_node_proto->add_input(std::to_string(pre_cast_node_idx));
  } else {
    fadd_node_proto->add_input(input_x);
  }
  SetOneInputNodeProtoInfo(fadd_node_proto, fadd_name, "Add", fmul_name);

  // Add second Mul node
  // Add input node for second Mul node
  auto smul_input_node_idx = AllocateNodeIndex();
  const float weight_for_smul = 0.79788456;
  onnx::NodeProto *smul_input_node_proto = graph_proto->add_node();
  onnx::AttributeProto *smul_input_attr_proto = smul_input_node_proto->add_attribute();
  onnx::TensorProto *smul_input_tensor_proto = smul_input_attr_proto->mutable_t();
  SetConstantNodeProtoInfoForGeLU(smul_input_node_proto, std::to_string(smul_input_node_idx), smul_input_attr_proto,
                                  smul_input_tensor_proto, "input_y_for_smul", weight_for_smul);
  // Add second Mul Node
  auto smul_name = std::to_string(AllocateNodeIndex());
  onnx::NodeProto *smul_node_proto = graph_proto->add_node();
  SetTwoInputNodeProtoInfo(smul_node_proto, smul_name, "Mul", fadd_name, std::to_string(smul_input_node_idx));

  // Add tanh node
  auto tanh_name = std::to_string(AllocateNodeIndex());
  onnx::NodeProto *tanh_node_proto = graph_proto->add_node();
  SetOneInputNodeProtoInfo(tanh_node_proto, tanh_name, "Tanh", smul_name);

  // Add second Add node
  // Add input node for second add node
  auto sadd_input_node_idx = AllocateNodeIndex();
  onnx::NodeProto *sadd_input_node_proto = graph_proto->add_node();
  onnx::AttributeProto *sadd_input_attr_proto = sadd_input_node_proto->add_attribute();
  onnx::TensorProto *sadd_input_tensor_proto = sadd_input_attr_proto->mutable_t();
  SetConstantNodeProtoInfoForGeLU(sadd_input_node_proto, std::to_string(sadd_input_node_idx), sadd_input_attr_proto,
                                  sadd_input_tensor_proto, "input_y_for_sadd", 1.0);
  // Add second Add node
  auto sadd_name = std::to_string(AllocateNodeIndex());
  onnx::NodeProto *sadd_node_proto = graph_proto->add_node();
  SetTwoInputNodeProtoInfo(sadd_node_proto, sadd_name, "Add", tanh_name, std::to_string(sadd_input_node_idx));

  // Add third Mul node
  // Add input node for third Mul node
  auto tmul_input_node_idx = AllocateNodeIndex();
  onnx::NodeProto *tmul_input_node_proto = graph_proto->add_node();
  onnx::AttributeProto *tmul_input_attr_proto = tmul_input_node_proto->add_attribute();
  onnx::TensorProto *tmul_input_tensor_proto = tmul_input_attr_proto->mutable_t();
  SetConstantNodeProtoInfoForGeLU(tmul_input_node_proto, std::to_string(tmul_input_node_idx), tmul_input_attr_proto,
                                  tmul_input_tensor_proto, "input_y_for_tmul", weight_for_mul);
  // Add third Mul Node
  auto tmul_name = std::to_string(AllocateNodeIndex());
  onnx::NodeProto *tmul_node_proto = graph_proto->add_node();
  SetTwoInputNodeProtoInfo(tmul_node_proto, tmul_name, "Mul", sadd_name, std::to_string(tmul_input_node_idx));

  // Add fourth Mul Node
  auto fomul_node_idx = AllocateNodeIndex();
  onnx::NodeProto *fomul_node_proto = graph_proto->add_node();
  if (elem_type == kNumberTypeFloat16) {
    fomul_node_proto->add_input(std::to_string(pre_cast_node_idx));
  } else {
    fomul_node_proto->add_input(input_x);
  }
  SetOneInputNodeProtoInfo(fomul_node_proto, std::to_string(fomul_node_idx), "Mul", tmul_name);

  // if type is float16, add cast node cast output node from float16 to float32
  if (elem_type == kNumberTypeFloat16) {
    auto aft_cast_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[node] = aft_cast_node_idx;
    onnx::NodeProto *aft_cast_node_proto = graph_proto->add_node();
    onnx::AttributeProto *aft_cast_attr_proto = aft_cast_node_proto->add_attribute();
    SetCastNodeProtoInfo(aft_cast_node_proto, std::to_string(aft_cast_node_idx), std::to_string(fomul_node_idx),
                         aft_cast_attr_proto, onnx::TensorProto_DataType_FLOAT16);
  } else {
    (*node_map_ptr)[node] = fomul_node_idx;
  }
}

void OnnxExporter::ExportPrimConcat(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();

  AnfNodePtr op = node->input(kZeroNum);
  auto op_value = dyn_cast<ValueNode>(op);
  auto prim = dyn_cast<Primitive>(op_value->value());
  auto input_node = node->input(kOneNum)->cast<CNodePtr>();
  if (input_node->IsApply(prim::kPrimMakeTuple)) {
    node_proto->set_op_type("ConcatFromSequence");
  } else {
    node_proto->set_op_type("Concat");
  }

  // set attr axis
  onnx::AttributeProto *onnx_attr_proto = node_proto->add_attribute();
  onnx_attr_proto->set_name("axis");
  SetAttrValueToProto<Int64Imm>(prim->GetAttr("axis"), onnx::AttributeProto_AttributeType_INT, onnx_attr_proto, prim);
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(input_data);
}

void OnnxExporter::ExportPrimCast(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_type = node->input(kTwoNum);

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type(prim::kPrimCast->name());
  node_proto->add_output(std::to_string(node_idx));
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
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_slope = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);

  auto x_shape = dyn_cast<abstract::Shape>(node->input(kOneNum)->Shape());
  auto slope_shape = dyn_cast<abstract::Shape>(node->input(kTwoNum)->Shape());
  MS_EXCEPTION_IF_NULL(x_shape);
  MS_EXCEPTION_IF_NULL(slope_shape);

  // format of x is NCHW, input format is NCHW, if length of input_slope is 1, insert Unsqueeze [1,2]
  if (x_shape->shape().size() == kFourNum && slope_shape->shape().size() == kOneNum) {
    auto node_idx = AllocateNodeIndex();
    onnx::NodeProto *node_proto = graph_proto->add_node();
    node_proto->set_op_type("Unsqueeze");
    node_proto->add_output(std::to_string(node_idx));

    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    attr_proto->set_name("axes");
    attr_proto->add_ints(kOneNum);
    attr_proto->add_ints(kTwoNum);

    node_proto->add_input(input_slope);
    input_slope = std::to_string(node_idx);
  }

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("PRelu");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(input_x);
  node_proto->add_input(input_slope);
}

void OnnxExporter::ExportPrimReLU6(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto node_idx = AllocateNodeIndex();
  auto node_name = std::to_string(node_idx);
  (*node_map_ptr)[node] = node_idx;

  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto onnx_input_type = GetOutputType(node->input(kOneNum));
  AddClipOp(input_x_name, node_name, 0.0f, 6.0f, onnx_input_type, graph_proto);
}

void OnnxExporter::ExportPrimDepthwiseConv2d(const FuncGraphPtr &, const CNodePtr &node,
                                             std::map<AnfNodePtr, size_t> *node_map_ptr,
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
  auto node_idx = AllocateNodeIndex();
  onnx::NodeProto *node_proto = graph_proto->add_node();
  std::string name_w_shape = std::to_string(node_idx);
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
  node_idx = AllocateNodeIndex();
  node_proto = graph_proto->add_node();
  node_proto->set_op_type(prim::kPrimReshape->name());
  node_proto->add_input(input_w);
  node_proto->add_input(name_w_shape);
  input_w = std::to_string(node_idx);
  node_proto->add_output(input_w);

  // add conv node
  node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  node_proto = graph_proto->add_node();
  node_proto->set_op_type("Conv");
  node_proto->add_input(input_x);
  node_proto->add_input(input_w);
  node_proto->add_output(std::to_string(node_idx));
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
                                  std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto multiples = node->input(kTwoNum);
  std::string name_multiples;
  if (multiples->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[multiples] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_multiples = std::to_string(const_node_idx);
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

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Tile");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(name_x);
  node_proto->add_input(name_multiples);
}

void OnnxExporter::ExportPrimSquare(const FuncGraphPtr &, const CNodePtr &node,
                                    std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  std::string name_exponent;
  auto const_node_idx = AllocateNodeIndex();
  onnx::NodeProto *node_proto_exp = graph_proto->add_node();
  name_exponent = std::to_string(const_node_idx);
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

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Pow");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(name_x);
  node_proto->add_input(name_exponent);
}

void OnnxExporter::ExportPrimGatherV2(const FuncGraphPtr &, const CNodePtr &node,
                                      std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto name_indices = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto axis = node->input(kThreeNum)->cast<ValueNodePtr>()->value();
  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Gather");
  node_proto->add_output(std::to_string(node_idx));
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
                                          std::map<AnfNodePtr, size_t> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto) {
  auto index = GetInt64Value(node->input(kTwoNum));

  auto input_node_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_name = MakeOutputName(input_node_name, index);

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;

  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Identity");
  node_proto->add_input(input_name);
  node_proto->add_output(std::to_string(node_idx));
}

void OnnxExporter::ExportPrimTopK(const FuncGraphPtr &, const CNodePtr &node,
                                  std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto x_input_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);

  auto node_idx = AllocateNodeIndex();
  auto node_name = std::to_string(node_idx);
  (*node_map_ptr)[node] = node_idx;

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

void OnnxExporter::ExportPrimSplit(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto node_idx = AllocateNodeIndex();
  auto node_name = std::to_string(node_idx);
  (*node_map_ptr)[node] = node_idx;
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
  if (input_shape[axis] % output_num != 0) {
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
    split_attr_proto->add_ints(input_shape[axis] / output_num);
  }
}

void OnnxExporter::ExportPrimSlice(const FuncGraphPtr &, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto node_idx = AllocateNodeIndex();
  auto node_name = std::to_string(node_idx);
  (*node_map_ptr)[node] = node_idx;
  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto begin_input_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto size_input_name = GetNodeInputName(node->input(kThreeNum), node_map_ptr, graph_proto);

  auto end_name = node_name + "end";
  AddOp("Add", {begin_input_name, size_input_name}, {end_name}, graph_proto);
  AddOp("Slice", {input_x_name, begin_input_name, end_name}, {node_name}, graph_proto);
}

void OnnxExporter::ExportPrimArgMaxWithValue(const FuncGraphPtr &, const CNodePtr &node,
                                             std::map<AnfNodePtr, size_t> *node_map_ptr,
                                             onnx::GraphProto *const graph_proto) {
  auto node_idx = AllocateNodeIndex();
  auto node_name = std::to_string(node_idx);
  (*node_map_ptr)[node] = node_idx;
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

void OnnxExporter::ExportPrimGreaterEqual(const FuncGraphPtr &, const CNodePtr &node,
                                          std::map<AnfNodePtr, size_t> *node_map_ptr,
                                          onnx::GraphProto *const graph_proto) {
  auto node_idx = AllocateNodeIndex();
  auto node_name = std::to_string(node_idx);
  (*node_map_ptr)[node] = node_idx;

  auto input_x_name = GetNodeInputName(node->input(kOneNum), node_map_ptr, graph_proto);
  auto input_y_name = GetNodeInputName(node->input(kTwoNum), node_map_ptr, graph_proto);
  auto less_name = node_name + "less";

  AddOp("Less", {input_x_name, input_y_name}, {less_name}, graph_proto);
  AddOp("Not", {less_name}, {node_name}, graph_proto);
}

void OnnxExporter::ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node,
                               std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  using ExportFunc = std::function<void(OnnxExporter *, const FuncGraphPtr &, const CNodePtr &,
                                        std::map<AnfNodePtr, size_t> *, onnx::GraphProto *const)>;
  static std::vector<std::pair<PrimitivePtr, ExportFunc>> export_table = {
    {prim::kPrimReshape, &OnnxExporter::ExportPrimReshape},
    {prim::kPrimReduceMean, &OnnxExporter::ExportPrimReduce},
    {prim::kPrimReduceSum, &OnnxExporter::ExportPrimReduce},
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
    {prim::kPrimSplit, &OnnxExporter::ExportPrimSplit},
    {prim::kPrimSlice, &OnnxExporter::ExportPrimSlice},
    {prim::kPrimArgMaxWithValue, &OnnxExporter::ExportPrimArgMaxWithValue},
    {prim::kPrimGreaterEqual, &OnnxExporter::ExportPrimGreaterEqual},
    {prim::kPrimExpandDims, &OnnxExporter::ExportPrimExpandDims},
    {prim::kPrimBatchMatMul, &OnnxExporter::ExportPrimBatchMatMul},
    {prim::kPrimGeLU, &OnnxExporter::ExportPrimGeLU},
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
  auto op_value = dyn_cast<ValueNode>(op);
  if (op_value == nullptr) {
    MS_LOG(EXCEPTION) << "Need to support node op type " << op->type_name();
  }
  auto prim = dyn_cast<Primitive>(op_value->value());
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Need to support node op type " << op_value->value()->type_name();
  }

  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim, op_inputs, graph_proto);
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
    auto element_type = tuple_type->elements()[output_index];
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
                                     onnx::TensorProto_DataType target_type, onnx::GraphProto *graph_proto) {
  if (target_type == onnx::TensorProto_DataType_UNDEFINED) {
    node_proto->add_output(output_name);
  } else {
    auto output_to_cast_name = output_name + "_output_to_cast";
    node_proto->add_output(output_to_cast_name);
    AddCastOp(output_to_cast_name, output_name, target_type, graph_proto);
  }
}

size_t OnnxExporter::ExportPrimitive(const FuncGraphPtr &, std::map<AnfNodePtr, size_t> *node_map_ptr,
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
  auto node_idx = AllocateNodeIndex();
  auto node_name = std::to_string(node_idx);

  std::vector<onnx::TensorProto_DataType> output_cast_types(op_convert_info.num_outputs(),
                                                            onnx::TensorProto_DataType_UNDEFINED);
  // Cast inputs if needed
  for (const auto &rule : op_convert_info.input_casts()) {
    auto original_type = GetOutputType(inputs[rule.input_index]);
    if (original_type != rule.input_type) {
      continue;
    }

    auto cast_input_name = node_name + "cast_input_" + std::to_string(rule.input_index);
    AddCastOp(input_list[rule.input_index], cast_input_name, rule.target_type, graph_proto);
    input_list[rule.input_index] = cast_input_name;

    auto output_cast = std::find_if(
      op_convert_info.output_casts().begin(), op_convert_info.output_casts().end(), [&rule](const OutputConversion &x) {
        return x.mode == OutputConversion::Mode::INPUT && x.input_with_matching_type == rule.input_index;
      });
    if (output_cast != op_convert_info.output_casts().end()) {
      output_cast_types[output_cast->output_index] = original_type;
    }
  }

  for (const auto &output_cast : op_convert_info.output_casts()) {
    if (output_cast.mode == OutputConversion::Mode::FIXED) {
      output_cast_types[output_cast.output_index] = output_cast.target_type;
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
      AddOutputWithCast(node_proto, output_name, output_cast_types[i], graph_proto);
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
  return node_idx;
}

void OnnxExporter::ExportMergeConv(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto conv_node = dyn_cast<CNode>(node->input(kOneNum));
  auto input_x = conv_node->input(kOneNum);  // conv input x
  auto input_w = conv_node->input(kTwoNum);  // conv weight(filter)
  auto input_b = node->input(kTwoNum);       // conv bias

  PrimitivePtr prim_conv = dyn_cast<Primitive>((dyn_cast<ValueNode>(conv_node->input(kZeroNum)))->value());
  std::vector<AnfNodePtr> inputs{input_x, input_w, input_b};
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_conv, inputs, graph_proto);
}

void OnnxExporter::ExportMergeGemm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto matmul_node = dyn_cast<CNode>(node->input(kOneNum));
  auto input_x = matmul_node->input(kOneNum);  // matmul input x
  auto input_y = matmul_node->input(kTwoNum);  // matmul input y
  auto input_b = node->input(kTwoNum);         // matmul bias

  PrimitivePtr prim_matmul = dyn_cast<Primitive>((dyn_cast<ValueNode>(matmul_node->input(kZeroNum)))->value());
  std::vector<AnfNodePtr> inputs{input_x, input_y, input_b};
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_matmul, inputs, graph_proto);
}

void OnnxExporter::ExportMergeBatchNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                        std::map<AnfNodePtr, size_t> *node_map_ptr,
                                        onnx::GraphProto *const graph_proto) {
  auto batch_norm_node = dyn_cast<CNode>(node->input(kOneNum));

  PrimitivePtr prim_batch_norm = dyn_cast<Primitive>((dyn_cast<ValueNode>(batch_norm_node->input(kZeroNum)))->value());
  std::vector<AnfNodePtr> inputs;
  for (size_t i = 1; i < batch_norm_node->inputs().size(); i++) {
    inputs.push_back(batch_norm_node->input(i));
  }
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_batch_norm, inputs, graph_proto);
}

void OnnxExporter::ExportMergeMaxPoolWithArgmax(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                                std::map<AnfNodePtr, size_t> *node_map_ptr,
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
                                        std::map<AnfNodePtr, size_t> *node_map_ptr,
                                        onnx::GraphProto *const graph_proto) {
  auto LayerNormNode = dyn_cast<CNode>(node->input(kOneNum));
  auto layernorm_input_x = GetNodeInputName(LayerNormNode->input(kOneNum), node_map_ptr, graph_proto);
  auto layernorm_input_gamma = GetNodeInputName(LayerNormNode->input(kTwoNum), node_map_ptr, graph_proto);
  auto layernorm_input_beta = GetNodeInputName(LayerNormNode->input(kThreeNum), node_map_ptr, graph_proto);

  auto layernorm_input_x_node = LayerNormNode->input(kOneNum);
  auto dtype = layernorm_input_x_node->Type();
  auto elem_type = dyn_cast<TensorType>(dtype)->element()->type_id();
  size_t pre_cast_node_idx = 0;

  // if type is float16, add cast node cast type from float16 to float32
  if (elem_type == kNumberTypeFloat16) {
    pre_cast_node_idx = AllocateNodeIndex();
    onnx::NodeProto *pre_cast_node_proto = graph_proto->add_node();
    onnx::AttributeProto *pre_cast_attr_proto = pre_cast_node_proto->add_attribute();
    SetCastNodeProtoInfo(pre_cast_node_proto, std::to_string(pre_cast_node_idx), layernorm_input_x, pre_cast_attr_proto,
                         onnx::TensorProto_DataType_FLOAT);
  }

  // reshape before MeanVarianceNormalization
  auto input_shape = dyn_cast<abstract::Shape>(LayerNormNode->input(kOneNum)->Shape());
  std::vector<int64_t> new_input_shape;
  int64_t n_shape = 1;
  int64_t c_shape = 1;
  int64_t h_shape = 1;
  size_t input_shape_size = input_shape->shape().size();
  for (size_t i = 0; i < input_shape_size - 1; i++) {
    c_shape = c_shape * input_shape->shape()[i];
  }
  new_input_shape.push_back(n_shape);
  new_input_shape.push_back(c_shape);
  new_input_shape.push_back(h_shape);
  new_input_shape.push_back(input_shape->shape()[input_shape_size - kOneNum]);

  // Add shape node for reshape(before MeanVarianceNormalization)
  auto new_shape_value = MakeValue<std::vector<int64_t>>(new_input_shape);
  auto shape_node = NewValueNode(new_shape_value)->cast<AnfNodePtr>();
  auto shape_node_idx = AllocateNodeIndex();

  onnx::NodeProto *shape_node_proto = graph_proto->add_node();
  shape_node_proto->add_output(std::to_string(shape_node_idx));
  shape_node_proto->set_op_type("Constant");
  onnx::AttributeProto *shape_attr_proto = shape_node_proto->add_attribute();
  shape_attr_proto->set_name("value");
  shape_attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  ConvertTupleToTensor(dyn_cast<ValueNode>(shape_node)->value(), shape_attr_proto->mutable_t());

  // Add reshape node before MeanVarianceNormalization
  auto pre_reshape_node_idx = AllocateNodeIndex();
  onnx::NodeProto *pre_reshape_node_proto = graph_proto->add_node();
  pre_reshape_node_proto->set_op_type("Reshape");
  pre_reshape_node_proto->add_output(std::to_string(pre_reshape_node_idx));
  if (elem_type == kNumberTypeFloat16) {
    pre_reshape_node_proto->add_input(std::to_string(pre_cast_node_idx));
  } else {
    pre_reshape_node_proto->add_input(layernorm_input_x);
  }
  pre_reshape_node_proto->add_input(std::to_string(shape_node_idx));

  // MeanVarianceNormalization
  auto meanvariancenormal_node_idx = AllocateNodeIndex();
  onnx::NodeProto *meanvariancenormal_node_proto = graph_proto->add_node();
  meanvariancenormal_node_proto->set_op_type("MeanVarianceNormalization");
  meanvariancenormal_node_proto->add_output(std::to_string(meanvariancenormal_node_idx));
  meanvariancenormal_node_proto->add_input(std::to_string(pre_reshape_node_idx));

  // if cast type from float16 to float32, add cast node cast type from float32 to float16
  size_t aft_cast_node_idx = 0;
  if (elem_type == kNumberTypeFloat16) {
    aft_cast_node_idx = AllocateNodeIndex();
    onnx::NodeProto *aft_cast_node_proto = graph_proto->add_node();
    onnx::AttributeProto *aft_cast_attr_proto = aft_cast_node_proto->add_attribute();
    SetCastNodeProtoInfo(aft_cast_node_proto, std::to_string(aft_cast_node_idx),
                         std::to_string(meanvariancenormal_node_idx), aft_cast_attr_proto,
                         onnx::TensorProto_DataType_FLOAT16);
  }

  // Add mul and add node
  auto mul_node_idx = AllocateNodeIndex();
  onnx::NodeProto *mul_node_proto = graph_proto->add_node();
  mul_node_proto->set_op_type("Mul");
  if (elem_type == kNumberTypeFloat16) {
    mul_node_proto->add_input(std::to_string(aft_cast_node_idx));
  } else {
    mul_node_proto->add_input(std::to_string(meanvariancenormal_node_idx));
  }
  mul_node_proto->add_input(layernorm_input_gamma);
  mul_node_proto->add_output(std::to_string(mul_node_idx));

  // add beta
  auto add_node_idx = AllocateNodeIndex();
  onnx::NodeProto *add_node_proto = graph_proto->add_node();
  SetTwoInputNodeProtoInfo(add_node_proto, std::to_string(add_node_idx), "Add", std::to_string(mul_node_idx),
                           layernorm_input_beta);

  // reshape after MeanVarianceNormalization
  // Add shape node for reshape(after MeanVarianceNormalization)
  auto output_shape_value = MakeValue<std::vector<int64_t>>(input_shape->shape());
  auto output_shape_node = NewValueNode(output_shape_value)->cast<AnfNodePtr>();
  auto output_shape_node_idx = AllocateNodeIndex();

  onnx::NodeProto *output_shape_node_proto = graph_proto->add_node();
  output_shape_node_proto->add_output(std::to_string(output_shape_node_idx));
  output_shape_node_proto->set_op_type("Constant");
  onnx::AttributeProto *output_shape_attr_proto = output_shape_node_proto->add_attribute();
  output_shape_attr_proto->set_name("value");
  output_shape_attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  ConvertTupleToTensor(dyn_cast<ValueNode>(output_shape_node)->value(), output_shape_attr_proto->mutable_t());
  // Add reshape node after MeanVarianceNormalization
  auto aft_reshape_node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = aft_reshape_node_idx;
  onnx::NodeProto *aft_reshape_node_proto = graph_proto->add_node();
  aft_reshape_node_proto->set_op_type("Reshape");
  aft_reshape_node_proto->add_output(std::to_string(aft_reshape_node_idx));
  aft_reshape_node_proto->add_input(std::to_string(add_node_idx));
  aft_reshape_node_proto->add_input(std::to_string(output_shape_node_idx));
}

void OnnxExporter::ExportOutput(const FuncGraphPtr &, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                                onnx::GraphProto *const graph_proto) {
  if (node->inputs().size() != kTwoNum) {
    MS_LOG(EXCEPTION) << "Number of inputs of return node is not equal to 2.";
  }
  AnfNodePtr arg = node->input(1);
  std::string name = GetNodeInputName(arg, node_map_ptr, graph_proto);
  onnx::ValueInfoProto *output_proto = graph_proto->add_output();
  output_proto->set_name(name);
  SetValueInfoType(arg, output_proto, false);
}

std::string OnnxExporter::GetNodeInputName(const AnfNodePtr &orig_node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                                           onnx::GraphProto *const graph_proto) {
  auto node = GetRealInput(orig_node);
  if (node->isa<CNode>()) {
    auto iter = node_map_ptr->find(node);
    if (iter == node_map_ptr->end()) {
      MS_LOG(EXCEPTION) << "Can not find node '" << node->DebugString() << "' in node_map";
    }
    return std::to_string(iter->second);
  }

  if (node->isa<Parameter>() && !node->cast<ParameterPtr>()->has_default()) {
    return node->ToString();
  }

  // for ValueNode or Parameter with default input, create an initializer
  if (node->isa<ValueNode>() || node->isa<Parameter>()) {
    auto iter = node_map_ptr->find(node);
    if (iter != node_map_ptr->end()) {
      return std::to_string(iter->second);
    }
    // the id number starts at 1, so the id of created node should be size of map plus one
    auto node_idx = AllocateNodeIndex();
    (*node_map_ptr)[node] = node_idx;
    std::string node_name = std::to_string(node_idx);

    ValuePtr value;
    if (node->isa<ValueNode>()) {
      value = node->cast<ValueNodePtr>()->value();
    } else if (node->isa<Parameter>()) {
      value = node->cast<ParameterPtr>()->default_param();
    } else {
      MS_LOG(EXCEPTION) << "Impossible branch";
    }

    onnx::TensorProto *initializer_proto = graph_proto->add_initializer();
    initializer_proto->set_name(node_name);
    SetTensorData(value, initializer_proto);

    return node_name;
  }

  MS_LOG(EXCEPTION) << "Unexpected node type " << node->type_name();
}

void OnnxExporter::ConvertTupleToTensor(const ValuePtr &value, onnx::TensorProto *const tensor_proto) {
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
