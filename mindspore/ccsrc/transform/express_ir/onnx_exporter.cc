/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <utility>
#include <functional>

#include "ir/tensor.h"
#include "ir/param_info.h"
#include "ir/func_graph.h"
#include "base/core_ops.h"
#include "proto/onnx.pb.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
enum OpMergeMode {
  OP_MERGE_UNDEFINED = 0,            // undefined behavior
  OP_MERGE_IGNORE = 1,               // indicate an input op merged into other op in compute node list
  OP_MERGE_CONV = 2,                 // indicate `MindSpore Conv + BiasAdd` --> `ONNX Conv`
  OP_MERGE_GEMM = 3,                 // indicate `MindSpore MatMul + BiasAdd` --> `ONNX Gemm`
  OP_MERGE_BATCH_NORM = 4,           // indicate `MindSpore BatchNorm(x)[0]` --> `ONNX BatchNormalization`
  OP_MERGE_MAXPOOL_WITH_ARGMAX = 5,  // indicate `MindSpore MaxPoolWithArgmax(x)[0]` --> `ONNX MaxPool`
};

struct OpMergedInfo {
  OpMergeMode mode = OP_MERGE_UNDEFINED;
  int referred_count = 0;
};

using GenAttrFuncType =
  std::function<void(ValuePtr, onnx::AttributeProto_AttributeType, onnx::AttributeProto *, const PrimitivePtr &)>;

static AnfNodePtr GetRealInput(const AnfNodePtr &origin_input) {
  AnfNodePtr input = origin_input;
  while (IsPrimitiveCNode(input, prim::kPrimDepend) || IsPrimitiveCNode(input, prim::kPrimLoad)) {
    // Skip Depend and Load cnodes.
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

 private:
  std::string op_type_;               // operator type of MindSpore
  std::string onnx_type_;             // corresponding ONNX operator type
  std::vector<OpAttrInfo> op_attrs_;  // operator attributes map info
};

#define OPERATOR_ONNX_CONVERT_DEFINE(name, onnx_name, impl) \
  OpNameInfo GetOpOnnxConvertInfo_##name() { return impl.set_op_type(#name).set_onnx_type(#onnx_name); }

OPERATOR_ONNX_CONVERT_DEFINE(Add, Add, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Mul, Mul, OpNameInfo())

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
    .Attr("pad_mode", "auto_pad", onnx::AttributeProto_AttributeType_STRING,
          [](ValuePtr value, onnx::AttributeProto_AttributeType, onnx::AttributeProto *const attr_proto,
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
              SetAttrTupleValueToProto(prim->GetAttr("pad_list"), onnx::AttributeProto_AttributeType_INTS, attr_proto,
                                       prim);
            }
          })
    .Attr("stride", "strides", onnx::AttributeProto_AttributeType_INTS, SetAttrTupleValueToProto<2>))
OPERATOR_ONNX_CONVERT_DEFINE(BiasAdd, Add, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(MatMul, Gemm,
                             OpNameInfo()
                               .Attr("transpose_a", "transA", onnx::AttributeProto_AttributeType_INT,
                                     SetAttrValueToProto<BoolImm>)
                               .Attr("transpose_b", "transB", onnx::AttributeProto_AttributeType_INT,
                                     SetAttrValueToProto<BoolImm>))

OPERATOR_ONNX_CONVERT_DEFINE(BatchNorm, BatchNormalization,
                             OpNameInfo().Attr("epsilon", "epsilon", onnx::AttributeProto_AttributeType_FLOAT,
                                               SetAttrValueToProto<FP32Imm>))

OPERATOR_ONNX_CONVERT_DEFINE(Reshape, Reshape, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(ReduceMean, ReduceMean, OpNameInfo())
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
                                     }))

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
OPERATOR_ONNX_CONVERT_DEFINE(Concat, Concat, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(RealDiv, Div, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(ReduceSum, ReduceSum, OpNameInfo())
OPERATOR_ONNX_CONVERT_DEFINE(Sub, Sub, OpNameInfo())

#define OP_CONVERT_FUNCTION_NAME(name) GetOpOnnxConvertInfo_##name

void RegisterOpConverters(const std::function<void(OpNameInfo &&)> &fn) {
  fn(OP_CONVERT_FUNCTION_NAME(Add)());
  fn(OP_CONVERT_FUNCTION_NAME(Mul)());

  fn(OP_CONVERT_FUNCTION_NAME(ReLU)());
  fn(OP_CONVERT_FUNCTION_NAME(Sigmoid)());

  fn(OP_CONVERT_FUNCTION_NAME(Conv2D)());
  fn(OP_CONVERT_FUNCTION_NAME(Argmax)());

  fn(OP_CONVERT_FUNCTION_NAME(Flatten)());
  fn(OP_CONVERT_FUNCTION_NAME(MaxPool)());
  fn(OP_CONVERT_FUNCTION_NAME(MaxPoolWithArgmax)());
  fn(OP_CONVERT_FUNCTION_NAME(AvgPool)());

  fn(OP_CONVERT_FUNCTION_NAME(Squeeze)());
  fn(OP_CONVERT_FUNCTION_NAME(BatchNorm)());
  fn(OP_CONVERT_FUNCTION_NAME(MatMul)());

  fn(OP_CONVERT_FUNCTION_NAME(MakeTuple)());
  fn(OP_CONVERT_FUNCTION_NAME(Concat)());
  fn(OP_CONVERT_FUNCTION_NAME(RealDiv)());
  fn(OP_CONVERT_FUNCTION_NAME(BiasAdd)());
  fn(OP_CONVERT_FUNCTION_NAME(Sub)());
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

  static const std::unordered_map<std::string, OpNameInfo> &GetOpConvertMap() { return GetSingleton().op_map_; }

  void Clear() noexcept { op_map_.clear(); }

 private:
  OpConvertRegistry() {}

  std::unordered_map<std::string, OpNameInfo> op_map_;
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
  void SetValueInfoType(const AnfNodePtr &node, onnx::ValueInfoProto *value_proto, bool is_output = false);
  void SetTensorProtoInfo(const ParameterPtr &param, onnx::TensorProto *tensor_proto);

  void MatchAndMark(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                    std::unordered_map<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr);
  void ExportNodes(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, size_t> *node_map_ptr,
                   onnx::GraphProto *graph_proto);

  void ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                   onnx::GraphProto *graph_proto);

  void ExportPrimReshape(const FuncGraphPtr &func_graph, const CNodePtr &node,
                         std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportPrimReduce(const FuncGraphPtr &func_graph, const CNodePtr &node,
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

  void ExportMergeConv(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportMergeGemm(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                       onnx::GraphProto *graph_proto);
  void ExportMergeBatchNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                            std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);
  void ExportMergeMaxPoolWithArgmax(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                    std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *graph_proto);

  void ExportOutput(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                    onnx::GraphProto *graph_proto);
  std::string GetNodeInputName(const AnfNodePtr &node, std::map<AnfNodePtr, size_t> *node_map_ptr,
                               onnx::GraphProto *const graph_proto);

  void ConvertTupleToTensor(const ValuePtr &value, onnx::TensorProto *tensor_proto);
  void SetNodeAttribute(const ValuePtr &value, onnx::NodeProto *node_proto);

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
  opset_proto->set_version(9);
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

    onnx::ValueInfoProto *input_proto = graph_proto->add_input();
    input_proto->set_name(param_ptr->ToString());
    SetValueInfoType(param_ptr, input_proto);

    if (!param_ptr->has_default()) {
      continue;
    }
    // parameter with default value is an ONNX initializer
    onnx::TensorProto *initializer_proto = graph_proto->add_initializer();
    initializer_proto->set_name(param_ptr->ToString());
    SetTensorProtoInfo(param_ptr, initializer_proto);
    // set value for initializer
    auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param_ptr->default_param());
    if (tensor) {
      initializer_proto->set_raw_data(tensor->data_c(), tensor->data().nbytes());
    }
  }
}

onnx::TensorProto_DataType OnnxExporter::GetOnnxDataType(TypeId type_id) {
  // clang-format off
  static std::unordered_map<int, onnx::TensorProto_DataType> type_map = {
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
                                std::unordered_map<AnfNodePtr, OpMergedInfo> *op_merged_infos_ptr) {
  std::unordered_map<AnfNodePtr, OpMergedInfo> &op_merged_infos = *op_merged_infos_ptr;

  for (auto &node : nodes) {
    if (!node->isa<CNode>()) {
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
    // MindSpore Conv + BiasAdd --> ONNX Conv
    if (cnode->IsApply(std::make_shared<Primitive>("BiasAdd")) &&
        IsPrimitiveCNode(cnode->input(1), prim::kPrimConv2D)) {
      op_merged_infos[cnode].mode = OP_MERGE_CONV;
      op_merged_infos[cnode->input(1)].mode = OP_MERGE_IGNORE;
      op_merged_infos[cnode->input(1)].referred_count -= 1;
    } else if (cnode->IsApply(std::make_shared<Primitive>("BiasAdd")) &&
               IsPrimitiveCNode(cnode->input(1), prim::kPrimMatMul)) {
      op_merged_infos[cnode].mode = OP_MERGE_GEMM;
      op_merged_infos[cnode->input(1)].mode = OP_MERGE_IGNORE;
      op_merged_infos[cnode->input(1)].referred_count -= 1;
    } else if (cnode->IsApply(prim::kPrimTupleGetItem) &&
               IsPrimitiveCNode(cnode->input(1), std::make_shared<Primitive>("BatchNorm")) &&
               GetInt64Value(cnode->input(2)) == 0) {
      op_merged_infos[cnode].mode = OP_MERGE_BATCH_NORM;
      op_merged_infos[cnode->input(1)].mode = OP_MERGE_IGNORE;
      op_merged_infos[cnode->input(1)].referred_count -= 1;
    } else if (cnode->IsApply(prim::kPrimTupleGetItem) &&
               IsPrimitiveCNode(cnode->input(1), std::make_shared<Primitive>("MaxPoolWithArgmax")) &&
               GetInt64Value(cnode->input(2)) == 0) {
      op_merged_infos[cnode].mode = OP_MERGE_MAXPOOL_WITH_ARGMAX;
      op_merged_infos[cnode->input(1)].mode = OP_MERGE_IGNORE;
      op_merged_infos[cnode->input(1)].referred_count -= 1;
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
void OnnxExporter::ExportNodes(const FuncGraphPtr &func_graph, std::map<AnfNodePtr, size_t> *node_map_ptr,
                               onnx::GraphProto *const graph_proto) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);

  std::unordered_map<AnfNodePtr, OpMergedInfo> op_merged_infos;
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
      default:
        ExportCNode(func_graph, cnode, node_map_ptr, graph_proto);
        break;
    }
  }
}

void OnnxExporter::ExportPrimReshape(const FuncGraphPtr & /*func_graph*/, const CNodePtr &node,
                                     std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto input_shape = node->input(2);
  std::string name_shape;
  if (input_shape->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[input_shape] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_shape = std::to_string(const_node_idx);
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

void OnnxExporter::ExportPrimReduce(const FuncGraphPtr & /*func_graph*/, const CNodePtr &node,
                                    std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto input_axis = node->input(2);

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  auto name = prim::kPrimReduceMean->name();
  if (node->IsApply(prim::kPrimReduceSum)) {
    name = prim::kPrimReduceSum->name();
  }
  node_proto->set_op_type(name);
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(input_data);

  if (input_axis->isa<ValueNode>()) {
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("axes");
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    auto axis_value = dyn_cast<ValueNode>(input_axis)->value();
    auto int_ptr = dyn_cast<Int32Imm>(axis_value);
    if (int_ptr == nullptr) {
      auto tuple_ptr = dyn_cast<ValueTuple>(axis_value);
      MS_EXCEPTION_IF_NULL(tuple_ptr);
      for (size_t i = 0; i < tuple_ptr->size(); ++i) {
        attr_proto->add_ints(GetValue<int64_t>((*tuple_ptr)[i]));
      }
    } else {
      attr_proto->add_ints(int_ptr->value());
    }
  } else {
    MS_LOG(EXCEPTION) << "Need to insert op convert variable from tuple to attributes for " << name;
  }
}

void OnnxExporter::ExportPrimCast(const FuncGraphPtr & /*func_graph*/, const CNodePtr &node,
                                  std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_data = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto input_type = node->input(2);

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

void OnnxExporter::ExportPrimPReLU(const FuncGraphPtr & /*func_graph*/, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto input_slope = GetNodeInputName(node->input(2), node_map_ptr, graph_proto);

  auto x_shape = dyn_cast<abstract::Shape>(node->input(1)->Shape());
  auto slope_shape = dyn_cast<abstract::Shape>(node->input(2)->Shape());
  MS_EXCEPTION_IF_NULL(x_shape);
  MS_EXCEPTION_IF_NULL(slope_shape);

  // format of x is NCHW, input format is NCHW, if length of input_slope is 1, insert Unsqueeze [1,2]
  if (x_shape->shape().size() == 4 && slope_shape->shape().size() == 1) {
    auto node_idx = AllocateNodeIndex();
    onnx::NodeProto *node_proto = graph_proto->add_node();
    node_proto->set_op_type("Unsqueeze");
    node_proto->add_output(std::to_string(node_idx));

    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INTS);
    attr_proto->set_name("axes");
    attr_proto->add_ints(1);
    attr_proto->add_ints(2);

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

void OnnxExporter::ExportPrimReLU6(const FuncGraphPtr & /*func_graph*/, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Clip");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(input_x);
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_FLOAT);
  attr_proto->set_name("min");
  attr_proto->set_f(0.f);
  attr_proto = node_proto->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_FLOAT);
  attr_proto->set_name("max");
  attr_proto->set_f(6.f);
}

void OnnxExporter::ExportPrimDepthwiseConv2d(const FuncGraphPtr & /*func_graph*/, const CNodePtr &node,
                                             std::map<AnfNodePtr, size_t> *node_map_ptr,
                                             onnx::GraphProto *const graph_proto) {
  auto input_x = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto input_w = GetNodeInputName(node->input(2), node_map_ptr, graph_proto);
  auto x_shape = dyn_cast<abstract::Shape>(node->input(1)->Shape());
  auto w_shape = dyn_cast<abstract::Shape>(node->input(2)->Shape());
  MS_EXCEPTION_IF_NULL(x_shape);
  MS_EXCEPTION_IF_NULL(w_shape);
  if (x_shape->shape().size() != 4 || w_shape->shape().size() != 4) {
    MS_LOG(EXCEPTION) << "DepthwiseConv2d input shape should be 4d.";
  }
  if (w_shape->shape()[0] != 1 && w_shape->shape()[1] != 1) {
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
  tensor_proto->add_int64_data(w_shape->shape()[1]);
  tensor_proto->add_int64_data(w_shape->shape()[0]);
  tensor_proto->add_int64_data(w_shape->shape()[2]);
  tensor_proto->add_int64_data(w_shape->shape()[3]);

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

void OnnxExporter::ExportPrimTile(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                  std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto multiples = node->input(2);
  std::string name_multiples;
  if (multiples->isa<ValueNode>()) {
    auto const_node_idx = AllocateNodeIndex();
    (*node_map_ptr)[multiples] = const_node_idx;
    onnx::NodeProto *node_proto = graph_proto->add_node();
    name_multiples = std::to_string(const_node_idx);
    node_proto->add_output(name_multiples);

    node_proto->set_op_type("Constant");
    onnx::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name("repeat");

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

void OnnxExporter::ExportPrimSquare(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                    std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  std::string name_exponent;
  auto const_node_idx = AllocateNodeIndex();
  onnx::NodeProto *node_proto_exp = graph_proto->add_node();
  name_exponent = std::to_string(const_node_idx);
  node_proto_exp->add_output(name_exponent);

  node_proto_exp->set_op_type("Constant");
  onnx::AttributeProto *attr_proto = node_proto_exp->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  tensor_proto->set_name("exponent");
  tensor_proto->add_dims(static_cast<::google::protobuf::int64>(1));
  tensor_proto->set_data_type(onnx::TensorProto_DataType_INT64);
  tensor_proto->add_int64_data(2);

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Pow");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(name_x);
  node_proto->add_input(name_exponent);
}

void OnnxExporter::ExportPrimGatherV2(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                      std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto name_x = GetNodeInputName(node->input(1), node_map_ptr, graph_proto);
  auto name_indices = GetNodeInputName(node->input(2), node_map_ptr, graph_proto);
  auto axis = node->input(3)->cast<ValueNodePtr>()->value();

  auto node_idx = AllocateNodeIndex();
  (*node_map_ptr)[node] = node_idx;
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_op_type("Gather");
  node_proto->add_output(std::to_string(node_idx));
  node_proto->add_input(name_x);
  node_proto->add_input(name_indices);
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto->set_i(static_cast<::google::protobuf::int64>(dyn_cast<Int64Imm>(axis)->value()));
}

void OnnxExporter::ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node,
                               std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  // Type of the 2nd input of 'Reshape' of MindSpore is tuple, but ONNX's is tensor, need to do some convert
  if (node->IsApply(prim::kPrimReshape)) {
    return ExportPrimReshape(func_graph, node, node_map_ptr, graph_proto);
  }

  if (node->IsApply(prim::kPrimReduceMean) || node->IsApply(prim::kPrimReduceSum)) {
    return ExportPrimReduce(func_graph, node, node_map_ptr, graph_proto);
  }

  // MindSpore Cast(x, T) --> ONNX Cast[to=T](x)
  if (node->IsApply(prim::kPrimCast)) {
    return ExportPrimCast(func_graph, node, node_map_ptr, graph_proto);
  }

  // ONNX PRelu requires unidirectional broadcasting, here need some process
  if (node->IsApply(std::make_shared<Primitive>("PReLU"))) {
    return ExportPrimPReLU(func_graph, node, node_map_ptr, graph_proto);
  }

  // MindSpore ReLU6(x) --> ONNX Clip[min=0.f, max=6.f](x)
  if (node->IsApply(std::make_shared<Primitive>("ReLU6"))) {
    return ExportPrimReLU6(func_graph, node, node_map_ptr, graph_proto);
  }

  // MindSpore DepthwiseConv2dNative --> ONNX Conv(x, reshape(w))
  if (node->IsApply(std::make_shared<Primitive>("DepthwiseConv2dNative"))) {
    return ExportPrimDepthwiseConv2d(func_graph, node, node_map_ptr, graph_proto);
  }

  // MindSpore Tile(x) --> ONNX Tile(x, repeat)
  if (node->IsApply(prim::kPrimTile)) {
    return ExportPrimTile(func_graph, node, node_map_ptr, graph_proto);
  }

  // MindSpore Square(x) --> ONNX Pow(x, 2)
  if (node->IsApply(prim::kPrimSquare)) {
    return ExportPrimSquare(func_graph, node, node_map_ptr, graph_proto);
  }

  // MindSpore GatherV2(x, indices, axis) --> ONNX Pow(x, indices)
  if (node->IsApply(prim::kPrimGather)) {
    return ExportPrimGatherV2(func_graph, node, node_map_ptr, graph_proto);
  }

  auto inputs = node->inputs();
  if (inputs.size() < 1) {
    MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
  }

  AnfNodePtr op = inputs[0];
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

  if (!IsPrimitiveEquals(prim, prim::kPrimMakeTuple)) {
    (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim, op_inputs, graph_proto);
  }
}

size_t OnnxExporter::ExportPrimitive(const FuncGraphPtr & /*func_graph*/, std::map<AnfNodePtr, size_t> *node_map_ptr,
                                     const PrimitivePtr &prim, const std::vector<AnfNodePtr> &inputs,
                                     onnx::GraphProto *const graph_proto) {
  auto op_map = OpConvertRegistry::GetOpConvertMap();
  auto op_iter = op_map.find(prim->name());
  if (op_iter == op_map.end()) {
    MS_LOG(EXCEPTION) << "Can not find key " << prim->name() << " in convert map. "
                      << "Exporting " << prim->name() << " operator is not yet supported.";
  }
  const OpNameInfo &op_convert_info = op_iter->second;
  auto node_idx = AllocateNodeIndex();
  onnx::NodeProto *node_proto = graph_proto->add_node();
  node_proto->add_output(std::to_string(node_idx));
  node_proto->set_op_type(op_convert_info.onnx_type());

  // Set inputs
  for (const auto &input : inputs) {
    auto input_name = GetNodeInputName(input, node_map_ptr, graph_proto);
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
  auto conv_node = dyn_cast<CNode>(node->input(1));
  auto input_x = conv_node->input(1);  // conv input x
  auto input_w = conv_node->input(2);  // conv weight(filter)
  auto input_b = node->input(2);       // conv bias

  PrimitivePtr prim_conv = dyn_cast<Primitive>((dyn_cast<ValueNode>(conv_node->input(0)))->value());
  std::vector<AnfNodePtr> inputs{input_x, input_w, input_b};
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_conv, inputs, graph_proto);
}

void OnnxExporter::ExportMergeGemm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                   std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  auto matmul_node = dyn_cast<CNode>(node->input(1));
  auto input_x = matmul_node->input(1);  // matmul input x
  auto input_y = matmul_node->input(2);  // matmul input y
  auto input_b = node->input(2);         // matmul bias

  PrimitivePtr prim_matmul = dyn_cast<Primitive>((dyn_cast<ValueNode>(matmul_node->input(0)))->value());
  std::vector<AnfNodePtr> inputs{input_x, input_y, input_b};
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_matmul, inputs, graph_proto);
}

void OnnxExporter::ExportMergeBatchNorm(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                        std::map<AnfNodePtr, size_t> *node_map_ptr,
                                        onnx::GraphProto *const graph_proto) {
  auto batch_norm_node = dyn_cast<CNode>(node->input(1));

  PrimitivePtr prim_batch_norm = dyn_cast<Primitive>((dyn_cast<ValueNode>(batch_norm_node->input(0)))->value());
  std::vector<AnfNodePtr> inputs;
  for (size_t i = 1; i < batch_norm_node->inputs().size(); i++) {
    inputs.push_back(batch_norm_node->input(i));
  }
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_batch_norm, inputs, graph_proto);
}

void OnnxExporter::ExportMergeMaxPoolWithArgmax(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                                std::map<AnfNodePtr, size_t> *node_map_ptr,
                                                onnx::GraphProto *const graph_proto) {
  auto maxpool_with_argmax_node = dyn_cast<CNode>(node->input(1));

  PrimitivePtr prim_maxpool_with_argmax =
    dyn_cast<Primitive>((dyn_cast<ValueNode>(maxpool_with_argmax_node->input(0)))->value());
  std::vector<AnfNodePtr> inputs;
  for (size_t i = 1; i < maxpool_with_argmax_node->inputs().size(); i++) {
    inputs.push_back(maxpool_with_argmax_node->input(i));
  }
  (*node_map_ptr)[node] = ExportPrimitive(func_graph, node_map_ptr, prim_maxpool_with_argmax, inputs, graph_proto);
}

void OnnxExporter::ExportOutput(const FuncGraphPtr & /*func_graph*/, const CNodePtr &node,
                                std::map<AnfNodePtr, size_t> *node_map_ptr, onnx::GraphProto *const graph_proto) {
  if (node->inputs().size() != 2) {
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

  if (node->isa<Parameter>()) {
    return node->ToString();
  }

  // for ValueNode input, create a Constant Operator
  if (node->isa<ValueNode>()) {
    auto iter = node_map_ptr->find(node);
    if (iter != node_map_ptr->end()) {
      return std::to_string(iter->second);
    }
    // the id number starts at 1, so the id of created node should be size of map plus one
    auto node_idx = AllocateNodeIndex();
    (*node_map_ptr)[node] = node_idx;
    std::string node_name = std::to_string(node_idx);

    onnx::NodeProto *node_proto = graph_proto->add_node();
    node_proto->add_output(node_name);

    SetNodeAttribute(node->cast<ValueNodePtr>()->value(), node_proto);

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
  auto type_id = (*tuple_ptr)[0]->type()->type_id();
  for (size_t i = 1; i < tuple_ptr->size(); ++i) {
    if ((*tuple_ptr)[i]->type()->type_id() != type_id) {
      MS_LOG(EXCEPTION) << "Convert tuple to tensor fail, type of tuple elements is not same.";
    }
  }

  tensor_proto->add_dims(static_cast<::google::protobuf::int64>(tuple_ptr->size()));
  tensor_proto->set_data_type(onnx::TensorProto_DataType_INT64);
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
    } else {
      MS_LOG(EXCEPTION) << "Convert tuple to tensor fail, unexpected tuple element type " << elem->type()->type_name()
                        << ".";
    }
  }
}

void OnnxExporter::SetNodeAttribute(const ValuePtr &value, onnx::NodeProto *const node_proto) {
  node_proto->set_op_type("Constant");
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_name("value");
  if (value->isa<Int32Imm>()) {
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
    auto casted_value = dyn_cast<Int32Imm>(value);
    if (casted_value == nullptr) {
      MS_LOG(EXCEPTION) << "Cast value " << value->ToString() << " to type T failed.";
    }
    auto attr_value = casted_value->value();
    attr_proto->set_i(static_cast<::google::protobuf::int64>(attr_value));
    attr_proto->set_type(onnx::AttributeProto_AttributeType_INT);
  } else if (value->isa<tensor::Tensor>()) {
    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
    auto data = dyn_cast<tensor::Tensor>(value);
    tensor_proto->set_raw_data(data->data_c(), static_cast<size_t>(data->data().nbytes()));
    auto dtype = data->data_type();
    auto shape = data->shape_c();

    tensor_proto->set_data_type(GetOnnxDataType(dtype));
    for (const auto &dim : shape) {
      tensor_proto->add_dims(dim);
    }
  } else {
    MS_LOG(EXCEPTION) << "Need to set value " << value->ToString() << " attribute for Constant node";
  }
}

std::string GetOnnxProtoString(const FuncGraphPtr &func_graph) {
  OnnxExporter exporter;
  return exporter.GetOnnxProtoString(func_graph);
}
}  // namespace mindspore
