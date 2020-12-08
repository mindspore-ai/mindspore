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

#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include <algorithm>
#include <cfloat>
#include <unordered_map>
#include <utility>
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"

namespace mindspore {
namespace lite {
OnnxModelParser::OnnxModelParser() = default;
OnnxModelParser::~OnnxModelParser() = default;

static const std::unordered_map<int, mindspore::TypeId> TYPE_MAP = {
  {onnx::TensorProto_DataType_INT8, mindspore::kNumberTypeInt8},
  {onnx::TensorProto_DataType_UINT8, mindspore::kNumberTypeUInt8},
  {onnx::TensorProto_DataType_INT16, mindspore::kNumberTypeInt16},
  {onnx::TensorProto_DataType_INT32, mindspore::kNumberTypeInt32},
  {onnx::TensorProto_DataType_UINT32, mindspore::kNumberTypeUInt32},
  {onnx::TensorProto_DataType_INT64, mindspore::kNumberTypeInt64},
  {onnx::TensorProto_DataType_FLOAT16, mindspore::kNumberTypeFloat16},
  {onnx::TensorProto_DataType_FLOAT, mindspore::kNumberTypeFloat32},
  {onnx::TensorProto_DataType_BOOL, mindspore::kNumberTypeBool}};

TypeId OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type) {
  auto iter = TYPE_MAP.find(onnx_type);
  if (iter == TYPE_MAP.end()) {
    MS_LOG(ERROR) << "unsupported onnx data type: " << onnx_type;
    return kTypeUnknown;
  }
  return iter->second;
}

std::vector<int32_t> OnnxModelParser::GetDimsFromOnnxValue(const onnx::ValueInfoProto &onnx_value) {
  std::vector<int32_t> dims;
  for (const auto &it : onnx_value.type().tensor_type().shape().dim()) {
    dims.emplace_back(it.dim_value());
  }
  return dims;
}

STATUS OnnxModelParser::SetGraphConstTensor(const onnx::GraphProto &onnx_graph) {
  MS_LOG(DEBUG) << "set onnx constant tensors";
  for (const auto &onnx_const_value : onnx_graph.initializer()) {
    int index;
    const auto status = AddTensorProto(onnx_const_value, onnx_const_value.name(), GRAPH_INPUT, &index);
    if (status != RET_OK) {
      return status;
    }
    MS_LOG(DEBUG) << "add const tensor: " << onnx_const_value.name() << ", index " << index;
  }
  MS_LOG(DEBUG) << "process onnx Constant ops";
  for (int i = 0; i < onnx_graph.node_size(); i++) {
    const auto &node = onnx_graph.node(i);
    if (node.op_type().compare("Constant") == 0) {
      for (const auto &attr : node.attribute()) {
        if (attr.name() == "sparse_value") {
          MS_LOG(ERROR) << "sparse_value";
        }
        if (attr.name() == "value") {
          const auto &t = attr.t();
          int index;
          const auto status = AddTensorProto(t, node.output(0), GRAPH_INPUT, &index);
          if (status != RET_OK) {
            return status;
          }
          MS_LOG(DEBUG) << "add const tensor: " << t.name() << ", index " << index;
        } else {
          MS_LOG(ERROR) << "processing Constant op attr " << attr.name() << " not implemented";
          return RET_INVALID_OP_ATTR;
        }
      }
    }
  }
  return RET_OK;
}

STATUS OnnxModelParser::AddValueInfo(const onnx::ValueInfoProto &proto, const std::string &name, const Category &type,
                                     int *index) {
  auto data_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(proto.type().tensor_type().elem_type()));
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "not support onnx data type "
                  << static_cast<onnx::TensorProto_DataType>(proto.type().tensor_type().elem_type());
    return RET_ERROR;
  }
  std::unique_ptr<schema::TensorT> tensor = std::make_unique<schema::TensorT>();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new tensor failed";
    return RET_ERROR;
  }
  tensor->dataType = data_type == kNumberTypeInt64 ? kNumberTypeInt32 : data_type;
  tensor->dims = GetDimsFromOnnxValue(proto);
  tensor->format = schema::Format::Format_NCHW;
  tensor->nodeType = schema::NodeType::NodeType_ValueNode;
  *index = OnnxTensorParser::GetInstance()->GetTensorCache()->AddTensor(name, tensor.release(), type);
  return RET_OK;
}

STATUS OnnxModelParser::AddTensorProto(const onnx::TensorProto &proto, const std::string &name, const Category &type,
                                       int *index) {
  auto data_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(proto.data_type()));
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "not support onnx data type " << static_cast<onnx::TensorProto_DataType>(proto.data_type());
    return RET_ERROR;
  }

  std::unique_ptr<schema::TensorT> tensor = std::make_unique<schema::TensorT>();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new tensor failed";
    return RET_ERROR;
  }
  tensor->dataType = data_type;
  std::copy(proto.dims().begin(), proto.dims().end(), std::back_inserter(tensor->dims));
  tensor->format = schema::Format::Format_NCHW;
  tensor->nodeType = schema::NodeType::NodeType_ValueNode;
  if (CopyOnnxTensorData(proto, tensor.get())) {
    MS_LOG(ERROR) << "copy onnx data failed";
    return RET_ERROR;
  }
  if (data_type == kNumberTypeInt64) {
    tensor->dataType = kNumberTypeInt32;  // CopyOnnxTensorData will convert int64 to int32
  }
  *index = OnnxTensorParser::GetInstance()->GetTensorCache()->AddTensor(name, tensor.release(), type);
  return RET_OK;
}

STATUS OnnxModelParser::SetGraphInputTensor(const onnx::GraphProto &onnx_graph, schema::SubGraphT *graph) {
  for (const auto &input_value : onnx_graph.input()) {
    auto ret = OnnxTensorParser::GetInstance()->GetTensorCache()->FindTensor(input_value.name());
    if (ret < 0) {
      int index;
      const auto status = AddValueInfo(input_value, input_value.name(), GRAPH_INPUT, &index);
      if (status != RET_OK) {
        return status;
      }
      MS_LOG(DEBUG) << "input_value name: " << input_value.name() << ", graph input index: " << index;
      graph->inputIndices.emplace_back(static_cast<uint32_t>(index));
    }
  }
  return RET_OK;
}

STATUS OnnxModelParser::SetGraphOutputTensor(const onnx::GraphProto &onnx_graph, schema::SubGraphT *graph) {
  for (const auto &output_value : onnx_graph.output()) {
    int index;
    if (OnnxTensorParser::GetInstance()->GetTensorCache()->FindTensor(output_value.name()) != -1) {
      index = OnnxTensorParser::GetInstance()->GetTensorCache()->FindTensor(output_value.name());
    } else {
      const auto status = AddValueInfo(output_value, output_value.name(), OP_OUTPUT, &index);
      if (status != RET_OK) {
        return status;
      }
    }
    graph->outputIndices.emplace_back(index);
    MS_LOG(DEBUG) << "output_value name: " << output_value.name() << ", graph output index: " << index;
  }
  return RET_OK;
}

void OnnxModelParser::ParseOnnxGemmNode(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                        schema::SubGraphT *sub_graph, schema::MetaGraphT *graph,
                                        const QuantType &quant_type) {
  std::unique_ptr<schema::CNodeT> dst_op_1 = std::make_unique<schema::CNodeT>();
  dst_op_1->name = "Gemm_MatMul_" + onnx_node.output(0);
  dst_op_1->quantType = quant_type;
  ParseOnnxNodeAttr(onnx_graph, onnx_node, "MatMul", dst_op_1.get());
  auto matmul_output_id = "Gemm_MatMul_" + onnx_node.output(0);
  std::vector<string> matmul_inputs{onnx_node.input(0), onnx_node.input(1)};
  std::vector<string> matmul_outputs{matmul_output_id};
  SetOpInputIndex(matmul_inputs, dst_op_1.get(), onnx_node);
  SetOpOutputIndex(matmul_outputs, dst_op_1.get());
  graph->nodes.emplace_back(std::move(dst_op_1));
  sub_graph->nodeIndices.push_back(graph->nodes.size() - 1);

  std::unique_ptr<schema::CNodeT> dst_op_2 = std::make_unique<schema::CNodeT>();
  dst_op_2->name = "Gemm_BiasAdd_" + onnx_node.output(0);
  dst_op_2->quantType = quant_type;
  ParseOnnxNodeAttr(onnx_graph, onnx_node, "BiasAdd", dst_op_2.get());
  std::vector<string> biasadd_inputs{matmul_output_id, onnx_node.input(2)};
  std::vector<string> biasadd_outputs{onnx_node.output(0)};
  SetOpInputIndex(biasadd_inputs, dst_op_2.get(), onnx_node);
  SetOpOutputIndex(biasadd_outputs, dst_op_2.get());
  graph->nodes.emplace_back(std::move(dst_op_2));
  sub_graph->nodeIndices.push_back(graph->nodes.size() - 1);
}

STATUS OnnxModelParser::ParseOnnxGivenFillNode(const onnx::NodeProto &onnx_node) {
  // convert GivenTensorFill node to a weight/bias tensor
  auto ret = OnnxTensorParser::GetInstance()->GetTensorCache()->FindTensor(onnx_node.output(0));
  if (ret < 0) {
    std::unique_ptr<schema::TensorT> tensor = std::make_unique<schema::TensorT>();
    std::vector<int> shape;
    auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                             [](const onnx::AttributeProto &attr) { return attr.name() == "shape"; });
    if (iter != onnx_node.attribute().end()) {
      (void)shape.insert(shape.begin(), iter->ints().begin(), iter->ints().end());
      std::for_each(shape.begin(), shape.end(), [](int sh) { MS_LOG(DEBUG) << "shape: " << sh; });
    }
    tensor->dims = shape;
    tensor->format = schema::Format::Format_NUM_OF_FORMAT;
    tensor->nodeType = schema::NodeType::NodeType_ValueNode;
    iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                        [](const onnx::AttributeProto &attr) { return attr.name() == "values"; });
    // copy GivenIntTensorFill node value to tensor
    if (iter != onnx_node.attribute().end()) {
      size_t data_count = 1;
      std::for_each(shape.begin(), shape.end(), [&data_count](int dim) { data_count *= dim; });
      size_t data_size = 0;
      if (onnx_node.op_type() == "Int8GivenIntTensorFill") {
        tensor->dataType = kNumberTypeInt32;
        data_size = data_count * sizeof(int32_t) / sizeof(uint8_t);
        tensor->data.resize(data_size);
        void *tensorData = tensor->data.data();
        auto castedTensorData = static_cast<int32_t *>(tensorData);
        MS_ASSERT(castedTensorData != nullptr);
        for (size_t i = 0; i < data_count; i++) {
          castedTensorData[i] = int32_t(iter->ints().data()[i]);
        }
      } else if (onnx_node.op_type() == "Int8GivenTensorFill") {
        tensor->dataType = kNumberTypeUInt8;
        data_size = data_count;
        tensor->data.resize(data_size);
        MS_LOG(DEBUG) << "tensor data size " << data_size << ", s: " << sizeof(iter->s().data());
        if (memcpy_s(tensor->data.data(), data_size, iter->s().data(), data_size) != 0) {
          MS_LOG(ERROR) << "memcpy_s failed";
          return RET_ERROR;
        }
        // set quantParams to Int8GivenTensor.
        std::unique_ptr<schema::QuantParamT> quant_param = std::make_unique<schema::QuantParamT>();
        for (const auto &onnx_node_attr : onnx_node.attribute()) {
          if (onnx_node_attr.name() == "Y_scale") {
            quant_param->scale = onnx_node_attr.f();
          } else if (onnx_node_attr.name() == "Y_zero_point") {
            quant_param->zeroPoint = static_cast<int32_t>(onnx_node_attr.i());
          }
        }
        quant_param->inited = true;
        tensor->quantParams.emplace_back(std::move(quant_param));
      } else {
        MS_LOG(ERROR) << "unsupported data type " << tensor->dataType;
        return RET_ERROR;
      }
    }
    auto index =
      OnnxTensorParser::GetInstance()->GetTensorCache()->AddTensor(onnx_node.output(0), tensor.release(), GRAPH_INPUT);
    MS_LOG(DEBUG) << "add given tensor: " << index;
  }
  return RET_OK;
}

STATUS OnnxModelParser::ParseOnnxNodeToDstOp(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                             schema::CNodeT *dst_op, const QuantType &quantType,
                                             schema::MetaGraphT *dst_graph) {
  // change op_type() to name(), that is unique
  static bool interrupt = false;
  dst_op->name = onnx_node.op_type() + "_" + onnx_node.output(0);
  dst_op->quantType = quantType;
  // dst_op->fmkType = FmkType_ONNX;
  MS_LOG(DEBUG) << "onnx op name " << onnx_node.op_type() << ", dst op name: " << dst_op->name << ", input size "
                << onnx_node.input_size();
  // get the real op type
  if (onnx_node.op_type() == "Loop") {
    NoSupportOp::GetInstance()->InsertOp(onnx_node.op_type());
    interrupt = true;
    return RET_NOT_FIND_OP;
    int status = ParseSubgraph(dst_op, onnx_node, quantType, dst_graph);
    if (status != RET_OK || interrupt) {
      interrupt = true;
      return status;
    }
  } else {
    auto node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser(onnx_node.op_type());
    if (node_parser == nullptr || interrupt) {
      interrupt = true;
      if (node_parser == nullptr) {
        NoSupportOp::GetInstance()->InsertOp(onnx_node.op_type());
      }
      return RET_NOT_FIND_OP;
    }
    auto status = node_parser->Parse(onnx_graph, onnx_node, dst_op);
    if (status != RET_OK) {
      interrupt = true;
      if (status == RET_NOT_FIND_OP) {
        NoSupportOp::GetInstance()->InsertOp(onnx_node.op_type());
      } else {
        MS_LOG(ERROR) << "parser onnx node " << onnx_node.op_type() << " attr failed";
      }
      return status;
    }
  }
  // set op input index
  std::vector<string> node_inputs;
  (void)node_inputs.insert(node_inputs.begin(), onnx_node.input().begin(), onnx_node.input().end());
  if (SetOpInputIndex(node_inputs, dst_op, onnx_node)) {
    interrupt = true;
    MS_LOG(ERROR) << "SetOpInputIndex failed";
    return RET_ERROR;
  }
  if (dst_op->primitive->value.type == schema::PrimitiveType_Conv2D) {
    auto &weight_tensor =
      OnnxTensorParser::GetInstance()->GetTensorCache()->GetCachedTensor().at(dst_op->inputIndex.at(kWeightIndex));
    weight_tensor->format = dst_op->primitive->value.AsConv2D()->format;
  } else if (dst_op->primitive->value.type == schema::PrimitiveType_DeConv2D) {
    auto &weight_tensor =
      OnnxTensorParser::GetInstance()->GetTensorCache()->GetCachedTensor().at(dst_op->inputIndex.at(kWeightIndex));
    weight_tensor->format = dst_op->primitive->value.AsDeConv2D()->format;
  }
  // set op output index
  std::vector<string> node_outputs;
  (void)node_outputs.insert(node_outputs.begin(), onnx_node.output().begin(), onnx_node.output().end());

  if (SetOpOutputIndex(node_outputs, dst_op) != RET_OK) {
    interrupt = true;
    MS_LOG(ERROR) << "SetOpOutputIndex failed";
    return RET_ERROR;
  }
  auto &output_tensor =
    OnnxTensorParser::GetInstance()->GetTensorCache()->GetCachedTensor().at(dst_op->outputIndex.front());
  if (output_tensor == nullptr) {
    interrupt = true;
    MS_LOG(ERROR) << "Output tensor of node " << onnx_node.op_type() << "is nullptr.";
    return RET_ERROR;
  }
  SetOpQuantParams(onnx_graph, onnx_node, dst_op, output_tensor);
  return RET_OK;
}

void OnnxModelParser::SetOpQuantParams(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                       schema::CNodeT *dst_op, schema::TensorT *dst_tensor) {
  MS_ASSERT(dst_op != nullptr);
  std::vector<string> quant_node_name;
  quant_node_name.insert(quant_node_name.begin(), onnx_node.input().begin(), onnx_node.input().end());
  quant_node_name.insert(quant_node_name.end(), onnx_node.output().begin(), onnx_node.output().end());
  std::vector<onnx::NodeProto> quant_node;
  for (const auto &str : quant_node_name) {
    for (auto &node : onnx_graph.node()) {
      if (node.output(0) == str) {
        quant_node.emplace_back(node);
        break;
      }
    }
  }
  auto needQuantParams = size_t(onnx_node.input().size() + onnx_node.output().size());
  for (auto iter = onnx_node.input().begin(); iter != onnx_node.input().end(); iter++) {
    if (IsContain(this->graphInputNames, *iter)) {
      needQuantParams--;
    }
  }
  size_t findQuantParams = 0;
  for (const auto &node : quant_node) {
    std::unique_ptr<schema::QuantParamT> quant_param = std::make_unique<schema::QuantParamT>();
    if (quant_param == nullptr) {
      MS_LOG(ERROR) << "new QuantParamT failed, node: " << dst_op->name;
      return;
    }
    int argNum = 0;
    for (const auto &onnx_node_attr : node.attribute()) {
      if (onnx_node_attr.name() == "Y_scale") {
        quant_param->scale = onnx_node_attr.f();
        argNum++;
      } else if (onnx_node_attr.name() == "Y_zero_point") {
        quant_param->zeroPoint = static_cast<int32_t>(onnx_node_attr.i());
        argNum++;
      }
    }
    if (argNum != 2) {
      continue;
    }
    dst_tensor->quantParams.emplace_back(std::move(quant_param));
    if (argNum == 2) {
      findQuantParams++;
    }
  }
  if (findQuantParams == needQuantParams) {
    dst_op->quantType = schema::QuantType_AwareTraining;
  }
}

STATUS OnnxModelParser::ParseOnnxNodeAttr(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                          const string &onnx_op_type, schema::CNodeT *dst_op) {
  auto node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser(onnx_op_type);
  if (node_parser == nullptr) {
    return RET_NOT_FIND_OP;
  }
  return node_parser->Parse(onnx_graph, onnx_node, dst_op);
}

STATUS OnnxModelParser::SetOpInputIndex(const std::vector<string> &node_inputs, schema::CNodeT *dst_op,
                                        const onnx::NodeProto &onnx_node) {
  for (const auto &onnx_node_input : node_inputs) {
    if (onnx_node_input != "") {
      int index = OnnxTensorParser::GetInstance()->GetTensorCache()->FindTensor(onnx_node_input);
      if (index < 0) {
        MS_LOG(ERROR) << "input " << onnx_node_input << " of node " << onnx_node.name() << " can't be found";
        return RET_ERROR;
      }
      MS_LOG(DEBUG) << "node: " << onnx_node_input << ", input index: " << index;
      dst_op->inputIndex.emplace_back(index);
    }
  }
  return RET_OK;
}

STATUS OnnxModelParser::SetOpOutputIndex(const std::vector<string> &node_outputs, schema::CNodeT *dst_op) {
  for (const auto &onnx_node_output : node_outputs) {
    auto index = OnnxTensorParser::GetInstance()->GetTensorCache()->FindTensor(onnx_node_output);
    if (index < 0) {  // when index >= 0, it's graph's output
      std::unique_ptr<schema::TensorT> tensor = std::make_unique<schema::TensorT>();
      tensor->nodeType = schema::NodeType_Parameter;
      index =
        OnnxTensorParser::GetInstance()->GetTensorCache()->AddTensor(onnx_node_output, tensor.release(), OP_OUTPUT);
    }
    MS_LOG(DEBUG) << "node: " << onnx_node_output << ", output index: " << index;
    dst_op->outputIndex.emplace_back(index);
  }
  return RET_OK;
}

STATUS OnnxModelParser::CopyOnnxTensorData(const onnx::TensorProto &onnx_const_value, schema::TensorT *tensor) {
  size_t data_count = 1;
  std::for_each(tensor->dims.begin(), tensor->dims.end(), [&data_count](int dim) { data_count *= dim; });
  size_t data_size = 0;
  const void *tensor_data = nullptr;
  std::unique_ptr<int32_t[]> buffer;
  switch (tensor->dataType) {
    case kNumberTypeFloat32:
      data_size = data_count * sizeof(float);
      if (onnx_const_value.float_data_size() == 0) {
        tensor_data = onnx_const_value.raw_data().data();
      } else {
        tensor_data = onnx_const_value.float_data().data();
      }
      break;
    case kNumberTypeInt32:
      data_size = data_count * sizeof(int);
      if (onnx_const_value.int32_data_size() == 0) {
        tensor_data = onnx_const_value.raw_data().data();
      } else {
        tensor_data = onnx_const_value.int32_data().data();
      }
      break;
    case kNumberTypeInt64:
      data_size = data_count * sizeof(int32_t);
      buffer = std::make_unique<int32_t[]>(data_count);
      const int64_t *in_data;
      in_data = nullptr;
      if (onnx_const_value.int64_data_size() == 0) {
        in_data = reinterpret_cast<const int64_t *>(onnx_const_value.raw_data().data());
      } else {
        in_data = onnx_const_value.int64_data().data();
      }
      for (size_t i = 0; i < data_count; ++i) {
        if (in_data[i] > static_cast<int64_t>(INT32_MAX) || in_data[i] < static_cast<int64_t>(INT32_MIN)) {
          MS_LOG(WARNING) << "int64 data " << in_data[i] << "too big to fit into int32";
          buffer[i] = in_data[i] > 0 ? INT32_MAX : INT32_MIN;
        } else {
          buffer[i] = static_cast<int>(in_data[i]);
        }
      }
      tensor_data = reinterpret_cast<void *>(buffer.get());
      break;
    case kNumberTypeUInt8:
    case kNumberTypeInt8:
    case kNumberTypeBool:
      data_size = data_count * sizeof(uint8_t);
      tensor_data = onnx_const_value.raw_data().data();
      break;
    default:
      MS_LOG(ERROR) << "unsupported data type " << tensor->dataType;
      return RET_ERROR;
  }
  tensor->data.resize(data_size);
  if (data_size != 0 && memcpy_s(static_cast<void *>(tensor->data.data()), data_size, tensor_data, data_size) != 0) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS OnnxModelParser::SetAllTensors(schema::MetaGraphT *graphDef) {
  std::vector<schema::TensorT *> tensors = OnnxTensorParser::GetInstance()->GetTensorCache()->GetCachedTensor();
  for (auto iter : tensors) {
    std::unique_ptr<schema::TensorT> temp(iter);
    graphDef->allTensors.emplace_back(move(temp));
  }
  return RET_OK;
}

void OnnxModelParser::FindGraphInputAndConst(const onnx::GraphProto &onnx_graph) {
  this->graphInputNames.clear();
  this->graphConstNames.clear();
  for (auto &onnx_const : onnx_graph.initializer()) {
    this->graphConstNames.emplace_back(onnx_const.name());
  }
  for (auto &onnx_input : onnx_graph.input()) {
    if (!IsContain(this->graphConstNames, onnx_input.name())) {
      this->graphInputNames.emplace_back(onnx_input.name());
    }
  }
}

STATUS OnnxModelParser::ParseSubgraph(schema::CNodeT *dst_op, const onnx::NodeProto &onnx_node,
                                      const QuantType &quantType, schema::MetaGraphT *dst_graph) {
  MS_LOG(DEBUG) << "onnx LoopParser";
  if (dst_op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  dst_op->primitive = std::make_unique<schema::PrimitiveT>();
  if (dst_op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }
  std::unique_ptr<schema::LoopT> attr = std::make_unique<schema::LoopT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  attr->subGraphIndex = subGraphNum;
  auto sub_graph = std::make_unique<schema::SubGraphT>();
  int ret = ParseGraph(dst_graph, sub_graph.get(), onnx_node.attribute().at(0).g(), quantType);
  dst_graph->subGraph.push_back(std::move(sub_graph));
  subGraphNum += 1;
  if (ret != RET_OK) {
    return ret;
  }
  dst_op->primitive->value.type = schema::PrimitiveType_Loop;
  dst_op->primitive->value.value = attr.release();
  return RET_OK;
}

int OnnxModelParser::ParseGraph(schema::MetaGraphT *dst_graph, schema::SubGraphT *dst_sub_graph,
                                const onnx::GraphProto &onnx_graph, const QuantType &quantType) {
  // dst_graph->name = onnx_graph.name();  // this is not used
  // find out input names and const names
  FindGraphInputAndConst(onnx_graph);
  // set const tensor
  int status = SetGraphConstTensor(onnx_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetGraphConstTensor failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }

  // init onnx model graph input tensor

  status = SetGraphInputTensor(onnx_graph, dst_sub_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetGraphInputTensor failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }

  // init op node input/output tensor, and dst_op attr
  NoSupportOp::GetInstance()->SetFmkType("ONNX");
  for (const auto &onnx_node : onnx_graph.node()) {
    int status_node = RET_OK;
    if (onnx_node.op_type() == "Constant") {
      continue;
    }
    if (onnx_node.op_type() == "Gemm") {
      if (status == RET_OK) {
        ParseOnnxGemmNode(onnx_graph, onnx_node, dst_sub_graph, dst_graph, quantType);
      }
      continue;
    } else if (onnx_node.op_type() == "Int8GivenIntTensorFill" || onnx_node.op_type() == "Int8GivenTensorFill") {
      if (status == RET_OK) {
        status_node = ParseOnnxGivenFillNode(onnx_node);
        if (status_node != RET_OK) {
          MS_LOG(ERROR) << "ParseOnnxGivenFillNode failed: " << status_node;
          status = (status == RET_OK ? status_node : status);
        }
      }
      continue;
    }

    std::unique_ptr<schema::CNodeT> dst_op = std::make_unique<schema::CNodeT>();
    status_node = ParseOnnxNodeToDstOp(onnx_graph, onnx_node, dst_op.get(), quantType, dst_graph);
    if (status_node != RET_OK) {
      status = (status == RET_OK ? status_node : status);
      continue;
    }
    dst_graph->nodes.emplace_back(std::move(dst_op));
    dst_sub_graph->nodeIndices.push_back((dst_graph->nodes.size() - 1));
  }
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    for (auto &tensor : OnnxTensorParser::GetInstance()->GetTensorCache()->GetCachedTensor()) {
      delete tensor;
    }
    return RET_ERROR;
  }
  // init onnx model graph output tensor
  status = SetGraphOutputTensor(onnx_graph, dst_sub_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetGraphOutputTensor failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }
  SetAllTensors(dst_graph);
  return RET_OK;
}

schema::MetaGraphT *OnnxModelParser::ParseToFb(const std::string &model_file, const std::string &weight_file,
                                               const QuantType &quant_type) {
  int status = ValidateFileStr(model_file, ".onnx");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.onnx";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  onnx::ModelProto onnx_model;
  status = ReadProtoFromBinaryFile((const char *)model_file.c_str(), &onnx_model);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read onnx model file failed, model path: " << model_file;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  OnnxNodeParser::set_opset_version(onnx_model.opset_import().Get(0).version());
  const onnx::GraphProto &onnx_graph = onnx_model.graph();
  MS_LOG(INFO) << "model producer name: " << onnx_model.producer_name() << ", graph name: " << onnx_graph.name();

  auto dst_graph = std::make_unique<schema::MetaGraphT>();
  auto dst_sub_graph = std::make_unique<schema::SubGraphT>();
  int ret = ParseGraph(dst_graph.get(), dst_sub_graph.get(), onnx_graph, quant_type);
  dst_graph->subGraph.push_back(std::move(dst_sub_graph));
  subGraphNum += 1;
  if (ret == RET_ERROR) {
    return nullptr;
  }
  dst_graph->name = GetModelName(model_file);

  std::vector<uint32_t> input_temp_index;
  for (size_t i = 0; i < dst_graph->subGraph.front()->inputIndices.size(); i++) {
    input_temp_index.push_back(dst_graph->subGraph.front()->inputIndices[i]);
  }
  dst_graph->inputIndex = input_temp_index;

  std::vector<uint32_t> output_temp_index;
  for (size_t i = 0; i < dst_graph->subGraph.front()->outputIndices.size(); i++) {
    output_temp_index.push_back(dst_graph->subGraph.front()->outputIndices[i]);
  }
  dst_graph->outputIndex = output_temp_index;

  return dst_graph.release();
}

}  // namespace lite
}  // namespace mindspore
