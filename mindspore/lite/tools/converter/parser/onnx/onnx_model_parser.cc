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

#include <cfloat>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "tools/common/graph_util.h"
#include "src/common/utils.h"

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
  {onnx::TensorProto_DataType_FLOAT, mindspore::kNumberTypeFloat32}};

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

STATUS OnnxModelParser::ReadOnnxModelFromBinary(const std::string &modelFile, google::protobuf::Message *onnx_model) {
  std::unique_ptr<char> onnx_file(new (std::nothrow) char[PATH_MAX]{0});
  if (realpath(modelFile.c_str(), onnx_file.get()) == nullptr) {
    MS_LOG(ERROR) << "get realpath " << modelFile << " fail";
    return RET_ERROR;
  }
  int fd = open(onnx_file.get(), O_RDONLY);
  google::protobuf::io::FileInputStream input(fd);
  google::protobuf::io::CodedInputStream code_input(&input);
  code_input.SetTotalBytesLimit(INT_MAX, 536870912);
  bool ret = onnx_model->ParseFromCodedStream(&code_input);
  if (!ret) {
    MS_LOG(ERROR) << "load onnx file failed";
    return RET_ERROR;
  }
  (void)close(fd);
  return RET_OK;
}

STATUS OnnxModelParser::SetGraphConstTensor(const onnx::GraphProto &onnx_graph, TensorCache *tensor_cache) {
  MS_LOG(DEBUG) << "set onnx constant tensors";
  for (const auto &onnx_const_value : onnx_graph.initializer()) {
    auto data_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(onnx_const_value.data_type()));
    if (data_type == kTypeUnknown) {
      MS_LOG(ERROR) << "not support onnx data type "
                    << static_cast<onnx::TensorProto_DataType>(onnx_const_value.data_type());
      return RET_ERROR;
    }
    std::unique_ptr<schema::TensorT> tensor(new (std::nothrow) schema::TensorT);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new tensor failed";
      return RET_ERROR;
    }
    tensor->dataType = data_type;
    tensor->format = schema::Format_NCHW;  // onnx use NCHW
    std::copy(onnx_const_value.dims().begin(), onnx_const_value.dims().end(), std::back_inserter(tensor->dims));
    tensor->nodeType = schema::NodeType_ValueNode;
    if (CopyOnnxTensorData(onnx_const_value, tensor.get())) {
      MS_LOG(ERROR) << "copy onnx data failed";
      return RET_ERROR;
    }
    // TODO(wangzhe) why use GRAPH_INPUT other than CONST(GRAPH_INPUT will add index to graphInputs)
    const auto index = tensor_cache->AddTensor(onnx_const_value.name(), tensor.release(), GRAPH_INPUT);
    MS_LOG(DEBUG) << "add const tensor: " << onnx_const_value.name() << ", index " << index;
  }
  return RET_OK;
}

// TODO(wangzhe) seems AddTensorCache should be renamed to prepare tensor to add to tensor_cache
STATUS OnnxModelParser::AddTensorCache(const onnx::ValueInfoProto &proto, schema::TensorT *tensor) {
  auto data_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(proto.type().tensor_type().elem_type()));
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "not support onnx type "
                  << static_cast<onnx::TensorProto_DataType>(proto.type().tensor_type().elem_type());
    return RET_ERROR;
  }
  tensor->dataType = data_type;
  tensor->dims = GetDimsFromOnnxValue(proto);
  tensor->format = schema::Format_NCHW;
  tensor->nodeType = schema::NodeType_ValueNode;
  // TODO(wangzhe) tensor->data and quantParams not set, should we need tensor_cache->AddTensor?
  return RET_OK;
}

STATUS OnnxModelParser::SetGraphInputTensor(const onnx::GraphProto &onnx_graph, schema::MetaGraphT *graph,
                                            TensorCache *tensor_cache) {
  for (const auto &input_value : onnx_graph.input()) {
    auto ret = tensor_cache->FindTensor(input_value.name());
    if (ret < 0) {
      std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
      // TODO(wangzhe) why there is an addtensorCache?
      if (AddTensorCache(input_value, tensor.get())) {
        return RET_ERROR;
      }
      // TODO(wangzhe) why inputTensor is value and should be added into tensor_cache?
      auto tensor_index = tensor_cache->AddTensor(input_value.name(), tensor.release(), GRAPH_INPUT);
      graph->inputIndex.emplace_back(static_cast<uint32_t>(tensor_index));
      MS_LOG(DEBUG) << "input_value name: " << input_value.name() << ", graph input index: " << tensor_index;
    }
  }
  return RET_OK;
}

STATUS OnnxModelParser::SetGraphOutputTensor(const onnx::GraphProto &onnx_graph, schema::MetaGraphT *graph,
                                             TensorCache *tensor_cache) {
  for (const auto &output_value : onnx_graph.output()) {
    std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
    if (AddTensorCache(output_value, tensor.get())) {
      return RET_ERROR;
    }
    // TODO(wangzhe) why we need AddTensor at OutputTensor
    auto tensor_index = tensor_cache->AddTensor(output_value.name(), tensor.release(), OP_OUTPUT);
    graph->outputIndex.emplace_back(tensor_index);
    MS_LOG(DEBUG) << "output_value name: " << output_value.name() << ", graph output index: " << tensor_index;
  }
  return RET_OK;
}

void OnnxModelParser::ParseOnnxGemmNode(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                        schema::MetaGraphT *graph, TensorCache *tensor_cache) {
  std::unique_ptr<schema::CNodeT> dst_op_1(new schema::CNodeT);
  dst_op_1->name = "Gemm_MatMul_" + onnx_node.output(0);
  ParseOnnxNodeAttr(onnx_graph, onnx_node, "MatMul", dst_op_1.get());
  auto matmul_output_id = "Gemm_MatMul_" + onnx_node.output(0);
  std::vector<string> matmul_inputs{onnx_node.input(0), onnx_node.input(1)};
  std::vector<string> matmul_outputs{matmul_output_id};
  SetOpInputIndex(matmul_inputs, dst_op_1.get(), onnx_node, tensor_cache);
  SetOpOutputIndex(matmul_outputs, dst_op_1.get(), tensor_cache);
  graph->nodes.emplace_back(std::move(dst_op_1));

  std::unique_ptr<schema::CNodeT> dst_op_2(new schema::CNodeT);
  dst_op_2->name = "Gemm_BiasAdd_" + onnx_node.output(0);
  ParseOnnxNodeAttr(onnx_graph, onnx_node, "BiasAdd", dst_op_2.get());
  std::vector<string> biasadd_inputs{matmul_output_id, onnx_node.input(2)};
  std::vector<string> biasadd_outputs{onnx_node.output(0)};
  SetOpInputIndex(biasadd_inputs, dst_op_2.get(), onnx_node, tensor_cache);
  SetOpOutputIndex(biasadd_outputs, dst_op_2.get(), tensor_cache);
  graph->nodes.emplace_back(std::move(dst_op_2));
}

STATUS OnnxModelParser::ParseOnnxGivenFillNode(const onnx::NodeProto &onnx_node, TensorCache *tensor_cache) {
  // convert GivenTensorFill node to a weight/bias tensor
  auto ret = tensor_cache->FindTensor(onnx_node.output(0));
  if (ret < 0) {
    std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
    std::vector<int> shape;
    auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                             [](const onnx::AttributeProto &attr) { return attr.name() == "shape"; });
    if (iter != onnx_node.attribute().end()) {
      (void)shape.insert(shape.begin(), iter->ints().begin(), iter->ints().end());
      std::for_each(shape.begin(), shape.end(), [](int sh) { MS_LOG(DEBUG) << "shape: " << sh; });
    }
    tensor->dims = shape;
    tensor->format = schema::Format_NUM_OF_FORMAT;
    tensor->nodeType = schema::NodeType_ValueNode;
    iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                        [](const onnx::AttributeProto &attr) { return attr.name() == "values"; });
    // copy GivenIntTensorFill node value to tensor
    if (iter != onnx_node.attribute().end()) {
      size_t data_count = 1;
      std::for_each(shape.begin(), shape.end(), [&data_count](int dim) { data_count *= dim; });
      size_t data_size = 0;
      if (onnx_node.op_type() == "Int8GivenIntTensorFill") {
        // todo  how to read onnx-ori-dataType
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
        // todo  how to read onnx-ori-dataType
        tensor->dataType = kNumberTypeUInt8;
        // todo: add * sizof(string)
        data_size = data_count;
        tensor->data.resize(data_size);
        MS_LOG(DEBUG) << "tensor data size " << data_size << ", s: " << sizeof(iter->s().data());
        if (memcpy_s(tensor->data.data(), data_size, iter->s().data(), data_size) != 0) {
          MS_LOG(ERROR) << "memcpy_s failed";
          return RET_ERROR;
        }
      } else {
        MS_LOG(ERROR) << "unsupported data type " << tensor->dataType;
        return RET_ERROR;
      }
    }
    auto index = tensor_cache->AddTensor(onnx_node.output(0), tensor.release(), GRAPH_INPUT);
    MS_LOG(DEBUG) << "add given tensor: " << index;
  }
  return RET_OK;
}

STATUS OnnxModelParser::ParseOnnxNodeToDstOp(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                             schema::CNodeT *dst_op, schema::TensorT *dst_tensor,
                                             TensorCache *tensor_cache) {
  // change op_type() to name(), that is unique
  dst_op->name = onnx_node.op_type() + "_" + onnx_node.output(0);
  // dst_op->fmkType = FmkType_ONNX;
  MS_LOG(DEBUG) << "onnx op name " << onnx_node.op_type() << ", dst op name: " << dst_op->name << ", input size "
                << onnx_node.input_size();
  // get the real op type
  SetOpQuantParams(onnx_graph, onnx_node, dst_op, dst_tensor, tensor_cache);
  auto status = ParseOnnxNodeAttr(onnx_graph, onnx_node, onnx_node.op_type(), dst_op);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "parser onnx node attr failed";
    return status;
  }
  // set op input index
  std::vector<string> node_inputs;
  (void)node_inputs.insert(node_inputs.begin(), onnx_node.input().begin(), onnx_node.input().end());
  if (SetOpInputIndex(node_inputs, dst_op, onnx_node, tensor_cache)) {
    MS_LOG(ERROR) << "SetOpInputIndex failed";
    return RET_ERROR;
  }
  // set op output index
  std::vector<string> node_outputs;
  (void)node_outputs.insert(node_outputs.begin(), onnx_node.output().begin(), onnx_node.output().end());

  if (SetOpOutputIndex(node_outputs, dst_op, tensor_cache) != RET_OK) {
    MS_LOG(ERROR) << "SetOpOutputIndex failed";
    return RET_ERROR;
  }
  return RET_OK;
}

void OnnxModelParser::SetOpQuantParams(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                       schema::CNodeT *dst_op, schema::TensorT *dst_tensor, TensorCache *tensor_cache) {
  MS_ASSERT(dst_op != nullptr);
  MS_ASSERT(tensor_cache != nullptr);
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
    std::unique_ptr<schema::QuantParamT> quant_param(new (std::nothrow) schema::QuantParamT());
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
      quant_param->scale = FLT_MAX;
      quant_param->zeroPoint = 0;
      quant_param->min = FLT_MAX;
      quant_param->max = FLT_MAX;
    }
    // quant_param_array->param.emplace_back(std::move(quant_param));
    dst_tensor->quantParams.emplace_back(std::move(quant_param));
    if (argNum == 2) {
      findQuantParams++;
    }
  }
  if (findQuantParams == needQuantParams) {
    dst_op->quantType = schema::QuantType_AwareTraining;
  } else {
    dst_op->quantType = schema::QuantType_QUANT_NONE;
  }
}

STATUS OnnxModelParser::ParseOnnxNodeAttr(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                          const string &onnx_op_type, schema::CNodeT *dst_op) {
  auto node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser(onnx_op_type);
  if (node_parser == nullptr) {
    MS_LOG(EXCEPTION) << "not find " << onnx_op_type << ", node parser is nullptr";
    return RET_NULL_PTR;
  }
  return node_parser->Parse(onnx_graph, onnx_node, dst_op);
}

STATUS OnnxModelParser::SetOpInputIndex(const std::vector<string> &node_inputs, schema::CNodeT *dst_op,
                                        const onnx::NodeProto &onnx_node, TensorCache *tensor_cache) {
  schema::Format format = schema::Format_MAX;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "order") {  // do we need this code? onnx doc don't have order attr
      MS_LOG(EXCEPTION) << "find order attr";
      if (onnx_node_attr.s() == "NHWC") {
        format = schema::Format_NHWC;
      } else {
        MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s();
        return RET_ERROR;
      }
    }
  }
  for (const auto &onnx_node_input : node_inputs) {
    auto index = tensor_cache->FindTensor(onnx_node_input);
    // MS_LOG(ERROR) << onnx_node.name() << " input " << onnx_node_input << " index in tensor_cache " << index;
    if (index < 0) {  // TODO(wangzhe) can this be ignored? because it's no use
      /*
      std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
      index = tensor_cache->AddTensor(onnx_node_input, tensor.release(), OP_OUTPUT);
       */
      MS_LOG(EXCEPTION) << "input " << onnx_node_input << " of node " << onnx_node.name() << " can't be found";
      // MS_LOG(INFO) << "new index: " << index;
    }
    if (format != schema::Format_MAX) {  // TODO(wangzhe) also this
      auto inTensor = tensor_cache->GetCachedTensor().at(index);
      inTensor->format = format;
    }
    MS_LOG(DEBUG) << "node: " << onnx_node_input << ", input index: " << index;
    dst_op->inputIndex.emplace_back(index);
  }
  return RET_OK;
}

STATUS OnnxModelParser::SetOpOutputIndex(const std::vector<string> &node_outputs, schema::CNodeT *dst_op,
                                         TensorCache *tensor_cache) {
  for (const auto &onnx_node_output : node_outputs) {
    auto index = tensor_cache->FindTensor(onnx_node_output);
    if (index < 0) {
      MS_LOG(INFO) << "output of node " << dst_op->name << " not in tensor_cache, creating";
      MS_LOG(INFO) << "total " << node_outputs.size() << " outputs";
      std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);

      // GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(proto.type().tensor_type().elem_type()));
      // tensor->dataType = ;
      // tensor->dims = tflite_tensor->shape;
      tensor->nodeType = schema::NodeType_Parameter;

      index = tensor_cache->AddTensor(onnx_node_output, tensor.release(), OP_OUTPUT);
    }
    MS_LOG(DEBUG) << "node: " << onnx_node_output << ", input index: " << index;
    dst_op->outputIndex.emplace_back(index);
  }
  return RET_OK;
}

STATUS OnnxModelParser::CopyOnnxTensorData(const onnx::TensorProto &onnx_const_value, schema::TensorT *tensor) {
  size_t data_count = 1;
  std::for_each(tensor->dims.begin(), tensor->dims.end(), [&data_count](int dim) { data_count *= dim; });
  size_t data_size = 0;
  const void *tensor_data = nullptr;
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
      data_size = data_count * sizeof(int64_t);
      if (onnx_const_value.int64_data_size() == 0) {
        tensor_data = onnx_const_value.raw_data().data();
      } else {
        tensor_data = onnx_const_value.int64_data().data();
      }
      break;
    case kNumberTypeUInt8:
    case kNumberTypeInt8:
      data_size = data_count * sizeof(uint8_t);
      tensor_data = onnx_const_value.raw_data().data();
      break;
    default:
      MS_LOG(ERROR) << "unsupported data type " << tensor->dataType;
      return RET_ERROR;
  }
  tensor->data.resize(data_size);
  if (memcpy_s(static_cast<void *>(tensor->data.data()), data_size, tensor_data, data_size) != 0) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS OnnxModelParser::SetAllTensors(const TensorCache &tensor_cache, schema::MetaGraphT *graphDef) {
  std::vector<schema::TensorT *> tensors = tensor_cache.GetCachedTensor();
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

MetaGraphT *OnnxModelParser::Parse(const std::string &modelFile, const std::string &weightFile,
                                   const QuantType &quantType) {
  if (ValidateFileStr(modelFile, ".onnx") != RET_OK) {
    MS_LOG(ERROR) << "Input illegal: modelFile must be *.onnx";
    return nullptr;
  }
  std::unique_ptr<schema::MetaGraphT> dst_graph(new schema::MetaGraphT());
  onnx::ModelProto onnx_model;
  if (ReadOnnxModelFromBinary(modelFile, &onnx_model) != RET_OK) {
    MS_LOG(ERROR) << "read onnx model fail";
    return nullptr;
  }
  const onnx::GraphProto &onnx_graph = onnx_model.graph();
  MS_LOG(INFO) << "model producer name: " << onnx_model.producer_name() << ", graph name: " << onnx_graph.name();
  TensorCache tensor_cache;
  // dst_graph->name = onnx_graph.name();  // this is not used
  // find out input names and const names
  FindGraphInputAndConst(onnx_graph);
  // set const tensor
  if (SetGraphConstTensor(onnx_graph, &tensor_cache)) {
    MS_LOG(ERROR) << "SetGraphConstTensor failed";
    return nullptr;
  }
  // init onnx model graph input tensor
  if (SetGraphInputTensor(onnx_graph, dst_graph.get(), &tensor_cache)) {
    MS_LOG(ERROR) << "SetGraphInputTensor failed";
    return nullptr;
  }
  // init onnx model graph output tensor
  if (SetGraphOutputTensor(onnx_graph, dst_graph.get(), &tensor_cache)) {
    MS_LOG(ERROR) << "SetGraphOutputTensor failed";
    return nullptr;
  }
  // init op node input/output tensor, and dst_op attr
  for (const auto &onnx_node : onnx_graph.node()) {
    if (onnx_node.op_type() == "Gemm") {
      ParseOnnxGemmNode(onnx_graph, onnx_node, dst_graph.get(), &tensor_cache);
      continue;
    } else if (onnx_node.op_type() == "Int8GivenIntTensorFill" || onnx_node.op_type() == "Int8GivenTensorFill") {
      auto status = ParseOnnxGivenFillNode(onnx_node, &tensor_cache);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "ParseOnnxGivenFillNode failed: " << status;
        return nullptr;
      }
      continue;
    }

    std::unique_ptr<schema::CNodeT> dst_op(new schema::CNodeT);
    std::unique_ptr<schema::TensorT> dst_tensor(new schema::TensorT);
    auto status = ParseOnnxNodeToDstOp(onnx_graph, onnx_node, dst_op.get(), dst_tensor.get(), &tensor_cache);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "parse node " << onnx_node.op_type() << " failed";
      return nullptr;
    }
    dst_graph->nodes.emplace_back(std::move(dst_op));
  }
  SetAllTensors(tensor_cache, dst_graph.get());
  dst_graph->name = GetModelName(modelFile);
  return dst_graph.release();
}
}  // namespace lite
}  // namespace mindspore
