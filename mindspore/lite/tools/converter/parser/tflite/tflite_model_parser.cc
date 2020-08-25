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

#include "tools/converter/parser/tflite/tflite_model_parser.h"
#include <utility>
#include <memory>
#include <vector>
#include "tools/common/graph_util.h"
#include "tools/common/storage.h"
#include "flatbuffers/flatbuffers.h"
#include "src/common/file_utils.h"
#include "tools/common/node_util.h"

namespace mindspore {
namespace lite {
TfliteModelParser::TfliteModelParser() = default;

TfliteModelParser::~TfliteModelParser() = default;

std::unique_ptr<tflite::ModelT> TfliteModelParser::ReadTfliteModel(const char *model_path) {
  size_t size;
  auto buf = ReadFile(model_path, &size);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "the file buffer is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)buf, size);
  if (!tflite::VerifyModelBuffer(verify)) {
    MS_LOG(ERROR) << "the buffer is invalid and fail to create graph";
    return nullptr;
  }
  return tflite::UnPackModel(buf);
}

STATUS TfliteModelParser::CopyConstTensorData(const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                              const tflite::TensorT *tflite_tensor,
                                              schema::TensorT *tensor) {
  auto count = 1;
  std::for_each(tflite_tensor->shape.begin(), tflite_tensor->shape.end(), [&](int32_t sha) { count *= sha; });
  auto data_size = count * GetDataTypeSize(TypeId(tensor->dataType));
  auto buffer_idx = tflite_tensor->buffer;
  if (!tflite_model_buffer[buffer_idx]->data.empty()) {
    tensor->data.resize(data_size);
    if (memcpy_s(tensor->data.data(), data_size, tflite_model_buffer[buffer_idx]->data.data(), data_size)) {
      MS_LOG(ERROR) << "memcpy tensor data failed";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "src tensor data is empty";
    return RET_ERROR;
  }
  return RET_OK;
}

void TfliteModelParser::SetTensorQuantParam(const std::unique_ptr<tflite::TensorT> &tflite_tensor,
                                            schema::TensorT *tensor) {
  std::unique_ptr<schema::QuantParamT> quant_param = std::make_unique<QuantParamT>();
  if (!tflite_tensor->quantization->scale.empty()) {
    quant_param->scale = tflite_tensor->quantization->scale[0];
  }

  if (!tflite_tensor->quantization->zero_point.empty()) {
    quant_param->zeroPoint = tflite_tensor->quantization->zero_point[0];
  }

  // change quant param min to 0 to fit ms-lite ops
  if (tensor->dataType == TypeId::kNumberTypeInt8) {
    quant_param->zeroPoint = quant_param->zeroPoint - 128;
  }

  if (!tflite_tensor->quantization->min.empty()) {
    quant_param->min = tflite_tensor->quantization->min[0];
  }

  if (!tflite_tensor->quantization->max.empty()) {
    quant_param->max = tflite_tensor->quantization->max[0];
  }
  quant_param->inited = true;
  tensor->quantParams.clear();
  tensor->quantParams.emplace_back(std::move(quant_param));
}

STATUS TfliteModelParser::ConvertOp(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                    const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                    const QuantType &quant_type,
                                    schema::MetaGraphT *sub_graph) {
  int idx = 0;
  for (const auto &tflite_op : tflite_subgraph->operators) {
    auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
    auto op_type = GetMSOpType(tflite_op_type);
    if (op_type == "CUSTOM") {
      auto custom_type = (tflite_model->operator_codes[tflite_op->opcode_index])->custom_code;
      MS_LOG(ERROR) << "CUSTOM op is not supported, the type is " << custom_type;
      return RET_ERROR;
    }

    std::unique_ptr<schema::CNodeT> op = std::make_unique<schema::CNodeT>();
    op->name = op_type + "-" + std::to_string(idx++);
    op->quantType = quant_type;
    MS_LOG(INFO) << "parse op: " << op->name.c_str();

    auto node_parser = TfliteNodeParserRegistry::GetInstance()->GetNodeParser(op_type);
    if (node_parser == nullptr) {
      MS_LOG(ERROR) << "cannot find node parser, opType: " << op_type.c_str();
      return RET_NULL_PTR;
    }
    if (node_parser->Parse(tflite_op, tflite_subgraph->tensors, tflite_model->buffers, op.get(), &tensorsId,
                           &tensorsFormat, &tensorsIdMap) != RET_OK) {
      MS_LOG(ERROR) << "node " << op_type.c_str() << " parser failed";
      return RET_ERROR;
    }

    sub_graph->nodes.emplace_back(op.release());
    opMap[sub_graph->nodes.back()->name] = sub_graph->nodes.back().get();
    tfliteOpMap[tflite_op.get()] = sub_graph->nodes.back().get();
  }
  return RET_OK;
}

STATUS TfliteModelParser::ConvertTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                        schema::MetaGraphT *sub_graph) {
  for (size_t i = 0; i < tensorsId.size(); i++) {
    auto idx = tensorsId[i];
    if (idx < 0) {
      idx += tflite_subgraph->tensors.size();
    }
    const auto &tflite_tensor = tflite_subgraph->tensors[idx];
    std::unique_ptr<schema::TensorT> tensor = std::make_unique<schema::TensorT>();

    tensor->format = tensorsFormat[i];
    tensor->dataType = GetTfliteDataType(tflite_tensor->type);
    tensor->dims = tflite_tensor->shape;

    // if graph input tensor
    bool isInput = false;
    auto tflite_inputs = tflite_subgraph->inputs;
    for (int tflite_input : tflite_inputs) {
      if (idx == tflite_input) {
        isInput = true;
        break;
      }
    }

    // add data for const tensor
    auto &tensor_buffer = tflite_model_buffer.at(tflite_tensor->buffer);
    auto isConst = (!tensor_buffer->data.empty());
    if (isConst) {
      CopyConstTensorData(tflite_model_buffer, tflite_tensor.get(), tensor.get());
    } else if (quantType == QuantType_AwareTraining && tensor->dataType == TypeId::kNumberTypeUInt8) {
      // set in/out tensor to int8 to fit ms-lite op
      tensor->dataType = TypeId::kNumberTypeInt8;
    }

    // set tensor attr
    if (isInput || isConst) {
      tensor->nodeType = schema::NodeType_ValueNode;
    } else {
      tensor->nodeType = schema::NodeType_Parameter;
    }

    // quant param
    if (tflite_tensor->quantization != nullptr &&
        !(tflite_tensor->quantization->scale.empty() && tflite_tensor->quantization->zero_point.empty() &&
          tflite_tensor->quantization->min.empty() && tflite_tensor->quantization->max.empty())) {
      SetTensorQuantParam(tflite_tensor, tensor.get());
    }

    tensors.push_back(tensor.release());
  }

  for (auto iter : tensors) {
    std::unique_ptr<schema::TensorT> temp(iter);
    sub_graph->allTensors.emplace_back(move(temp));
  }
  return RET_OK;
}

STATUS TfliteModelParser::GetGraphInfo(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       schema::MetaGraphT *sub_graph) {
  int id;

  // graph input
  std::vector<int> graph_inputs;
  for (size_t i = 0; i < tflite_subgraph->inputs.size(); i++) {
    const int idx = tflite_subgraph->inputs[i];
    if (idx < 0) {
      id = idx + tflite_subgraph->tensors.size();
    } else {
      id = idx;
    }
    auto iter = tensorsIdMap.find(id);
    if (iter != tensorsIdMap.end()) {
      graph_inputs.push_back(iter->second);
    }
  }
  sub_graph->inputIndex.assign(graph_inputs.begin(), graph_inputs.end());

  // graph output
  std::vector<int> graph_outputs;
  for (size_t i = 0; i < tflite_subgraph->outputs.size(); i++) {
    const int idx = tflite_subgraph->outputs[i];
    if (idx < 0) {
      id = idx + tflite_subgraph->tensors.size();
    } else {
      id = idx;
    }
    auto iter = tensorsIdMap.find(id);
    if (iter != tensorsIdMap.end()) {
      graph_outputs.push_back(iter->second);
    }
  }
  sub_graph->outputIndex.assign(graph_outputs.begin(), graph_outputs.end());
  return RET_OK;
}

STATUS TfliteModelParser::ConvertGroupDepthwiseOp(schema::MetaGraphT* sub_graph) {
  for (auto &op : sub_graph->nodes) {
    if (op->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
      auto attr = op->primitive->value.AsDepthwiseConv2D();
      if (attr->channelMultiplier > 1) {
        std::unique_ptr<schema::Conv2DT> conv_attr = std::make_unique<schema::Conv2DT>();
        // get channel attr
        if (op->inputIndex.empty()) {
          MS_LOG(ERROR) << "the input of DepthwiseConv2D is null";
          return RET_NULL_PTR;
        }
        auto data_id = op->inputIndex[0];
        if (sub_graph->allTensors.size() <= data_id) {
          MS_LOG(ERROR) << "the number of allTensors is less than " << data_id;
          return RET_ERROR;
        }
        auto &data_tensor = sub_graph->allTensors.at(data_id);
        if (data_tensor == nullptr) {
          MS_LOG(ERROR) << "the data tensor is null";
          return RET_NULL_PTR;
        }
        auto data_shape = data_tensor->dims;

        if (data_shape[3] == 1) {
          conv_attr->channelIn = data_shape[3];
          conv_attr->channelOut = conv_attr->channelIn * attr->channelMultiplier;

          // update attr
          conv_attr->group = 1;
          conv_attr->format = attr->format;
          conv_attr->kernelH = attr->kernelH;
          conv_attr->kernelW = attr->kernelW;
          conv_attr->strideH = attr->strideH;
          conv_attr->strideW = attr->strideW;
          conv_attr->padMode = attr->padMode;
          conv_attr->padUp = attr->padUp;
          conv_attr->padDown = attr->padDown;
          conv_attr->padLeft = attr->padLeft;
          conv_attr->padRight = attr->padRight;
          conv_attr->dilateH = attr->dilateH;
          conv_attr->dilateW = attr->dilateW;
          conv_attr->hasBias = attr->hasBias;
          conv_attr->activationType = attr->activationType;

          op->primitive->value.type = schema::PrimitiveType_Conv2D;
          op->primitive->value.value = conv_attr.release();

          // update weight
          auto weight_id = op->inputIndex[1];
          auto &weight_tensor = sub_graph->allTensors.at(weight_id);
          if (weight_tensor->dataType == TypeId::kNumberTypeUInt8) {
            auto status = TransFilterFormat<uint8_t>(weight_tensor.get(), kKHWC2CHWK);
            if (status != RET_OK) {
              MS_LOG(ERROR) << "Trans depthwiseConv Filter Format failed.";
              return RET_ERROR;
            }
          } else if (weight_tensor->dataType == kNumberTypeFloat32 || weight_tensor->dataType == kNumberTypeFloat) {
            auto status = TransFilterFormat<float>(weight_tensor.get(), kKHWC2CHWK);
            if (status != RET_OK) {
              MS_LOG(ERROR) << "Trans filter format failed.";
              return RET_ERROR;
            }
          } else {
            MS_LOG(ERROR) << "The dataType of weight tensor is unsupported.";
            return RET_ERROR;
          }
          weight_tensor->format = schema::Format_CHWK;
        }
      }
    }
  }
  return RET_OK;
}

MetaGraphT *TfliteModelParser::Parse(const std::string &model_file,
                                     const std::string &weight_file,
                                     const QuantType &quant_type) {
  std::unique_ptr<schema::MetaGraphT> sub_graph = std::make_unique<schema::MetaGraphT>();
  sub_graph->name = "MS_model converted by TF-Lite";
  quantType = quant_type;

  // load graph
  std::unique_ptr<tflite::ModelT> tflite_model = ReadTfliteModel(model_file.c_str());

  if (tflite_model->subgraphs.size() != 1) {
    MS_LOG(ERROR) << "read tflite model subgraphs failed";
    return nullptr;
  }
  const auto &tflite_subgraph = tflite_model->subgraphs[0];

  // convert op
  if (ConvertOp(tflite_model, tflite_subgraph, quant_type, sub_graph.get()) != RET_OK) {
    MS_LOG(ERROR) << "parse op failed.";
    return nullptr;
  }

  // convert tensor
  if (ConvertTensor(tflite_subgraph, tflite_model->buffers, sub_graph.get()) != RET_OK) {
    MS_LOG(ERROR) << "convert tensor failed";
    return nullptr;
  }

  // set graph input/output
  if (GetGraphInfo(tflite_subgraph, sub_graph.get()) != RET_OK) {
    MS_LOG(ERROR) << "convert tensors failed";
    return nullptr;
  }

  // update for depthwiseConv
  if (ConvertGroupDepthwiseOp(sub_graph.get()) != RET_OK) {
    MS_LOG(ERROR) << "convert group depthwise conv failed";
    return nullptr;
  }

  return sub_graph.release();
}
}  // namespace lite
}  // namespace mindspore
