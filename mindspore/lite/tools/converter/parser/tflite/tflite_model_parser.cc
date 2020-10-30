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
#include <set>
#include "tools/common/graph_util.h"
#include "tools/common/storage.h"
#include "flatbuffers/flatbuffers.h"
#include "src/common/file_utils.h"
#include "tools/common/node_util.h"

namespace mindspore {
namespace lite {
TfliteModelParser::TfliteModelParser() = default;

TfliteModelParser::~TfliteModelParser() { delete[](this->tfliteModelBuf); }

std::unique_ptr<tflite::ModelT> TfliteModelParser::ReadTfliteModel(const char *model_path) {
  size_t size;
  tfliteModelBuf = ReadFile(model_path, &size);
  if (tfliteModelBuf == nullptr) {
    MS_LOG(ERROR) << "the file buffer is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)tfliteModelBuf, size);
  if (!tflite::VerifyModelBuffer(verify)) {
    MS_LOG(ERROR) << "the buffer is invalid and fail to create graph";
    return nullptr;
  }
  return tflite::UnPackModel(tfliteModelBuf);
}

STATUS TfliteModelParser::CopyConstTensorData(const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                              const tflite::TensorT *tflite_tensor, schema::TensorT *tensor) {
  auto buffer_idx = tflite_tensor->buffer;
  if (!tflite_model_buffer[buffer_idx]->data.empty()) {
    auto data_size = tflite_model_buffer[buffer_idx]->data.size();
    tensor->data.resize(data_size);
    if (memcpy_s(tensor->data.data(), data_size, tflite_model_buffer[buffer_idx]->data.data(), data_size) != EOK) {
      MS_LOG(ERROR) << "memcpy tensor data failed";
      return RET_MEMORY_FAILED;
    }
  } else {
    MS_LOG(ERROR) << "src tensor data is empty";
    return RET_INPUT_TENSOR_ERROR;
  }
  return RET_OK;
}

void TfliteModelParser::SetTensorQuantParam(const std::unique_ptr<tflite::TensorT> &tflite_tensor,
                                            schema::TensorT *tensor) {
  tensor->quantParams.clear();
  for (size_t i = 0; i < tflite_tensor->quantization->scale.size(); i++) {
    std::unique_ptr<schema::QuantParamT> quant_param = std::make_unique<QuantParamT>();
    if (!tflite_tensor->quantization->scale.empty()) {
      quant_param->scale = tflite_tensor->quantization->scale[i];
    }

    if (!tflite_tensor->quantization->zero_point.empty()) {
      quant_param->zeroPoint = tflite_tensor->quantization->zero_point[i];
    }

    if (!tflite_tensor->quantization->min.empty()) {
      quant_param->min = tflite_tensor->quantization->min[i];
    }

    if (!tflite_tensor->quantization->max.empty()) {
      quant_param->max = tflite_tensor->quantization->max[i];
    }
    quant_param->inited = true;
    tensor->quantParams.emplace_back(std::move(quant_param));
  }
}

STATUS TfliteModelParser::ConvertOp(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                    const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                    const QuantType &quant_type, schema::MetaGraphT *sub_graph) {
  int idx = 0;
  int status = RET_OK;
  NoSupportOp::GetInstance()->SetFmkType("TFLITE");
  for (const auto &tflite_op : tflite_subgraph->operators) {
    auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
    auto op_type = GetMSOpType(tflite_op_type);

    auto op = std::make_unique<schema::CNodeT>();
    op->name = op_type + "-" + std::to_string(idx++);
    op->quantType = quant_type;
    MS_LOG(INFO) << "parse op: " << op->name.c_str();

    auto node_parser = TfliteNodeParserRegistry::GetInstance()->GetNodeParser(op_type);
    if (node_parser == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(op_type);
      status = (status == RET_OK ? RET_NOT_FIND_OP : status);
      continue;
    }
    if (status == RET_OK) {
      status = node_parser->Parse(&tensorsInfo, tflite_op, tflite_model, tflite_subgraph, op.get());
      if (status != RET_OK) {
        if (status == RET_NOT_FIND_OP) {
          op_type =
            (op_type != "Custom" ? op_type : (tflite_model->operator_codes[tflite_op->opcode_index])->custom_code);
          NoSupportOp::GetInstance()->InsertOp(op_type);
        } else {
          MS_LOG(ERROR) << "node " << op_type.c_str() << " parser failed";
        }
        continue;
      }
      sub_graph->nodes.emplace_back(op.release());
      opMap[sub_graph->nodes.back()->name] = sub_graph->nodes.back().get();
      tfliteOpMap[tflite_op.get()] = sub_graph->nodes.back().get();
    }
  }
  return status;
}

STATUS TfliteModelParser::ConvertTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                        schema::MetaGraphT *sub_graph) {
  std::set<int> output_index;
  for (const auto &tflite_op : tflite_subgraph->operators) {
    for (size_t j = 0; j < tflite_op->outputs.size(); ++j) {
      int idx = tflite_op->outputs[j];
      if (idx < 0) {
        idx += tflite_subgraph->tensors.size();
      }
      output_index.insert(idx);
    }
  }
  for (size_t i = 0; i < tensorsInfo.tensorsId.size(); i++) {
    auto idx = tensorsInfo.tensorsId[i];
    if (idx < 0) {
      idx += tflite_subgraph->tensors.size();
    }
    const auto &tflite_tensor = tflite_subgraph->tensors[idx];
    std::unique_ptr<schema::TensorT> tensor = std::make_unique<schema::TensorT>();

    tensor->format = tensorsInfo.tensorsFormat[i];
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
      int status = CopyConstTensorData(tflite_model_buffer, tflite_tensor.get(), tensor.get());
      if (status != RET_OK) {
        MS_LOG(ERROR) << "obtain const tensor failed";
        return status;
      }
    }

    // set tensor attr
    if (isInput || isConst) {
      tensor->nodeType = schema::NodeType::NodeType_ValueNode;
    } else {
      if (output_index.find(idx) == output_index.end() && tflite_tensor->shape[0] == 0) {
        tensor->nodeType = schema::NodeType::NodeType_ValueNode;
      } else {
        tensor->nodeType = schema::NodeType_Parameter;
      }
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
    auto iter = tensorsInfo.tensorsIdMap.find(id);
    if (iter != tensorsInfo.tensorsIdMap.end()) {
      graph_inputs.push_back(iter->second);
    } else {
      MS_LOG(ERROR) << "get graph input failed";
      return RET_INPUT_TENSOR_ERROR;
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
    auto iter = tensorsInfo.tensorsIdMap.find(id);
    if (iter != tensorsInfo.tensorsIdMap.end()) {
      graph_outputs.push_back(iter->second);
    } else {
      MS_LOG(ERROR) << "get graph output failed";
      return RET_INPUT_TENSOR_ERROR;
    }
  }
  sub_graph->outputIndex.assign(graph_outputs.begin(), graph_outputs.end());
  return RET_OK;
}

STATUS TfliteModelParser::ConvertGroupDepthwiseOp(schema::MetaGraphT *sub_graph) {
  for (auto &op : sub_graph->nodes) {
    if (op->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
      auto attr = op->primitive->value.AsDepthwiseConv2D();
      if (attr->channelMultiplier > 1) {
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
        if (data_shape.empty()) {
          MS_LOG(DEBUG) << "the tensor's shape is dynamic, which obtain only when running";
          return RET_NO_CHANGE;
        }
        std::unique_ptr<schema::Conv2DT> conv_attr = std::make_unique<schema::Conv2DT>();
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
              MS_LOG(ERROR) << "Trans depthwiseConv Filter schema::Format failed.";
              return RET_ERROR;
            }
          } else if (weight_tensor->dataType == kNumberTypeInt8) {
            auto status = TransFilterFormat<int8_t>(weight_tensor.get(), kKHWC2CHWK);
            if (status != RET_OK) {
              MS_LOG(ERROR) << "Trans filter format failed.";
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
          weight_tensor->format = schema::Format::Format_CHWK;
        }
      }
    }
  }
  return RET_OK;
}

std::unique_ptr<schema::MetaGraphT> TfliteModelParser::ConstructMainGraph(
  const std::unique_ptr<tflite::ModelT> &tflite_model, const QuantType &quant_type) {
  if (tflite_model->subgraphs.size() < 1) {
    MS_LOG(ERROR) << "read tflite model main subgraphs failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }
  const auto &tflite_subgraph = tflite_model->subgraphs[0];

  auto meta_graph = std::make_unique<schema::MetaGraphT>();
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "new meta graph failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
    return nullptr;
  }
  meta_graph->name = "MS_model converted by TF-Lite";
  quantType = quant_type;
  // convert op
  int status = ConvertOp(tflite_model, tflite_subgraph, quant_type, meta_graph.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "parse op failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  // convert tensor
  status = ConvertTensor(tflite_subgraph, tflite_model->buffers, meta_graph.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert tensor failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  // set graph input/output
  status = GetGraphInfo(tflite_subgraph, meta_graph.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert tensors failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  // update for depthwiseConv
  status = ConvertGroupDepthwiseOp(meta_graph.get());
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "convert group depthwise conv failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  return meta_graph;
}

schema::MetaGraphT *TfliteModelParser::ParseToFb(const std::string &model_file, const std::string &weight_file,
                                                 const QuantType &quant_type) {
  // load graph
  auto tflite_model = ReadTfliteModel(model_file.c_str());
  if (tflite_model == nullptr) {
    MS_LOG(ERROR) << "read tflite model failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }

  // construct main_meta_graph
  auto main_meta_graph = ConstructMainGraph(tflite_model, quant_type);
  if (main_meta_graph == nullptr) {
    MS_LOG(ERROR) << "ConstructMainGraph failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }

  return main_meta_graph.release();
}
}  // namespace lite
}  // namespace mindspore
