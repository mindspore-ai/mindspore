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

#include "mindspore/lite/tools/converter/parser/tflite/tflite_model_parser.h"
#include <fstream>
#include <utility>
#include <memory>
#include "tools/common/graph_util.h"
#include "tools/common/storage.h"
#include "flatbuffers/flatbuffers.h"
#include "utils/log_adapter.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
TfliteModelParser::TfliteModelParser() {}

TfliteModelParser::~TfliteModelParser() {}

std::unique_ptr<tflite::ModelT> TfliteModelParser::ReadTfliteModelFromFlat(const char *model_path) {
  size_t size;
  auto buf = ReadFile(model_path, &size);
  if (buf == nullptr) {
    // MS_LOGE("the file buffer is nullptr");
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)buf, size);
  if (!tflite::VerifyModelBuffer(verify)) {
    // MS_LOGE("the buffer is invalid and fail to create graph");
    return nullptr;
  }
  return tflite::UnPackModel(buf);
}

std::string TfliteModelParser::GetTfliteNodeType(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                 const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  auto msOpType = GetMSOpType(tflite_op_type);
  return msOpType;
}

STATUS TfliteModelParser::SetAllTensors(const TensorCache &tensor_cache, schema::MetaGraphT *sub_graphDef) {
  std::vector<schema::TensorT *> tensors = tensor_cache.GetCachedTensor();
  for (auto iter : tensors) {
    std::unique_ptr<schema::TensorT> temp(iter);
    temp->format = schema::Format_NHWC;
    sub_graphDef->allTensors.emplace_back(move(temp));
  }
  return RET_OK;
}

STATUS TfliteModelParser::ParseTfliteQuantParams(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                                 const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  auto dst_op = tfliteOpMap.at(tflite_op.get());

  std::vector<uint32_t> quant_params_index;
  quant_params_index.insert(quant_params_index.end(), tflite_op->inputs.begin(), tflite_op->inputs.end());
  quant_params_index.insert(quant_params_index.end(), tflite_op->outputs.begin(), tflite_op->outputs.end());
  for (const auto &index : quant_params_index) {
    const auto &tflite_tensor = tflite_subgraph->tensors[index];
    if (tflite_tensor->quantization->scale.empty() && tflite_tensor->quantization->zero_point.empty() &&
        tflite_tensor->quantization->min.empty() && tflite_tensor->quantization->max.empty()) {
      continue;
    }
    std::unique_ptr<schema::QuantParamT> quant_param(new schema::QuantParamT());
    if (!tflite_tensor->quantization->scale.empty()) {
      quant_param->scale = tflite_tensor->quantization->scale[0];
    }

    if (!tflite_tensor->quantization->zero_point.empty()) {
      quant_param->zeroPoint = tflite_tensor->quantization->zero_point[0];
    }

    if (!tflite_tensor->quantization->min.empty()) {
      quant_param->min = tflite_tensor->quantization->min[0];
    }

    if (!tflite_tensor->quantization->max.empty()) {
      quant_param->max = tflite_tensor->quantization->max[0];
    }
  }
  dst_op->quantType = schema::QuantType_AwareTrainning;
  return RET_OK;
}

STATUS TfliteModelParser::SetOpOutputIdx(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                         const std::unique_ptr<tflite::OperatorT> &tflite_op, schema::CNodeT *op,
                                         TensorCache *tensorCache) {
  for (const auto &index : tflite_op->outputs) {
    const auto &tflite_tensor = tflite_subgraph->tensors[index];
    std::unique_ptr<schema::TensorT> tensor(new schema::TensorT());
    tensor->dataType = GetTfliteDataType(tflite_tensor->type);
    tensor->dims = tflite_tensor->shape;
    tensor->nodeType = schema::NodeType_Parameter;
    auto opOutputIndex = tensorCache->AddTensor(tflite_tensor->name, tensor.release(), OP_OUTPUT);
    op->outputIndex.emplace_back(opOutputIndex);
  }

  return RET_OK;
}

STATUS TfliteModelParser::SetOpInputIdx(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::OperatorT> &tflite_op, TensorCache *tensorCache) {
  auto op_type = GetTfliteNodeType(tflite_op, tflite_model);
  std::vector<int32_t> op_inputs(tflite_op->inputs);
  if (op_type == "DeConv2D") {
    reverse(op_inputs.begin(), op_inputs.end());
  }

  for (const auto &tflite_index : op_inputs) {
    const auto &tflite_tensor = tflite_subgraph->tensors[tflite_index];
    auto tensor_name = tflite_tensor->name;
    auto op = tfliteOpMap[tflite_op.get()];
    unsigned int index = tensorCache->FindTensor(tensor_name);
    if (index != -1) {
      op->inputIndex.push_back(index);
    }
  }

  return RET_OK;
}

STATUS TfliteModelParser::ParseOp(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                  const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                  schema::MetaGraphT *subGraph,
                                  mindspore::lite::TensorCache *tensorCache) {
  auto i = 0;
  for (const auto &tflite_op : tflite_subgraph->operators) {
    auto opType = GetTfliteNodeType(tflite_op, tflite_model);

    std::unique_ptr<schema::CNodeT> op(new schema::CNodeT);
    op->name = opType + "-" + std::to_string(i++);

    // MS_LOGD("parse op: [%s]", op->name.c_str());

    // 1. init op attr params
    auto node_parser = TfliteNodeParserRegistry::GetInstance()->GetNodeParser(opType);
    if (node_parser == nullptr) {
      // MS_LOGE("node %s parser is nullptr", opType.c_str());
      return RET_NULL_PTR;
    }

    auto status = node_parser->Parse(tflite_op, tflite_subgraph->tensors, tflite_model->buffers,
                                     tflite_model->operator_codes, op.get(), tensorCache, false);
    if (status != RET_OK) {
      // MS_LOGE("node %s parser failed", opType.c_str());
      return RET_ERROR;
    }

    status = SetOpOutputIdx(tflite_subgraph, tflite_op, op.get(), tensorCache);
    if (status != RET_OK) {
      // MS_LOGE("Set Op %s Output Index Failed!", op->name.c_str());
      return RET_ERROR;
    }

    subGraph->nodes.emplace_back(std::move(op));
    opMap[subGraph->nodes.back()->name] = subGraph->nodes.back().get();
    tfliteOpMap[tflite_op.get()] = subGraph->nodes.back().get();
  }
  return RET_OK;
}

void TfliteModelParser::SetInputTensor(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       TensorCache *tensor_cache) {
  for (const auto &index : tflite_subgraph->inputs) {
    const auto &tflite_tensor = tflite_subgraph->tensors[index];
    std::unique_ptr<schema::TensorT> tensor(new schema::TensorT());
    tensor->format = schema::Format_NHWC;
    tensor->dataType = GetTfliteDataType(tflite_tensor->type);
    tensor->nodeType = schema::NodeType_ValueNode;
    tensor->dims = tflite_tensor->shape;
    tensor_cache->AddTensor(tflite_tensor->name, tensor.release(), GRAPH_INPUT);
  }
}

void TfliteModelParser::SetGraphTensorIndex(const mindspore::lite::TensorCache &tensorCache,
                                            schema::MetaGraphT *subGraphDef) {
  auto opGraph = OpGraphT::Build(subGraphDef);
  auto graphInputs = tensorCache.GetGraphInputs();
  auto graphOutputs = opGraph->GetOutputNode();

  subGraphDef->inputIndex.assign(graphInputs.begin(), graphInputs.end());

  for (const auto &output : graphOutputs) {
    auto op = opMap[output->ID()];
    for (auto outputIndex : op->outputIndex) {
      subGraphDef->outputIndex.emplace_back(outputIndex);
    }
  }
}

MetaGraphT *TfliteModelParser::Parse(const std::string &modelFile, const std::string &weightFile) {
  std::unique_ptr<schema::MetaGraphT> subGraph(new schema::MetaGraphT);
  if (ValidateFileStr(modelFile, ".tflite") != RET_OK) {
    // MS_LOGE("INPUT ILLEGAL: modelFile must be *.tflite");
    return nullptr;
  }
  std::unique_ptr<tflite::ModelT> tflite_model(new tflite::ModelT());
  tflite_model = ReadTfliteModelFromFlat(modelFile.c_str());
  if (tflite_model == nullptr) {
    // MS_LOGE("read tflite model failed");
    return nullptr;
  }
  TensorCache tensorCache;
  if (tflite_model->subgraphs.size() != 1) {
    MS_LOG(ERROR) << "read tflite model subgraphs failed";
    return nullptr;
  }

  const auto &tflite_subgraph = tflite_model->subgraphs[0];
  subGraph->name = "MS_model converted by TF-Lite";

  // set dst subGraph input/output tensor
  SetInputTensor(tflite_model, tflite_subgraph, &tensorCache);
  // set dst subGraph op attr etc.
  auto status = ParseOp(tflite_model, tflite_subgraph, subGraph.get(), &tensorCache);
  if (status != RET_OK) {
    // MS_LOGE("ParseOp failed.");
    return nullptr;
  }

  for (const auto &tflite_op : tflite_subgraph->operators) {
    auto status_tmp = SetOpInputIdx(tflite_model, tflite_subgraph, tflite_op, &tensorCache);
    if (status_tmp != RET_OK) {
      // MS_LOGE("Set Op %s Input Index Failed!", tfliteOpMap.at(tflite_op.get())->name.c_str());
    }
  }

  for (const auto &tflite_op : tflite_subgraph->operators) {
    auto statusTmp = ParseTfliteQuantParams(tflite_subgraph, tflite_op);
    if (statusTmp != RET_OK) {
      // MS_LOGE("ParseTfliteQuantParams %s Failed!", tfliteOpMap.at(tflite_op.get())->name.c_str());
    }
  }

  SetGraphTensorIndex(tensorCache, subGraph.get());
  SetAllTensors(tensorCache, subGraph.get());
  return subGraph.release();
//  return Fb2Anf(subGraph.release());
}
}  // namespace lite
}  // namespace mindspore

