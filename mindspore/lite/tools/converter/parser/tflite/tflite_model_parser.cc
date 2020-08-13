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
#include "tools/common/graph_util.h"
#include "tools/common/storage.h"
#include "flatbuffers/flatbuffers.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
TfliteModelParser::TfliteModelParser() {}

TfliteModelParser::~TfliteModelParser() {}

std::unique_ptr<tflite::ModelT> TfliteModelParser::ReadTfliteModelFromFlat(const char *model_path) {
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
void TfliteModelParser::SetMsTensorFromTflite(const std::unique_ptr<tflite::TensorT> &tflite_tensor,
                                              schema::TensorT *tensor) {
  std::unique_ptr<schema::QuantParamT> quant_param(new QuantParamT());
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

STATUS TfliteModelParser::ParseTfliteQuantParams(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                                 const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                 schema::CNodeT *op, TensorCache *tensor_cache) {
  MS_ASSERT(op->outputIndex.size() == tflite_op->outputs.size());
  for (size_t i = 0; i < tflite_op->inputs.size() && i < op->inputIndex.size(); i++) {
    const auto &tflite_tensor = tflite_subgraph->tensors[tflite_op->inputs.at(i)];
    if (tflite_tensor->quantization->scale.empty() && tflite_tensor->quantization->zero_point.empty() &&
        tflite_tensor->quantization->min.empty() && tflite_tensor->quantization->max.empty()) {
      continue;
    }
    auto &inTensor = tensor_cache->GetCachedTensor().at(op->inputIndex.at(i));
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Parse tflite quant params inTensor is null";
      return RET_NULL_PTR;
    }
    SetMsTensorFromTflite(tflite_tensor, inTensor);
  }
  for (size_t i = 0; i < tflite_op->outputs.size() && i < op->outputIndex.size(); i++) {
    const auto &tflite_tensor = tflite_subgraph->tensors[tflite_op->outputs.at(i)];
    if (tflite_tensor->quantization->scale.empty() && tflite_tensor->quantization->zero_point.empty() &&
        tflite_tensor->quantization->min.empty() && tflite_tensor->quantization->max.empty()) {
      continue;
    }
    auto &outTensor = tensor_cache->GetCachedTensor().at(op->outputIndex.at(i));
    if (outTensor == nullptr) {
      MS_LOG(ERROR) << "Parse tflite quant params outTensor is null";
      return RET_NULL_PTR;
    }
    SetMsTensorFromTflite(tflite_tensor, outTensor);
  }
  return RET_OK;
}

STATUS TfliteModelParser::SetOpOutputIdx(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                         const std::unique_ptr<tflite::OperatorT> &tflite_op, schema::CNodeT *op,
                                         TensorCache *tensorCache) {
  for (const auto &index : tflite_op->outputs) {
    const auto &tflite_tensor = tflite_subgraph->tensors[index];
    if (tflite_tensor == nullptr) {
      MS_LOG(ERROR) << "tensor with id = " << index << " is null";
      return RET_ERROR;
    }
    std::unique_ptr<schema::TensorT> tensor(new schema::TensorT());
    tensor->dataType = GetTfliteDataType(tflite_tensor->type);
    // change dataType to int8 to fit ms-lite op
    if (tensor->dataType == TypeId::kNumberTypeUInt8) {
      tensor->dataType = TypeId::kNumberTypeInt8;
    }
    tensor->dims = tflite_tensor->shape;
    tensor->nodeType = schema::NodeType_Parameter;
    auto opOutputIndex = tensorCache->AddTensor(tflite_tensor->name, tensor.release(), OP_OUTPUT);
    op->outputIndex.emplace_back(opOutputIndex);
  }
  return RET_OK;
}

STATUS TfliteModelParser::SetOpInputIdx(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::OperatorT> &tflite_op, schema::CNodeT *op,
                                        TensorCache *tensor_cache) {
  auto op_type = GetTfliteNodeType(tflite_op, tflite_model);
  std::vector<int32_t> op_inputs(tflite_op->inputs);
  if (op_type == "DeConv2D") {
    reverse(op_inputs.begin(), op_inputs.end());
  }

  for (const auto &tflite_index : op_inputs) {
    const auto &tflite_tensor = tflite_subgraph->tensors[tflite_index];
    if (tflite_tensor == nullptr) {
      MS_LOG(ERROR) << "tensor with id = " << tflite_index << " is null";
      return RET_ERROR;
    }
    auto tensor_name = tflite_tensor->name;
    unsigned int index = tensor_cache->FindTensor(tensor_name);
    if (index != -1) {
      op->inputIndex.push_back(index);
    }
  }

  return RET_OK;
}

STATUS TfliteModelParser::ParseOp(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                  const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                  schema::MetaGraphT *subGraph, mindspore::lite::TensorCache *tensorCache,
                                  const QuantType &quantType) {
  auto i = 0;
  for (const auto &tflite_op : tflite_subgraph->operators) {
    auto opType = GetTfliteNodeType(tflite_op, tflite_model);

    std::unique_ptr<schema::CNodeT> op(new schema::CNodeT);
    op->name = opType + "-" + std::to_string(i++);
    op->quantType = quantType;
    MS_LOG(INFO) << "parse op: " << op->name.c_str();

    auto node_parser = TfliteNodeParserRegistry::GetInstance()->GetNodeParser(opType);
    if (node_parser == nullptr) {
      MS_LOG(ERROR) << "cannot find node parser, opType: " << opType.c_str();
      continue;
      // return RET_NULL_PTR;
    }

    auto status = node_parser->Parse(tflite_op, tflite_subgraph->tensors, tflite_model->buffers,
                                     tflite_model->operator_codes, op.get(), tensorCache, false);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "node " << opType.c_str() << " parser failed";
      return RET_ERROR;
    }

    status = SetOpOutputIdx(tflite_subgraph, tflite_op, op.get(), tensorCache);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set op " << opType.c_str() << " output index failed";
      return RET_ERROR;
    }

    status = SetOpInputIdx(tflite_model, tflite_subgraph, tflite_op, op.get(), tensorCache);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set op " << opType.c_str() << " input index failed";
      return RET_ERROR;
    }

    status = ParseTfliteQuantParams(tflite_subgraph, tflite_op, op.get(), tensorCache);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "parse op " << opType.c_str() << " quant parameters failed";
      return RET_ERROR;
    }

    subGraph->nodes.emplace_back(std::move(op));
    opMap[subGraph->nodes.back()->name] = subGraph->nodes.back().get();
    tfliteOpMap[tflite_op.get()] = subGraph->nodes.back().get();
  }
  return RET_OK;
}

void TfliteModelParser::SetInputTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       TensorCache *tensor_cache) {
  for (const auto &index : tflite_subgraph->inputs) {
    const auto &tflite_tensor = tflite_subgraph->tensors[index];
    std::unique_ptr<schema::TensorT> tensor(new schema::TensorT());
    tensor->format = schema::Format_NHWC;
    tensor->dataType = GetTfliteDataType(tflite_tensor->type) != TypeId::kNumberTypeUInt8
                         ? GetTfliteDataType(tflite_tensor->type)
                         : TypeId::kNumberTypeInt8;
    tensor->nodeType = schema::NodeType_Parameter;
    tensor->dims = tflite_tensor->shape;
    tensor_cache->AddTensor(tflite_tensor->name, tensor.release(), GRAPH_INPUT);
  }
}

void TfliteModelParser::SetGraphTensorIndex(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                            const std::unique_ptr<tflite::ModelT> &tflite_model,
                                            const mindspore::lite::TensorCache &tensorCache,
                                            schema::MetaGraphT *subGraphDef) {
  auto graphInputs = tensorCache.GetGraphInputs();
  subGraphDef->inputIndex.assign(graphInputs.begin(), graphInputs.end());

  for (auto outputIndex : tflite_subgraph->outputs) {
    int i = 0;
    bool found = false;
    for (const auto &tfliteOp : tflite_subgraph->operators) {
      int j = 0;
      auto opType = GetTfliteNodeType(tfliteOp, tflite_model);
      std::string opName = opType + "-" + std::to_string(i++);
      for (auto opOutputIndex : tfliteOp->outputs) {
        if (outputIndex == opOutputIndex) {
          subGraphDef->outputIndex.emplace_back(opMap[opName]->outputIndex[j]);
          found = true;
          break;
        }
        j++;
      }
      if (found) {
        break;
      }
    }
  }
}

MetaGraphT *TfliteModelParser::Parse(const std::string &modelFile, const std::string &weightFile,
                                     const QuantType &quantType) {
  if (ValidateFileStr(modelFile, ".tflite") != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.tflite";
    return nullptr;
  }

  std::unique_ptr<tflite::ModelT> tflite_model(new tflite::ModelT());
  tflite_model = ReadTfliteModelFromFlat(modelFile.c_str());
  if (tflite_model == nullptr) {
    MS_LOG(ERROR) << "read tflite model failed";
    return nullptr;
  }
  if (tflite_model->subgraphs.size() != 1) {
    MS_LOG(ERROR) << "read tflite model subgraphs failed";
    return nullptr;
  }
  const auto &tflite_subgraph = tflite_model->subgraphs[0];

  // set dst subGraph input/output tensor
  TensorCache tensorCache;
  SetInputTensor(tflite_subgraph, &tensorCache);

  // set dst subGraph op attr and tensor_cache.
  std::unique_ptr<schema::MetaGraphT> subGraph(new schema::MetaGraphT);
  subGraph->name = "MS_model converted by TF-Lite";
  auto status = ParseOp(tflite_model, tflite_subgraph, subGraph.get(), &tensorCache, quantType);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseOp failed.";
    return nullptr;
  }

  SetGraphTensorIndex(tflite_subgraph, tflite_model, tensorCache, subGraph.get());
  SetAllTensors(tensorCache, subGraph.get());
  return subGraph.release();
}
}  // namespace lite
}  // namespace mindspore
