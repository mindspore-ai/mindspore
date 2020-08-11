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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_model_parser.h"
#include <vector>
#include <iostream>
#include <utility>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_node_parser_registry.h"
#include "mindspore/lite/tools/converter/parser/caffe/caffe_parse_utils.h"
#include "mindspore/lite/tools/converter/parser/caffe/caffe_inspector.h"
#include "tools/common/graph_util.h"

namespace mindspore {
namespace lite {
CaffeModelParser::CaffeModelParser() {}

CaffeModelParser::~CaffeModelParser() {}

const std::set<std::string> CaffeModelParser::skipedLayerType = {"Dropout"};

schema::MetaGraphT *CaffeModelParser::Parse(const std::string &modelFile, const std::string &weightFile,
                                            const QuantType &quantType) {
  std::unique_ptr<schema::MetaGraphT> graph(new schema::MetaGraphT());

  if (ValidateFileStr(modelFile, ".prototxt") != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.prototxt";
    return nullptr;
  }

  if (weightFile.empty()) {
    MS_LOG(ERROR) << "INPUT MISSING: weightFile is necessary";
    return nullptr;
  }

  if (ValidateFileStr(weightFile, ".caffemodel") != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: weightFile must be *.caffemodel";
    return nullptr;
  }

  std::unique_ptr<schema::MetaGraphT> subGraphDef(new schema::MetaGraphT());
  TensorCache tensorCache;

  caffe::NetParameter proto;
  if (ReadProtoFromText((const char *)modelFile.c_str(), &proto) != RET_OK) {
    MS_LOG(ERROR) << "Read prototxt file failed, model path: " << modelFile;
    return nullptr;
  }
  subGraphDef->name = proto.name();

  caffe::NetParameter weight;
  if (ReadProtoFromBinaryFile((const char *)weightFile.c_str(), &weight) != RET_OK) {
    MS_LOG(ERROR) << "Read caffemodel file failed, model path: " << weightFile;
    return nullptr;
  }

  auto status = GetModelInput(proto, &tensorCache);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetModelInput failed " << status;
    return nullptr;
  }

  status = ParseLayer(proto, weight, &tensorCache, subGraphDef.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseLayer failed " << status;
    return nullptr;
  }

  // set inputTensor index and outputTensor index for the whole graph
  status = SetGraphTensorIndex(proto, &tensorCache, subGraphDef.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Set inputTensor index and outputTensor index for graph failed!";
    return nullptr;
  }
  subGraphDef->name = GetModelName(modelFile);
  // set all tensors to graph
  SetAllTensors(tensorCache, subGraphDef.get());
  graph = move(subGraphDef);

  // ConvertCaffeBatchNorm(graph.get());

  return graph.release();
  //  return Fb2Anf(graph.release());
}

STATUS CaffeModelParser::SetOpInputIdx(const caffe::LayerParameter &layer, schema::CNodeT *op,
                                       TensorCache *tensorCache) {
  for (int i = 0; i < layer.bottom_size(); i++) {
    int index = tensorCache->FindTensor(layer.bottom(i));
    if (index >= 0) {
      op->inputIndex.emplace_back(index);
    } else {
      // MS_LOGE("Can't find input layer for %s.", layer.name().c_str());
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetOpOutputIdx(const caffe::LayerParameter &layer, schema::CNodeT *op,
                                        TensorCache *tensorCache) {
  for (int i = 0; i < layer.top_size(); i++) {
    std::unique_ptr<schema::TensorT> msTensor(new schema::TensorT());
    op->outputIndex.emplace_back(tensorCache->AddTensor(layer.top(i), msTensor.release(), OP_OUTPUT));
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetWeightTensor(const std::vector<schema::TensorT *> &weightVec, schema::CNodeT *op,
                                         TensorCache *tensorCache) {
  for (auto iter : weightVec) {
    op->inputIndex.emplace_back(tensorCache->AddTensor("Weight", iter, CONST));
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetAllTensors(const TensorCache &tensorCache, schema::MetaGraphT *subGraphDef) {
  std::vector<schema::TensorT *> tensors = tensorCache.GetCachedTensor();
  for (auto iter : tensors) {
    std::unique_ptr<schema::TensorT> temp(iter);
    subGraphDef->allTensors.emplace_back(move(temp));
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetGraphTensorIndex(const caffe::NetParameter &proto, TensorCache *tensorCache,
                                             schema::MetaGraphT *subGraphDef) {
  CaffeInspector caffeInspector;
  caffeInspector.InspectModel(proto);
  for (auto iter : caffeInspector.GetGraphInput()) {
    int index = tensorCache->FindTensor(iter);
    if (index >= 0) {
      subGraphDef->inputIndex.emplace_back(index);
    } else {
      // MS_LOGE("Can't find input tensor layer for graph.");
      return RET_ERROR;
    }
  }

  for (auto iter : caffeInspector.GetGraphOutput()) {
    int index = tensorCache->FindTensor(iter);
    if (index >= 0) {
      subGraphDef->outputIndex.emplace_back(index);
    } else {
      // MS_LOGE("Can't find output tensor layer for graph.");
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::ParseLayer(const caffe::NetParameter &proto, const caffe::NetParameter &weight,
                                    TensorCache *tensorCache, schema::MetaGraphT *subGraphDef) {
  for (int i = 0; i < proto.layer_size(); i++) {
    auto layer = proto.layer(i);

    caffe::LayerParameter layerP;
    for (int j = 0; j < weight.layer_size(); j++) {
      auto tempLayer = weight.layer(j);
      if (tempLayer.name() == layer.name()) {
        layerP = tempLayer;
        break;
      }
    }
    // todo y00520784 : layer.input_param().shape(0)
    if (layer.type() == "Input") {
      std::unique_ptr<schema::TensorT> msTensor(new schema::TensorT());
      for (int j = 0; j < layer.input_param().shape(0).dim_size(); j++) {
        msTensor->dims.push_back(layer.input_param().shape(0).dim(j));
      }
      msTensor->nodeType = schema::NodeType_ValueNode;
      msTensor->refCount = 1;
      msTensor->dataType = kNumberTypeFloat32;
      tensorCache->AddTensor(layer.top(0), msTensor.release(), GRAPH_INPUT);
    } else {
      if (skipedLayerType.find(layer.type()) != skipedLayerType.end()) {
        MS_LOG(INFO) << "Skip layer " << layer.name();
        continue;
      }

      std::unique_ptr<schema::CNodeT> op(new schema::CNodeT());
      op->name = layer.name();

      // set op input index
      auto status = SetOpInputIdx(layer, op.get(), tensorCache);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Set Op " << layer.name() << " Input Index Failed!";
        return status;
      }

      auto nodeParser = CaffeNodeParserRegistry::GetInstance()->GetNodeParser(layer.type().c_str());
      if (nodeParser == nullptr) {
        MS_LOG(ERROR) << "Don't support type " << layer.type() << ". for caffe op " << layer.name();
        return RET_ERROR;
      }

      std::vector<schema::TensorT *> weightVec;
      status = nodeParser->Parse(layer, layerP, op.get(), &weightVec);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Parse weight for " << layer.name() << " Failed!";
        return status;
      }
      // set op weight tensor to tensorcache
      SetWeightTensor(weightVec, op.get(), tensorCache);

      // set op output index
      status = SetOpOutputIdx(layer, op.get(), tensorCache);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Set Op " << layer.name() << " Output Index Failed!";
        return status;
      }

      // op->fmkType = FmkType_CAFFE;
      subGraphDef->nodes.emplace_back(move(op));
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::GetModelInput(const caffe::NetParameter &proto, TensorCache *tensorCache) {
  for (int i = 0; i < proto.input_size(); i++) {
    if (proto.input_dim_size() <= 0) {
      continue;
    }
    std::unique_ptr<schema::TensorT> msTensor(new schema::TensorT());
    for (int j = 0; j < proto.input_dim_size(); j++) {
      msTensor->dims.push_back(proto.input_dim(j));
    }
    msTensor->refCount = schema::NodeType_ValueNode;
    msTensor->dataType = kNumberTypeFloat32;
    tensorCache->AddTensor(proto.input(i), msTensor.release(), GRAPH_INPUT);
  }

  for (int i = 0; i < proto.input_shape_size(); i++) {
    auto shape = proto.input_shape(i);
    std::unique_ptr<schema::TensorT> msTensor(new schema::TensorT());
    for (int j = 0; j < shape.dim_size(); j++) {
      msTensor->dims.push_back(shape.dim(j));
    }
    msTensor->refCount = schema::NodeType_ValueNode;
    msTensor->dataType = kNumberTypeFloat32;
    tensorCache->AddTensor(proto.input(i), msTensor.release(), GRAPH_INPUT);
  }
  return RET_OK;
}

void CaffeModelParser::ConvertCaffeBatchNorm(schema::MetaGraphT *meta_graph) {
  MS_ASSERT(meta_graph != nullptr);
  auto &nodes = meta_graph->nodes;
  for (auto &node : nodes) {
    if (node->primitive->value.type != schema::PrimitiveType_FusedBatchNorm) {
      continue;
    }
    MS_ASSERT(node->inputIndex.size() == 2);
    MS_ASSERT(node->inputIndex.back() < meta_graph->allTensors.size());
    auto &meanTensor = meta_graph->allTensors.at(node->inputIndex.back());
    MS_ASSERT(nullptr != meanTensor);
    auto shape = meanTensor->dims;
    auto shapeSize = GetShapeSize(shape);

    auto scaleTensor = std::make_unique<schema::TensorT>();
    scaleTensor->dims = shape;
    scaleTensor->nodeType = NodeType_ValueNode;
    scaleTensor->refCount = 1;
    scaleTensor->format = schema::Format_NUM_OF_FORMAT;
    scaleTensor->dataType = TypeId::kNumberTypeFloat32;
    scaleTensor->data.resize(shapeSize * sizeof(float));
    auto scaleData = reinterpret_cast<float *>(scaleTensor->data.data());
    for (size_t i = 0; i < shapeSize; i++) {
      scaleData[i] = 1;
    }

    auto biasTensor = std::make_unique<schema::TensorT>();
    biasTensor->dims = shape;
    biasTensor->nodeType = NodeType_ValueNode;
    biasTensor->refCount = 1;
    biasTensor->format = schema::Format_NUM_OF_FORMAT;
    biasTensor->dataType = TypeId::kNumberTypeInt32;
    biasTensor->data.resize(shapeSize * sizeof(int32_t));
    auto biasData = reinterpret_cast<int32_t *>(biasTensor->data.data());
    for (size_t i = 0; i < shapeSize; i++) {
      biasData[i] = 0;
    }

    node->inputIndex.insert(node->inputIndex.begin() + 1, meta_graph->allTensors.size());
    meta_graph->allTensors.emplace_back(std::move(biasTensor));

    node->inputIndex.insert(node->inputIndex.begin() + 1, meta_graph->allTensors.size());
    meta_graph->allTensors.emplace_back(std::move(scaleTensor));
  }
}
}  // namespace lite
}  // namespace mindspore
