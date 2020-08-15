/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/aware_quantizer.h"
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "schema/inner/model_generated.h"
#include "utils/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/utils.h"
#include "tools/converter/quantizer/calc_quant_param.h"
#include "tools/common/tensor_util.h"
#include "tools/common/converter_op_utils.h"
#include "tools/common/node_util.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
struct InputArray {
  std::unique_ptr<QuantParamT> quantParam;
  float mMin = 0.0f;
  float mMax = 0.0f;
  bool narrowRange = false;
  int numBits = 8;
  TypeId dataType = TypeId::kTypeUnknown;

  InputArray(float mean, float stdDev, TypeId dataType = TypeId::kNumberTypeFloat) {
    this->dataType = dataType;
    constexpr float qmin = 0;
    constexpr float qmax = 255;
    mMin = (qmin - mean) / stdDev;
    mMax = (qmax - mean) / stdDev;
  }

  STATUS InitQuantParam() {
    this->quantParam = std::make_unique<schema::QuantParamT>();
    auto status = CalQuantizationParams(quantParam.get(), mMin, mMax, narrowRange, numBits);
    if (status != RET_OK) {
      return status;
    }
    return RET_OK;
  }

  STATUS SetInputArrayQP(schema::MetaGraphT *graph, size_t inputTensorIdx) {
    MS_ASSERT(graph != nullptr);
    auto &tensor = graph->allTensors.at(inputTensorIdx);
    MS_ASSERT(tensor != nullptr);
    if (!tensor->quantParams.empty()) {
      auto param = GetTensorQuantParam(tensor);
      if (param != nullptr && param->inited) {
        MS_LOG(DEBUG) << "tensor " << inputTensorIdx << " already has quantParam";
        return RET_OK;
      }
      tensor->quantParams.clear();
    }
    std::unique_ptr<schema::QuantParamT> tmpQuantParam(new QuantParamT());
    tmpQuantParam->inited = this->quantParam->inited;
    tmpQuantParam->scale = this->quantParam->scale;
    tmpQuantParam->zeroPoint = this->quantParam->zeroPoint;
    tmpQuantParam->min = this->quantParam->min;
    tmpQuantParam->max = this->quantParam->max;
    tensor->quantParams.push_back(std::move(tmpQuantParam));
    return RET_OK;
  }
};

const std::array<schema::PrimitiveType, 7> AwareQuantizer::propagatedOps = {
  {schema::PrimitiveType_Concat, schema::PrimitiveType_Resize, schema::PrimitiveType_Reshape,
   schema::PrimitiveType_Squeeze, schema::PrimitiveType_RealDiv, schema::PrimitiveType_Activation,
   schema::PrimitiveType_DetectionPostProcess}};

AwareQuantizer::AwareQuantizer(schema::MetaGraphT *graph, const string &inputInferType, const string &stdValues,
                               const string &meanValues)
    : FbQuantizer(graph) {
  MS_ASSERT(graph != nullptr);
  string::size_type sz;
  const float stdValue = std::stof(stdValues, &sz);
  sz = 0;
  const float mean = std::stof(meanValues, &sz);
  if (inputInferType == "FLOAT") {
    mInputArray = new InputArray(mean, stdValue);
  } else {
    mInputArray = new InputArray(mean, stdValue, TypeId::kNumberTypeUInt8);
  }
  mInputArray->InitQuantParam();
}

STATUS AwareQuantizer::RemoveFakeQuant() {
  //  for (auto &subGraph : graphDefT->subgraphs) {
  //    auto status = GenerateDefaultQuantParam(subGraph.get());
  //    if (status != RET_OK) {
  //      MS_LOGE("GenerateDefaultQuantParam failed: %d", status);
  //      return RET_ERROR;
  //    }
  //    for (auto iter = subGraph->nodes.begin(); iter != subGraph->nodes.end(); iter++) {
  //      auto *node = (*iter).get();
  //      if (GetCNodeTType(*node) != OpT_FakeQuantWithMinMaxVars && GetCNodeTType(*node) != OpT_FakeQuantWithMinMax) {
  //        continue;
  //      }
  //      auto inputIndexes = node->inputIndex;
  //      if (inputIndexes.size() != 3) {
  //        MS_LOGE("invalid fakequant node's input tensors count!");
  //        return RET_ERROR;
  //      }
  //      bool narrorRange;
  //      int numBits;
  //      if (GetCNodeTType(*node) == OpT_FakeQuantWithMinMaxVars) {
  //        narrorRange = node->attr.AsFakeQuantWithMinMaxVars()->narrowRange;
  //        numBits = node->attr.AsFakeQuantWithMinMaxVars()->numBits;
  //      }
  //      if (GetCNodeTType(*node) == OpT_FakeQuantWithMinMax) {
  //        narrorRange = false;
  //        numBits = 8;
  //      }
  //
  //      TensorDefT *tensor0 = subGraph->allTensors.at(inputIndexes[0]).get();
  //      TensorDefT *tensor1 = subGraph->allTensors.at(inputIndexes[1]).get();
  //      TensorDefT *tensor2 = subGraph->allTensors.at(inputIndexes[2]).get();
  //      MS_ASSERT(tensor0 != nullptr);
  //      MS_ASSERT(tensor1 != nullptr);
  //      MS_ASSERT(tensor2 != nullptr);
  //      // calculate quant param
  //      MS_ASSERT(tensor1->dataType == DataType_DT_FLOAT);
  //      MS_ASSERT(tensor2->dataType == DataType_DT_FLOAT);
  //      auto *minData = reinterpret_cast<const float *>(tensor1->data.data());
  //      auto *maxData = reinterpret_cast<const float *>(tensor2->data.data());
  //      MS_ASSERT(minData != nullptr);
  //      MS_ASSERT(maxData != nullptr);
  //      std::unique_ptr<QuantParamT> quantParam(new (std::nothrow) QuantParamT());
  //      if (quantParam == nullptr) {
  //        MS_LOGE("new quantParam failed");
  //        return RET_ERROR;
  //      }
  //      auto realMin = (double)minData[0];
  //      auto realMax = (double)maxData[0];
  //      status = CalQuantizationParams(quantParam.get(), realMin, realMax, narrorRange, numBits);
  //      if (status != RET_OK) {
  //        MS_LOGE("in aware quantization run CalQuantizationParams failed, node: %s", node->name.c_str());
  //        return RET_ERROR;
  //      }
  //      if (tensor0->refCount == MSCONST_WEIGHT_REFCOUNT) {
  //        CalFakeNode(tensor0, quantParam.get());
  //      }
  //      std::unique_ptr<QuantParamArrayT> quantParamArray(new (std::nothrow) QuantParamArrayT());
  //      if (quantParamArray == nullptr) {
  //        MS_LOGE("new quantParamArray failed");
  //        return RET_ERROR;
  //      }
  //      quantParamArray->param.push_back(std::move(quantParam));
  //      auto quantParamArrayCopy = CopyQuantParamArrayT(quantParamArray);
  //      if (quantParamArrayCopy == nullptr) {
  //        MS_LOGE("CopyQuantParamArray %s return nullptr", iter->get()->name.c_str());
  //        return RET_ERROR;
  //      }
  //      node->quantParam.emplace_back(std::move(quantParamArrayCopy));
  //      node->quantParam.emplace_back(nullptr);  // secondInTensor and thirdInTensor are weightTensors who have no
  //      preNode node->quantParam.emplace_back(nullptr); node->quantParam.emplace_back(std::move(quantParamArray));
  //
  //      // BroadCast fakeQuantNode QuantParam
  //      status = BroadCastQuantParam(subGraph, *iter);
  //      if (status != RET_OK) {
  //        MS_LOGE("BroadCastQuantParam %s failed: %d", iter->get()->name.c_str(), status);
  //        return status;
  //      }
  //      // save post node index for SetAttrToConvolution
  //      auto postNodeIdxes = GetOutputNodeIdx(*subGraph, *node);
  //      // remove fakequantwithminmax node
  //      status = IsolateNode(subGraph.get(), node);
  //      if (status != RET_OK) {
  //        MS_LOGE("in aware quant IsolateNode failed!");
  //        return RET_ERROR;
  //      }
  //      // set filter param to node
  //      if (tensor0->refCount == MSCONST_WEIGHT_REFCOUNT && !postNodeIdxes.empty()) {
  //        auto postNode = subGraph->nodes.at(postNodeIdxes.front()).get();
  //        if (GetCNodeTType(*postNode) == OpT_Conv2D || GetCNodeTType(*postNode) == OpT_DepthwiseConv2D ||
  //            GetCNodeTType(*postNode) == OpT_DeConv2D || GetCNodeTType(*postNode) == OpT_DeDepthwiseConv2D) {
  //          auto status = SetAttrToConvolution(subGraph.get(), postNode);
  //          if (status != RET_OK) {
  //            MS_LOGE("in aware quant SetAttrToConvolution failed!");
  //            return RET_ERROR;
  //          }
  //        }
  //      }
  //    }
  //
  //    // remove IsolatedNode
  //    for (auto iter = subGraph->nodes.begin(); iter != subGraph->nodes.end();) {
  //      if ((*iter)->inputIndex.empty() && (*iter)->outputIndex.empty()) {
  //        iter = subGraph->nodes.erase(iter);
  //      } else {
  //        iter++;
  //      }
  //    }
  //    // set graphInputNode inputTensor quantParams
  //    MS_ASSERT(subGraph->inputIndex.size() == 1);
  //    for (auto graphInputIndex : subGraph->inputIndex) {
  //      auto linkedPostIdx = GetLinkedPostIdx(*(subGraph.get()), graphInputIndex);
  //      for (auto nodeIdx : linkedPostIdx) {
  //        MS_ASSERT(subGraph->nodes.size() > nodeIdx);
  //        mInputArray->SetInputArrayQP(subGraph->nodes.at(nodeIdx).get());
  //      }
  //    }
  //  }
  return RET_OK;
}

STATUS AwareQuantizer::GenerateDefaultQuantParam(const schema::MetaGraphT *subGraph) {
  MS_ASSERT(subGraph != nullptr);
  for (const auto &tensor : subGraph->allTensors) {
    if (!tensor->quantParams.empty()) {
      continue;
    }
    std::unique_ptr<schema::QuantParamT> defaultQuantParam(new QuantParamT());
    tensor->quantParams.emplace_back(std::move(defaultQuantParam));
  }
  return RET_OK;
}

STATUS AwareQuantizer::SetAttrToConvolution(const schema::MetaGraphT *subGraph, schema::CNodeT *node) {
  //  MS_ASSERT(subGraph != nullptr);
  //  MS_ASSERT(node != nullptr);
  //  auto inputIndexes = node->inputIndex;
  //  MS_ASSERT(GetCNodeTType(*node) == OpT_Conv2D || GetCNodeTType(*node) == OpT_DepthwiseConv2D ||
  //            GetCNodeTType(*node) == OpT_DeConv2D || GetCNodeTType(*node) == OpT_DeDepthwiseConv2D);
  //  if (inputIndexes.size() < 2) {
  //    MS_LOGE("in aware quant %s node's input tensors is invalid(%zu)!", node->name.c_str(), inputIndexes.size());
  //    return RET_ERROR;
  //  }
  //  TensorDefT *filterTensor = subGraph->allTensors.at(inputIndexes[1]).get();
  //  MS_ASSERT(filterTensor != nullptr);
  //  auto filterDims = filterTensor->dims;
  //  MS_ASSERT(filterDims.size() == 4);
  //  if (GetCNodeTType(*node) == OpT_Conv2D) {
  //    if (node->fmkType == FmkType_MS) {
  //      node->attr.AsConv2D()->channelOut = (int32_t)filterDims[0];
  //      node->attr.AsConv2D()->channelIn = (int32_t)filterDims[1];
  //      node->attr.AsConv2D()->kernelH = (int32_t)filterDims[2];
  //      node->attr.AsConv2D()->kernelW = (int32_t)filterDims[3];
  //    } else if (node->fmkType == FmkType_TF) {
  //      node->attr.AsConv2D()->kernelH = (int32_t)filterDims[0];
  //      node->attr.AsConv2D()->kernelW = (int32_t)filterDims[1];
  //      node->attr.AsConv2D()->channelIn = (int32_t)filterDims[2];
  //      node->attr.AsConv2D()->channelOut = (int32_t)filterDims[3];
  //    } else {
  //      MS_LOGE("Unsupport");
  //    }
  //  }
  //  if (GetCNodeTType(*node) == OpT_DepthwiseConv2D) {
  //    if (node->fmkType == FmkType_MS) {
  //      node->attr.AsDepthwiseConv2D()->channelIn = (int32_t)filterDims[0];
  //      node->attr.AsDepthwiseConv2D()->channelMultiplier = (int32_t)filterDims[1];
  //      node->attr.AsDepthwiseConv2D()->kernelH = (int32_t)filterDims[2];
  //      node->attr.AsDepthwiseConv2D()->kernelW = (int32_t)filterDims[3];
  //    } else if (node->fmkType == FmkType_TF) {
  //      node->attr.AsDepthwiseConv2D()->kernelH = (int32_t)filterDims[0];
  //      node->attr.AsDepthwiseConv2D()->kernelW = (int32_t)filterDims[1];
  //      node->attr.AsDepthwiseConv2D()->channelIn = (int32_t)filterDims[2];
  //      node->attr.AsDepthwiseConv2D()->channelMultiplier = (int32_t)filterDims[3];
  //    } else {
  //      MS_LOGE("Unsupport");
  //    }
  //  }
  //  if (GetCNodeTType(*node) == OpT_DeConv2D) {
  //    MS_ASSERT(false);
  //  }
  //  if (GetCNodeTType(*node) == OpT_DeDepthwiseConv2D) {
  //    MS_ASSERT(false);
  //  }
  return RET_OK;
}

STATUS AwareQuantizer::GenerateQuantParam() {
  // todo why?
  MS_ASSERT(graph->inputIndex.size() == 1);
  // set graphInputNode input
  for (auto graphInputIndex : graph->inputIndex) {
    auto status = mInputArray->SetInputArrayQP(graph, graphInputIndex);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "SetInputArrayQP failed";
      return status;
    }
  }
  auto status = GenerateDefaultQuantParam(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GenerateDefaultQuantParam failed";
    return status;
  }
  auto *quantParamRegister = QuantParamCalcRegister::GetInstance();

  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    MS_ASSERT(node != nullptr);
    if (GetCNodeTType(*node) == schema::PrimitiveType_FakeQuantWithMinMax ||
        GetCNodeTType(*node) == schema::PrimitiveType_FakeQuantWithMinMaxVars) {
      MS_ASSERT(false);
    }
    auto *quantParamCalcer = quantParamRegister->GetQuantParamCalcer(GetCNodeTType(*node));
    if (quantParamCalcer == nullptr) {
      MS_LOG(ERROR) << "Can not find QuantParamCalcer for " << node->name.c_str()
                    << ", type: " << GetCNodeTTypeName(*node).c_str() << " set node to QuantNone and skip";
      node->quantType = static_cast<schema::QuantType>(QuantType_QUANT_NONE);
    } else {
      status = quantParamCalcer->Calc(graph, *node);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "quantParamCalcer failed: " << status << " node: " << node->name.c_str();
        node->quantType = schema::QuantType_QUANT_NONE;
      } else {
        node->quantType = schema::QuantType_AwareTraining;
      }
    }
  }
  return RET_OK;
}

STATUS AwareQuantizer::DoQuantize() {
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    if (!IsContain(GetUint8OpList(), GetCNodeTType(*node))) {
      continue;
    }
    if (node->quantType != schema::QuantType_AwareTraining) {
      continue;
    }
    STATUS status;
    if (GetCNodeTType(*node) == schema::PrimitiveType_Conv2D ||
        GetCNodeTType(*node) == schema::PrimitiveType_DepthwiseConv2D) {
      auto inputIndexes = node->inputIndex;
      if (inputIndexes.size() < 2) {
        MS_LOG(ERROR) << node->name.c_str() << " node input has invalid inputs tensor count";
        return RET_ERROR;
      }
      // quant weight
      status = QuantConvWeight(graph, node.get());
      if (status != RET_OK) {
        MS_LOG(ERROR) << "QuantConvWeight failed!";
        return RET_ERROR;
      }
      // quant bias
      if (inputIndexes.size() == 3) {
        status = QuantConvBias(graph, node.get());
        if (status != RET_OK) {
          MS_LOG(ERROR) << "QuantConvBias failed!";
          return RET_ERROR;
        }
      }
    } else if (GetCNodeTType(*node) == schema::PrimitiveType_DetectionPostProcess) {
      status = QuantDetectionPostProcessConstTensor(graph, node.get());
      if (status != RET_OK) {
        MS_LOG(ERROR) << "QuantDetectionPostProcessConstTensor failed!";
        return RET_ERROR;
      }
    } else if (GetCNodeTType(*node) == schema::PrimitiveType_Add) {
      status = QuantAddConstTensor(graph, node.get());
      if (status != RET_OK) {
        MS_LOG(ERROR) << "QuantAddConstTensor failed!";
        return RET_ERROR;
      }
    }
    const auto nodeType = GetCNodeTType(*node);
    auto find = std::find(propagatedOps.begin(), propagatedOps.end(), nodeType);
    if (find != propagatedOps.end()) {
      auto inputTensor = graph->allTensors.at(node->inputIndex[0]).get();
      auto outputTensor = graph->allTensors.at(node->outputIndex[0]).get();
      MS_ASSERT(inputTensor != nullptr);
      MS_ASSERT(outputTensor != nullptr);
      outputTensor->dataType = inputTensor->dataType;
    }
  }
  return RET_OK;
}

STATUS AwareQuantizer::QuantAddConstTensor(const schema::MetaGraphT *graph, schema::CNodeT *node) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(node != nullptr);
  for (size_t i = 0; i < node->inputIndex.size(); i++) {
    auto inTensorIdx = node->inputIndex.at(i);
    MS_ASSERT(graph->allTensors.size() > inTensorIdx);
    auto &inTensor = graph->allTensors.at(inTensorIdx);
    MS_ASSERT(inTensor != nullptr);
    if (inTensor->refCount == 999) {
      switch (inTensor->dataType) {
        case TypeId::kNumberTypeFloat: {
          auto quantParam = GetTensorQuantParam(inTensor);
          MS_ASSERT(quantParam != nullptr);
          MS_ASSERT(quantParam->inited);
          auto constTensorShapeSize = GetShapeSize(*(inTensor.get()));
          vector<uint8_t> qDatas(constTensorShapeSize);
          void *inData = inTensor->data.data();
          auto *castedInData = static_cast<float *>(inData);
          for (size_t j = 0; j < constTensorShapeSize; j++) {
            qDatas[j] = QuantizeData<uint8_t>(castedInData[j], quantParam.get());
          }
          inTensor->data = std::move(qDatas);
          inTensor->dataType = kNumberTypeUInt8;
        } break;
        case kNumberTypeUInt8:
          break;
        default:
          //          MS_LOGE("Unsupported dataType: %d", inTensor->dataType);
          return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS AwareQuantizer::QuantDetectionPostProcessConstTensor(const schema::MetaGraphT *subGraph, schema::CNodeT *node) {
  MS_ASSERT(subGraph != nullptr);
  MS_ASSERT(node != nullptr);
  auto &constTensor = subGraph->allTensors.at(node->inputIndex[2]);
  MS_ASSERT(constTensor != nullptr);
  const auto *constData = reinterpret_cast<const float *>(constTensor->data.data());

  if (constTensor->refCount == 999 && constTensor->dataType == TypeId::kNumberTypeFloat) {
    size_t constTensorShapeSize = GetShapeSize(*constTensor);
    std::unique_ptr<QuantParamT> quantParam = GetTensorQuantParam(constTensor);
    if (quantParam == nullptr) {
      //    MS_LOGE("new QuantParamT failed");
      return RET_NULL_PTR;
    }
    vector<uint8_t> qDatas(constTensorShapeSize);
    for (size_t j = 0; j < constTensorShapeSize; j++) {
      float rawData = constData[j];
      qDatas[j] = QuantizeData<uint8_t>(rawData, quantParam.get());
    }
    constTensor->data = std::move(qDatas);
    constTensor->dataType = TypeId::kNumberTypeUInt8;
  }
  return RET_OK;
}

STATUS AwareQuantizer::QuantConvBias(const mindspore::schema::MetaGraphT *graph, mindspore::schema::CNodeT *node) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto inputIndexes = node->inputIndex;
  MS_ASSERT(inputIndexes.size() >= 3);
  MS_ASSERT(graph->allTensors.size() > inputIndexes.at(0));
  MS_ASSERT(graph->allTensors.size() > inputIndexes.at(1));
  MS_ASSERT(graph->allTensors.size() > inputIndexes.at(2));
  auto &biasTensor = graph->allTensors.at(inputIndexes.at(2));
  MS_ASSERT(biasTensor != nullptr);
  if (biasTensor->dataType != TypeId::kNumberTypeFloat) {
    //    MS_LOGD("conv %s's bias data is not float", node->name.c_str());
    return RET_OK;
  }

  if (biasTensor->dataType == TypeId::kNumberTypeInt32) {
    return RET_OK;
  }
  if (biasTensor->dataType != TypeId::kNumberTypeFloat) {
    //    MS_LOGE("conv %s's bias data is not float", node->name.c_str());
    return RET_ERROR;
  }
  auto &inputTensor = graph->allTensors.at(inputIndexes.at(0));
  auto &weightTensor = graph->allTensors.at(inputIndexes.at(1));

  MS_ASSERT(inputTensor != nullptr);
  MS_ASSERT(weightTensor != nullptr);
  auto inputScale = inputTensor->quantParams.front()->scale;
  auto weightScale = weightTensor->quantParams.front()->scale;
  auto scale = inputScale * weightScale;
  // set bias quant param
  std::unique_ptr<QuantParamT> biasQuantParam = GetTensorQuantParam(biasTensor);
  if (biasQuantParam == nullptr) {
    //    MS_LOGE("new QuantParamT failed");
    return RET_ERROR;
  }
  biasQuantParam->inited = true;
  biasQuantParam->scale = scale;
  biasQuantParam->zeroPoint = 0;
  biasQuantParam->numBits = 8;
  biasQuantParam->narrowRange = false;
  biasQuantParam->min = 0.0;
  biasQuantParam->max = 0.0;

  // quant bias data
  auto bShapeSize = GetShapeSize(*(biasTensor.get()));
  auto *qDatas = new (std::nothrow) int32_t[bShapeSize];
  if (qDatas == nullptr) {
    //    MS_LOGE("new qDatas failed");
    return RET_ERROR;
  }
  void *biasData = biasTensor->data.data();
  auto *rawDatas = static_cast<float *>(biasData);
  for (size_t i = 0; i < bShapeSize; ++i) {
    qDatas[i] = (int32_t)std::round(rawDatas[i] / scale);
  }
  biasTensor->dataType = TypeId::kNumberTypeInt32;
  biasTensor->data.clear();
  biasTensor->data.resize(bShapeSize * sizeof(int32_t));
  auto ret = memcpy_s(biasTensor->data.data(), bShapeSize * sizeof(int32_t), qDatas, bShapeSize * sizeof(int32_t));
  if (ret != EOK) {
    //    MS_LOGE("memcpy_s failed: %d", ret);
    return RET_ERROR;
  }
  delete[] qDatas;
  return RET_OK;
}

STATUS AwareQuantizer::QuantConvWeight(const schema::MetaGraphT *subGraph, schema::CNodeT *node) {
  MS_ASSERT(subGraph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(node->quantParam.size() == node->inputIndex.size() + node->outputIndex.size());
  auto inputIndexes = node->inputIndex;
  MS_ASSERT(inputIndexes.size() >= 2);
  MS_ASSERT(subGraph->allTensors.size() > inputIndexes.at(1));
  auto &weightTensor = subGraph->allTensors.at(inputIndexes.at(1));
  if (weightTensor->dataType == TypeId::kNumberTypeInt8) {
    return RET_OK;
  }
  if (weightTensor->dataType != TypeId::kNumberTypeFloat && weightTensor->dataType != TypeId::kNumberTypeUInt8) {
    MS_LOG(ERROR) << "conv " << node->name.c_str() << "'s weight data is not float or uint8";
    return RET_ERROR;
  }
  size_t wShapeSize = GetShapeSize(*(weightTensor.get()));
  void *oriWeightData = weightTensor->data.data();
  MS_ASSERT(node->quantParam.at(1)->param.front() != nullptr);
  vector<int8_t> qDatas(wShapeSize);
  auto weightQauntParam = GetTensorQuantParam(weightTensor);
  if (weightTensor->dataType == TypeId::kNumberTypeFloat) {  // normal awareing quant
    auto *weightData = static_cast<float *>(oriWeightData);
    for (size_t j = 0; j < wShapeSize; j++) {
      qDatas[j] = QuantizeData<int8_t>(weightData[j], weightQauntParam.get());
    }
  } else {  // tflite awareing quant
    auto *weightData = static_cast<uint8_t *>(oriWeightData);
    for (size_t j = 0; j < wShapeSize; j++) {
      qDatas[j] = (int32_t)weightData[j] - 128;
    }
    weightQauntParam->zeroPoint -= 128;
    weightTensor->quantParams.clear();
    weightTensor->quantParams.emplace_back(weightQauntParam.release());
  }

  ::memcpy(weightTensor->data.data(), qDatas.data(), wShapeSize);
  weightTensor->dataType = TypeId::kNumberTypeInt8;
  return RET_OK;
}
STATUS AwareQuantizer::DetermineNodeQuantType() {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    MS_ASSERT(node != nullptr);
    bool canQuant = true;
    for (auto &inTensorIdx : node->inputIndex) {
      MS_ASSERT(graph->allTensors.size() > inTensorIdx);
      auto &inTensor = graph->allTensors.at(inTensorIdx);
      MS_ASSERT(inTensor != nullptr);
      if (inTensor->quantParams.empty() || inTensor->quantParams.front() == nullptr ||
          !inTensor->quantParams.front()->inited) {
        canQuant = false;
        break;
      }
    }

    if (canQuant) {
      for (auto &outTensorIdx : node->outputIndex) {
        MS_ASSERT(graph->allTensors.size() > outTensorIdx);
        auto &outTensor = graph->allTensors.at(outTensorIdx);
        MS_ASSERT(outTensor != nullptr);
        if (outTensor->quantParams.empty() || outTensor->quantParams.front() == nullptr ||
            !outTensor->quantParams.front()->inited) {
          canQuant = false;
          break;
        }
      }
    }
    if (canQuant && IsContain(GetUint8OpList(), GetCNodeTType(*node))) {
      node->quantType = schema::QuantType_AwareTraining;
    } else {
      node->quantType = schema::QuantType_QUANT_NONE;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
