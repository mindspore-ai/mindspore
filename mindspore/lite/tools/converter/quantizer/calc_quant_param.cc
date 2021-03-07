/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/calc_quant_param.h"
#include <cfloat>
#include <memory>
#include <algorithm>
#include <utility>
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "schema/inner/ops_generated.h"
#include "src/common/utils.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite {
static constexpr size_t BIAS_SIZE = 3;
static constexpr size_t BIAS_ADD_SIZE = 2;

STATUS QuantParamCalcer::ComputeConstQuantParam(const schema::TensorT &tensor, QuantParamT *quantParam) {
  MS_ASSERT(quantParam != nullptr);
  // int32 weight no need to quant
  if (tensor.dataType == TypeId::kNumberTypeInt32 || tensor.dataType == TypeId::kNumberTypeUInt8) {
    return RET_OK;
  }
  if (tensor.dataType != TypeId::kNumberTypeFloat) {
    MS_LOG(DEBUG) << "Const Tensor without quantParam should has float dataType, in fact: " << tensor.dataType;
    return RET_ERROR;
  }
  const auto *constData = reinterpret_cast<const float *>(tensor.data.data());
  size_t constTensorShapeSize = GetShapeSize(tensor);
  float min = 0.0f;
  float max = 0.0f;
  // find min and max
  for (size_t i = 0; i < constTensorShapeSize; i++) {
    min = std::min(min, constData[i]);
    max = std::max(max, constData[i]);
  }
  if (min == 0.0f && max == 0.0f) {
    max = 1.0f;
  }
  bool isQuantExact = true;
  for (size_t i = 0; i < constTensorShapeSize; i++) {
    isQuantExact &= (constData[i] == min || constData[i] == max);
  }
  if (!isQuantExact) {
    MS_LOG(DEBUG) << "compute quantParam for const tensor may be a cause of poor inference accuracy";
  }
  return quant::CalQuantizationParams(quantParam, min, max);
}

// init inTensor quantParam from preNode if possible
// init outTensor quantParam from postNode if possible
int QuantParamCalcer::Calc(MetaGraphT *graph, const CNodeT &node) {
  MS_ASSERT(node.inputIndex.size() > 0);
  MS_ASSERT(node.quantParam.size() == node.inputIndex.size() + node.outputIndex.size());
  inputParamDone = 0;
  auto inputTensorSize = node.inputIndex.size();
  for (size_t i = 0; i < inputTensorSize; i++) {
    MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(i));
    auto &tensor = graph->allTensors.at(node.inputIndex.at(i));
    MS_ASSERT(tensor != nullptr);
    auto quantParam = GetTensorQuantParam(tensor);
    if (quantParam == nullptr) {
      continue;
    }
    if (quantParam->inited) {  // inited
      inputParamDone++;
      continue;
    }
    if (!tensor->data.empty() && !IsContain(graph->inputIndex, node.inputIndex.at(i))) {
      auto status = ComputeConstQuantParam((*tensor), quantParam.get());
      if (status != RET_OK) {
        MS_LOG(DEBUG) << "ComputeConstQuantParam failed: " << status;
        return status;
      }
      tensor->quantParams.front() = std::move(quantParam);
      inputParamDone++;
      continue;
    }
  }
  outputParamDone = 0;
  for (unsigned int i : node.outputIndex) {
    MS_ASSERT(graph->allTensors.size() > i);
    auto &tensor = graph->allTensors.at(i);
    MS_ASSERT(tensor != nullptr);
    auto quantParam = GetTensorQuantParam(tensor);
    if (quantParam != nullptr && quantParam->inited) {  // inited
      outputParamDone++;
      continue;
    }
    MS_ASSERT(tensor->data.empty());
  }
  return RET_OK;
}

int CommonCalcer::Calc(MetaGraphT *subGraph, const CNodeT &node) {
  auto status = QuantParamCalcer::Calc(subGraph, node);
  if (status != RET_OK) {
    MS_LOG(DEBUG) << "Call QuantParamCalcer::Calc failed: " << status;
    return status;
  }
  if (inputParamDone != node.inputIndex.size()) {
    MS_LOG(DEBUG) << "Can not determine inputTensor quantParam, node " << node.name;
    return RET_ERROR;
  }
  if (outputParamDone != node.outputIndex.size()) {
    MS_LOG(DEBUG) << "Can not determine outputTensor quantParam, node " << node.name;
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvCalcer::Calc(MetaGraphT *subGraph, const CNodeT &node) {
  auto status = CommonCalcer::Calc(subGraph, node);
  if (status != RET_OK) {
    MS_LOG(DEBUG) << "Call CommonCalcer::Calc failed: " << status;
    return status;
  }
  if (node.inputIndex.size() == BIAS_SIZE) {
    auto &biasTensor = subGraph->allTensors.at(node.inputIndex.at(BIAS_SIZE - 1));
    for (auto &quantParam : biasTensor->quantParams) {
      quantParam->dstDtype = TypeId::kNumberTypeInt32;
    }
  }
  return RET_OK;
}

int BiasAddCalcer::Calc(MetaGraphT *subGraph, const CNodeT &node) {
  auto status = CommonCalcer::Calc(subGraph, node);
  if (status != RET_OK) {
    MS_LOG(DEBUG) << "Call CommonCalcer::Calc failed: " << status;
    return status;
  }
  if (node.inputIndex.size() == BIAS_ADD_SIZE) {
    auto &biasTensor = subGraph->allTensors.at(node.inputIndex.at(BIAS_ADD_SIZE - 1));
    for (auto &quantParam : biasTensor->quantParams) {
      quantParam->dstDtype = TypeId::kNumberTypeInt32;
    }
  }
  return RET_OK;
}

int LinearCalcer::Calc(MetaGraphT *graph, const CNodeT &node) {
  auto status = QuantParamCalcer::Calc(graph, node);
  if (status != RET_OK) {
    MS_LOG(DEBUG) << "Call QuantParamCalcer::Calc failed: " << status;
    return status;
  }
  if (inputParamDone != node.inputIndex.size()) {
    MS_ASSERT(graph->allTensors.size() > node.outputIndex.at(0));
    auto &outTensor = graph->allTensors.at(node.outputIndex.at(0));
    MS_ASSERT(outTensor != nullptr);
    auto outputQuantParam = GetTensorQuantParam(outTensor);
    MS_ASSERT(outputQuantParam != nullptr);
    if (outputQuantParam == nullptr || !outputQuantParam->inited) {
      MS_LOG(DEBUG) << "Can not determine inputTensor quantParam from outputTensor for node " << node.name;
      return RET_ERROR;
    }
    for (unsigned int i : node.inputIndex) {
      MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(i));
      auto &inTensor = graph->allTensors.at(i);
      MS_ASSERT(inTensor != nullptr);
      auto inQuantParam = GetTensorQuantParam(inTensor);
      if (inQuantParam == nullptr || inQuantParam->inited) {
        continue;
      }
      inTensor->quantParams.front() = std::move(outputQuantParam);
    }
  }
  if (outputParamDone != node.outputIndex.size()) {
    MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(0));
    auto &inTensor = graph->allTensors.at(node.inputIndex.at(0));
    MS_ASSERT(inTensor != nullptr);
    auto inQuantParam = GetTensorQuantParam(inTensor);
    if (inQuantParam == nullptr || !inQuantParam->inited) {
      MS_LOG(DEBUG) << "Can not determine outputTensor quantParam from inputTensor for node %s" << node.name;
      return RET_ERROR;
    }
    for (size_t i = 0; i < node.outputIndex.size(); i++) {
      MS_ASSERT(graph->allTensors.size() > node.outputIndex.at(i));
      auto &outTensor = graph->allTensors.at(node.outputIndex.at(i));
      MS_ASSERT(outTensor != nullptr);
      auto outQuantParam = GetTensorQuantParam(outTensor);
      if (outQuantParam == nullptr) {
        outTensor->quantParams.emplace_back(std::move(inQuantParam));
        continue;
      }
      if (outQuantParam->inited) {
        continue;
      }
      outTensor->quantParams.front() = std::move(inQuantParam);
    }
  }
  return RET_OK;
}

class CalcConcat : public QuantParamCalcer {
 public:
  CalcConcat() = default;
  ~CalcConcat() override = default;

  int Calc(MetaGraphT *graph, const CNodeT &node) override {
    MS_ASSERT(node.outputIndex.size() == 1);
    auto status = QuantParamCalcer::Calc(graph, node);
    if (status != RET_OK) {
      MS_LOG(DEBUG) << "Call QuantParamCalcer::Calc failed: " << status;
      return status;
    }

    if (inputParamDone != node.inputIndex.size()) {
      MS_LOG(DEBUG) << "Can not determine concat inputTensor quantParam, node " << node.name;
      return RET_ERROR;
    }

    if (outputParamDone != 1) {
      MS_ASSERT(outputParamDone == 0);
      float minMin = FLT_MAX;
      float maxMax = FLT_MIN;
      bool narrowRange = false;
      int numBits = -1;
      for (size_t i = 0; i < node.inputIndex.size(); i++) {
        MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(i));
        auto &inTensor = graph->allTensors.at(i);
        MS_ASSERT(inTensor != nullptr);
        auto inQuantParam = GetTensorQuantParam(inTensor);
        if (inQuantParam == nullptr || !inQuantParam->inited) {
          return RET_ERROR;
        }
        if (numBits == -1) {
          narrowRange = inQuantParam->narrowRange;
          numBits = inQuantParam->numBits;
        } else {
          MS_ASSERT(narrowRange == quantParam->narrowRange);
          MS_ASSERT(numBits == quantParam->numBits);
        }
        if (minMin > inQuantParam->min) {
          minMin = inQuantParam->min;
        }
        if (maxMax < inQuantParam->max) {
          maxMax = inQuantParam->max;
        }
      }

      MS_ASSERT(graph->allTensors.size() > node.outputIndex.front());
      auto &outTensor = graph->allTensors.at(node.outputIndex.front());
      MS_ASSERT(outTensor != nullptr);
      auto outQuantParam = std::make_unique<QuantParamT>();

      status = quant::CalQuantizationParams(outQuantParam.get(), minMin, maxMax, narrowRange, numBits);
      if (status != RET_OK) {
        MS_LOG(DEBUG) << "in aware quantization run CalQuantizationParams failed!";
        return RET_ERROR;
      }
      outTensor->quantParams.emplace_back(std::move(outQuantParam));
      outputParamDone++;
    }

    return RET_OK;
  }
};

class CalcAdd : public QuantParamCalcer {
 public:
  CalcAdd() = default;
  ~CalcAdd() override = default;

  int Calc(MetaGraphT *graph, const CNodeT &node) override {
    MS_ASSERT(node.inputIndex.size() == 2);
    MS_ASSERT(node.outputIndex.size() == 1);
    auto status = QuantParamCalcer::Calc(graph, node);
    if (status != RET_OK) {
      MS_LOG(DEBUG) << "Call QuantParamCalcer::Calc failed: " << status;
      return status;
    }

    if (inputParamDone != 2) {
      MS_LOG(DEBUG) << "Can not determine add inputTensor quantParam, node " << node.name;
      return RET_ERROR;
    }
    if (outputParamDone != 1) {
      MS_ASSERT(outputParamDone == 0);
      MS_ASSERT(graph->allTensors.size() > node.outputIndex.front());
      auto &outTensor = graph->allTensors.at(node.outputIndex.front());
      MS_ASSERT(outTensor != nullptr);
      auto outQuantParam = std::make_unique<QuantParamT>();

      MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(0));
      auto &tensor0 = graph->allTensors.at(node.inputIndex.at(0));
      MS_ASSERT(tensor0 != nullptr);
      MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(1));
      auto &tensor1 = graph->allTensors.at(node.inputIndex.at(1));
      MS_ASSERT(tensor1 != nullptr);
      auto biasTensor = &tensor0;
      auto paramTensor = &tensor1;
      if (!tensor0->data.empty() && (tensor0->dims.empty() || tensor0->dims.size() == 1)) {
        biasTensor = &tensor0;
        paramTensor = &tensor1;
      } else if (!tensor1->data.empty() && (tensor1->dims.empty() || tensor1->dims.size() == 1)) {
        biasTensor = &tensor1;
        paramTensor = &tensor0;
      } else {
        MS_LOG(DEBUG) << "Can not determine add outputTensor quantParam, node " << node.name;
        return RET_ERROR;
      }
      auto quantParam = GetTensorQuantParam(*paramTensor);
      MS_ASSERT(quantParam != nullptr);
      MS_ASSERT(quantParam->inited);
      auto min = quantParam->min;
      auto max = quantParam->max;
      {
        if ((*biasTensor)->dataType == TypeId::kNumberTypeFloat) {
          MS_ASSERT((*biasTensor)->data.size() == sizeof(float) / sizeof(uint8_t));
          void *oriTensorData = (*biasTensor)->data.data();
          auto *bias = static_cast<float *>(oriTensorData);
          status = quant::CalQuantizationParams(outQuantParam.get(), min + (*bias), max + (*bias));
          if (status != RET_OK) {
            MS_LOG(DEBUG) << "in aware quantization run CalQuantizationParams failed!";
            return RET_ERROR;
          }
        } else if ((*biasTensor)->dataType == TypeId::kNumberTypeUInt8) {
          MS_ASSERT((*biasTensor)->data.size() == 1);
          void *oriTensorData = (*biasTensor)->data.data();
          auto *bias = static_cast<uint8_t *>(oriTensorData);
          status = quant::CalQuantizationParams(outQuantParam.get(), min + (*bias), max + (*bias));
          if (status != RET_OK) {
            MS_LOG(DEBUG) << "in aware quantization run CalQuantizationParams failed!";
            return RET_ERROR;
          }
        } else {
          MS_LOG(DEBUG) << "Unsupported tensor dataType: " << (*biasTensor)->dataType;
          return RET_ERROR;
        }
      }
      outTensor->quantParams.front() = std::move(outQuantParam);
    }
    return RET_OK;
  }
};

class CalcRealDiv : public QuantParamCalcer {
 public:
  CalcRealDiv() = default;
  ~CalcRealDiv() override = default;

  int Calc(MetaGraphT *graph, const CNodeT &node) override {
    MS_ASSERT(node.inputIndex.size() == 2);
    MS_ASSERT(node.outputIndex.size() == 1);
    auto status = QuantParamCalcer::Calc(graph, node);
    if (status != RET_OK) {
      MS_LOG(DEBUG) << "Call QuantParamCalcer::Calc failed: " << status;
      return status;
    }

    if (inputParamDone != 2) {
      MS_LOG(DEBUG) << "Can not determine realdiv inputTensor quantParam, node " << node.name;
      return RET_ERROR;
    }
    if (outputParamDone != 1) {
      MS_ASSERT(outputParamDone == 0);
      MS_ASSERT(graph->allTensors.size() > node.outputIndex.front());
      auto &outTensor = graph->allTensors.at(node.outputIndex.front());
      MS_ASSERT(outTensor != nullptr);
      auto outQuantParam = std::make_unique<QuantParamT>();

      MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(0));
      MS_ASSERT(graph->allTensors.size() > node.inputIndex.at(1));
      auto &tensor1 = graph->allTensors.at(node.inputIndex.at(1));
      MS_ASSERT(tensor1 != nullptr);
      if (!tensor1->data.empty() && (tensor1->dims.empty() || tensor1->dims.size() == 1)) {
        auto quantParam = GetTensorQuantParam(tensor1);
        auto min = quantParam->min;
        auto max = quantParam->max;
        {
          if (tensor1->dataType == TypeId::kNumberTypeFloat) {
            MS_ASSERT(tensor1->data.size() == sizeof(float) / sizeof(uint8_t));
            void *oriTensorData = tensor1->data.data();
            auto *div = static_cast<float *>(oriTensorData);
            MS_ASSERT(*div != 0);
            status = quant::CalQuantizationParams(outQuantParam.get(), min / (*div), max / (*div));
            if (status != RET_OK) {
              MS_LOG(DEBUG) << "in aware quantization run CalQuantizationParams failed!";
              return RET_ERROR;
            }
          } else if (tensor1->dataType == TypeId::kNumberTypeUInt8) {
            MS_ASSERT(tensor1->data.size() == 1);
            void *oriTensorData = tensor1->data.data();
            auto *div = static_cast<uint8_t *>(oriTensorData);
            status = quant::CalQuantizationParams(outQuantParam.get(), min / (*div), max + (*div));
            if (status != RET_OK) {
              MS_LOG(DEBUG) << "in aware quantization run CalQuantizationParams failed!";
              return RET_ERROR;
            }
          } else {
            MS_LOG(DEBUG) << "Unsupported tensor dataType: " << tensor1->dataType;
            return RET_ERROR;
          }
          outTensor->quantParams.front() = std::move(outQuantParam);
        }
      } else {
        MS_LOG(DEBUG) << "Can not determine realDiv outputTensor quantParam, node " << node.name;
        return RET_ERROR;
      }
    }
    return RET_OK;
  }
};

class CalcToSet : public QuantParamCalcer {
 public:
  CalcToSet(float min, float max) : min(min), max(max) {}
  ~CalcToSet() override = default;

  int Calc(MetaGraphT *graph, const CNodeT &node) override {
    MS_ASSERT(node.inputIndex.size() == 1);
    MS_ASSERT(node.outputIndex.size() == 1);
    auto status = QuantParamCalcer::Calc(graph, node);
    if (status != RET_OK) {
      MS_LOG(DEBUG) << "Call QuantParamCalcer::Calc failed: %d" << status;
      return status;
    }
    // input
    if (inputParamDone != node.inputIndex.size()) {
      MS_LOG(DEBUG) << "Can not determine inputTensor quantParam, node " << node.name;
      return RET_ERROR;
    }
    // output
    if (outputParamDone != node.outputIndex.size()) {
      std::unique_ptr<QuantParamT> quantParam = std::make_unique<QuantParamT>();
      if (quantParam == nullptr) {
        MS_LOG(DEBUG) << "new QuantParamT failed";
        return RET_ERROR;
      }
      quantParam->scale = (max - min) / 256;
      MS_ASSERT(quantParam->scale != 0);
      quantParam->zeroPoint = int32_t(std::round(256 - max / quantParam->scale));
      quantParam->min = min;
      quantParam->max = max;
      quantParam->inited = true;
      MS_ASSERT(graph->allTensors.size() > node.outputIndex.front());
      auto &outTensor = graph->allTensors.at(node.outputIndex.front());
      MS_ASSERT(outTensor != nullptr);
      outTensor->quantParams.emplace_back(std::move(quantParam));
      outputParamDone++;
    }
    return RET_OK;
  }

 protected:
  float min;
  float max;
};

class CalcActivation : public QuantParamCalcer {
 public:
  CalcActivation() = default;
  ~CalcActivation() override = default;

  int Calc(MetaGraphT *subGraph, const CNodeT &node) override {
    MS_ASSERT(node.inputIndex.size() == 1);
    MS_ASSERT(node.outputIndex.size() == 1);
    MS_ASSERT(node.attr.AsActivation() != nullptr);
    if (node.primitive->value.AsActivation()->activation_type == schema::ActivationType_SIGMOID) {
      auto calcToSet = CalcToSet(0, 1);
      return calcToSet.Calc(subGraph, node);
    } else {
      auto calCommon = CommonCalcer();
      return calCommon.Calc(subGraph, node);
    }
  }
};
QuantParamCalcRegister::~QuantParamCalcRegister() = default;

QuantParamCalcRegister::QuantParamCalcRegister() {
  bool hasError = false;
  std::shared_ptr<QuantParamCalcer> baseCalcer = std::make_shared<QuantParamCalcer>();
  if (baseCalcer == nullptr) {
    MS_LOG(DEBUG) << "new QuantParamCalcer failed";
    hasError = true;
  }
  std::shared_ptr<QuantParamCalcer> commonCalcer = std::make_shared<CommonCalcer>();
  if (commonCalcer == nullptr) {
    MS_LOG(DEBUG) << "new commonCalcer failed";
    hasError = true;
  }

  std::shared_ptr<QuantParamCalcer> linearCalcer = std::make_shared<LinearCalcer>();
  if (linearCalcer == nullptr) {
    MS_LOG(DEBUG) << "new linearCalcer failed";
    hasError = true;
  }
  if (!hasError) {
    _registerMap[schema::PrimitiveType_Concat] = std::make_shared<CalcConcat>();
    _registerMap[schema::PrimitiveType_Activation] = std::make_shared<CalcActivation>();
    _registerMap[schema::PrimitiveType_AddFusion] = std::make_shared<CalcAdd>();
    _registerMap[schema::PrimitiveType_MulFusion] = commonCalcer;
    _registerMap[schema::PrimitiveType_ScaleFusion] = std::make_shared<ConvCalcer>();
    _registerMap[schema::PrimitiveType_Conv2DFusion] = std::make_shared<ConvCalcer>();
    _registerMap[schema::PrimitiveType_Conv2dTransposeFusion] = std::make_shared<ConvCalcer>();
    _registerMap[schema::PrimitiveType_AvgPoolFusion] = linearCalcer;
    _registerMap[schema::PrimitiveType_MaxPoolFusion] = linearCalcer;
    _registerMap[schema::PrimitiveType_Resize] = linearCalcer;
    _registerMap[schema::PrimitiveType_Reshape] = linearCalcer;
    _registerMap[schema::PrimitiveType_StridedSlice] = linearCalcer;
    _registerMap[schema::PrimitiveType_Shape] = linearCalcer;
    _registerMap[schema::PrimitiveType_Softmax] = std::make_shared<CalcToSet>(0, 1);
    _registerMap[schema::PrimitiveType_Squeeze] = linearCalcer;
    _registerMap[schema::PrimitiveType_RealDiv] = std::make_shared<CalcRealDiv>();
    _registerMap[schema::PrimitiveType_ReduceFusion] = commonCalcer;
    _registerMap[schema::PrimitiveType_BiasAdd] = std::make_shared<BiasAddCalcer>();
    _registerMap[schema::PrimitiveType_Transpose] = linearCalcer;
    _registerMap[schema::PrimitiveType_MatMul] = std::make_shared<ConvCalcer>();
    _registerMap[schema::PrimitiveType_FullConnection] = std::make_shared<ConvCalcer>();
    // detection_postprocess op's quant param will not infer only fetch from preNode or postNode
    // because we will not insert quantTransNode after this node in tflite_graph_8bit model if input data is float.
    // if quantTransNode is inserted after detection_postprocess node, there will be some errors
    _registerMap[schema::PrimitiveType_DetectionPostProcess] = baseCalcer;
  }
}

QuantParamCalcRegister *QuantParamCalcRegister::GetInstance() {
  static QuantParamCalcRegister instance;
  return &instance;
}

std::shared_ptr<QuantParamCalcer> QuantParamCalcRegister::GetQuantParamCalcer(schema::PrimitiveType opType) {
  auto it = _registerMap.find(opType);
  if (it != _registerMap.end()) {
    return it->second;
  }
  return nullptr;
}
}  // namespace mindspore::lite
