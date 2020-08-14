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
#include "src/common/anf_importer/anf_populater/anf_matmul_populater.h"

#include <memory>
#include <vector>

#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "src/common/anf_importer/anf_populater/anf_node_populater_registry.h"
#include "src/ir/tensor.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite {
void AnfMatmulPopulater::CalQuantParam(const double &mean, const double &stdDev,
                                       float *mMin, float *mMax) {
  constexpr float qmin = 0;
  constexpr float qmax = 255;
  *mMin = static_cast<float>((qmin - mean) / stdDev);
  *mMax = static_cast<float>((qmax - mean) / stdDev);
}

void AnfMatmulPopulater::PopulaterQuantParam(
    const PrimitivePtr &prim,
    std::vector<std::vector<schema::QuantParamT>> *vecQuantParam) {
  auto narrow_range = prim->GetAttr("narrow_range");
  bool narrowRangeQuantParam = GetValue<bool>(narrow_range);
  auto num_bits = prim->GetAttr("num_bits");
  int32_t numbitsRangeQuantParam = GetValue<int32_t>(num_bits);

  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quantParam;
  auto mean = prim->GetAttr("mean");
  auto std_dev = prim->GetAttr("std_dev");
  if (mean != nullptr && std_dev != nullptr) {
    auto meanQuantOaram = GetValue<double>(mean);
    double stddevQuantOaram = GetValue<double>(std_dev);
    float mMin = 0.0;
    float mMax = 0.0;
    CalQuantParam(meanQuantOaram, stddevQuantOaram, &mMin, &mMax);
    quantParam.min = mMin;
    quantParam.max = mMax;
  } else {
    auto inputMin = prim->GetAttr("input_minq");
    auto inputMax = prim->GetAttr("input_maxq");
    auto inputMinPtr = inputMin->cast<lite::tensor::TensorPtr>();
    auto inputMaxPtr = inputMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(inputMinPtr->Data());
    float *maxBuf = static_cast<float *>(inputMaxPtr->Data());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
  }
  quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max,
                               narrowRangeQuantParam, numbitsRangeQuantParam);
  quants.emplace_back(quantParam);
  vecQuantParam->emplace_back(quants);

  quants.clear();
  auto filterMin = prim->GetAttr("filter_minq");
  auto filterMax = prim->GetAttr("filter_maxq");
  if (filterMin != nullptr && filterMax != nullptr) {
    auto filterMinPtr = filterMin->cast<lite::tensor::TensorPtr>();
    auto filterMaxPtr = filterMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(filterMinPtr->Data());
    float *maxBuf = static_cast<float *>(filterMaxPtr->Data());
    for (int i = 0; i < filterMinPtr->DataSize(); ++i) {
      quantParam.min = *(minBuf++);
      quantParam.max = *(maxBuf++);
      quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max,
                                   narrowRangeQuantParam,
                                   numbitsRangeQuantParam);
      quants.emplace_back(quantParam);
    }
    vecQuantParam->emplace_back(quants);
  }

  quants.clear();
  auto outputMin = prim->GetAttr("output_minq");
  auto outputMax = prim->GetAttr("output_maxq");
  if (outputMin != nullptr && outputMax != nullptr) {
    auto outputMinPtr = outputMin->cast<lite::tensor::TensorPtr>();
    auto outputMaxPtr = outputMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(outputMinPtr->Data());
    float *maxBuf = static_cast<float *>(outputMaxPtr->Data());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
    quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max,
                                 narrowRangeQuantParam, numbitsRangeQuantParam);
    quants.emplace_back(quantParam);
    vecQuantParam->emplace_back(quants);
  }
}

int AnfMatmulPopulater::Populate(const PrimitivePtr &prim,
                                 PrimitiveTValue *primitiveTValuePtr,
                                 const std::vector<AnfNodePtr> &inputs) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  auto attr = std::make_unique<schema::MatMulT>();
  attr->transposeA = GetValue<bool>(prim->GetAttr("transpose_a"));
  attr->transposeB = GetValue<bool>(prim->GetAttr("transpose_b"));

  primitive->value.type = schema::PrimitiveType_MatMul;
  primitive->value.value = attr.release();
  MS_ASSERT(primitiveTValuePtr != nullptr);
  primitiveTValuePtr->SetPrimitiveT(primitive.release());
  if (primitiveTValuePtr->GetQuantType()) {
    std::vector<std::vector<schema::QuantParamT>> vecQuantParam;
    PopulaterQuantParam(prim, &vecQuantParam);
    primitiveTValuePtr->SetInputQuantParam(vecQuantParam);
  }

  return 0;
}
AnfNodePopulaterRegistrar anfMatmulPopulater("Matmul",
                                             new AnfMatmulPopulater());
AnfNodePopulaterRegistrar anfMatMulPopulater("MatMul",
                                             new AnfMatmulPopulater());
}  // namespace mindspore::lite
