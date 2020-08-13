/**
 * This is the C++ adaptation and derivative work of Myia
 * (https://github.com/mila-iqia/myia/).
 *
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

#include "src/common/anf_importer/anf_populater/anf_conv_populater.h"

#include <mindspore/lite/src/ir/tensor.h>

#include <memory>
#include <string>
#include <vector>

#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "src/common/anf_importer/anf_populater/anf_node_populater_registry.h"
#include "src/ir/tensor.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite {
void AnfConvPopulater::PopulaterConv2DMultiGroup(
    const PrimitivePtr &prim,
    const std::unique_ptr<schema::PrimitiveT> &primitive, const int &group) {
  auto attr = std::make_unique<schema::DepthwiseConv2DT>();
  auto format = GetValue<std::string>(prim->GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }
  auto pad_list = GetValue<std::vector<int>>(prim->GetAttr("pad_list"));
  attr->padUp = pad_list[0];
  attr->padDown = pad_list[1];
  attr->padLeft = pad_list[2];
  attr->padRight = pad_list[3];

  auto dilation = GetValue<std::vector<int>>(prim->GetAttr("dilation"));
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  auto kernel_size = GetValue<std::vector<int>>(prim->GetAttr("kernel_size"));
  attr->kernelH = kernel_size[0];
  attr->kernelW = kernel_size[1];

  auto stride = GetValue<std::vector<int>>(prim->GetAttr("stride"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  auto pad_mode = GetValue<std::string>(prim->GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }

  primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  primitive->value.value = attr.release();
}

void AnfConvPopulater::PopulaterConv2DSingleGroup(
    const PrimitivePtr &prim,
    const std::unique_ptr<schema::PrimitiveT> &primitive, const int &group) {
  auto attr = std::make_unique<schema::Conv2DT>();
  attr->group = group;
  auto format = GetValue<std::string>(prim->GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }
  auto pad_list = GetValue<std::vector<int>>(prim->GetAttr("pad_list"));
  attr->padUp = pad_list[0];
  attr->padDown = pad_list[1];
  attr->padLeft = pad_list[2];
  attr->padRight = pad_list[3];

  auto dilation = GetValue<std::vector<int>>(prim->GetAttr("dilation"));
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  auto kernel_size = GetValue<std::vector<int>>(prim->GetAttr("kernel_size"));
  attr->kernelH = kernel_size[0];
  attr->kernelW = kernel_size[1];

  auto stride = GetValue<std::vector<int>>(prim->GetAttr("stride"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  attr->channelOut = GetValue<int>(prim->GetAttr("out_channel"));

  auto pad_mode = GetValue<std::string>(prim->GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }
  primitive->value.type = schema::PrimitiveType_Conv2D;
  primitive->value.value = attr.release();
}

void AnfConvPopulater::CalQuantParam(const double &mean, const double &stdDev,
                                     float *mMin, float *mMax) {
  constexpr float qmin = 0;
  constexpr float qmax = 255;
  *mMin = static_cast<float>((qmin - mean) / stdDev);
  *mMax = static_cast<float>((qmax - mean) / stdDev);
}

void AnfConvPopulater::PopulaterQuantParam(
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
  int biasQuantSize = 0;
  auto filterMin = prim->GetAttr("filter_minq");
  auto filterMax = prim->GetAttr("filter_maxq");
  if (filterMin != nullptr && filterMax != nullptr) {
    auto filterMinPtr = filterMin->cast<lite::tensor::TensorPtr>();
    auto filterMaxPtr = filterMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(filterMinPtr->Data());
    float *maxBuf = static_cast<float *>(filterMaxPtr->Data());
    biasQuantSize = filterMinPtr->DataSize();
    for (int i = 0; i < biasQuantSize; ++i) {
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
  for (int i = 0; i < biasQuantSize; ++i) {
    quantParam.min = 0.0;
    quantParam.max = 0.0;
    quantParam.zeroPoint = 0;
    quantParam.scale =
        vecQuantParam->at(0).at(0).scale * vecQuantParam->at(1).at(i).scale;
    quants.emplace_back(quantParam);
  }
  vecQuantParam->emplace_back(quants);

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

int AnfConvPopulater::Populate(const PrimitivePtr &prim,
                               PrimitiveTValue *primitiveTValuePtr,
                               const std::vector<AnfNodePtr> &inputs) {
  MS_ASSERT(primitiveTValuePtr != nullptr);
  auto primitive = std::make_unique<schema::PrimitiveT>();

  int group = GetValue<int>(prim->GetAttr("group"));
  if (group > 1) {
    PopulaterConv2DMultiGroup(prim, primitive, group);
  } else {
    PopulaterConv2DSingleGroup(prim, primitive, group);
  }
  primitiveTValuePtr->SetPrimitiveT(primitive.release());
  if (primitiveTValuePtr->GetQuantType() == schema::QuantType_AwareTraining) {
    std::vector<std::vector<schema::QuantParamT>> vecQuantParam;
    PopulaterQuantParam(prim, &vecQuantParam);
    primitiveTValuePtr->SetInputQuantParam(vecQuantParam);
  }
  return 0;
}
AnfNodePopulaterRegistrar anfConvPopulater("Conv2D", new AnfConvPopulater());
}  // namespace mindspore::lite
