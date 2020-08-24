/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/matmul.h"
#include <memory>
#include <utility>
#ifdef PRIMITIVE_WRITEABLE
#include "tools/converter/quantizer/quantize_util.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool MatMul::GetTransposeA() const { return this->primitive_->value.AsMatMul()->transposeA; }
bool MatMul::GetTransposeB() const { return this->primitive_->value.AsMatMul()->transposeB; }

void MatMul::SetTransposeA(bool transpose_a) { this->primitive_->value.AsMatMul()->transposeA = transpose_a; }
void MatMul::SetTransposeB(bool transpose_b) { this->primitive_->value.AsMatMul()->transposeB = transpose_b; }

void MatMul::CalQuantParam(const double &mean, const double &stdDev, float *mMin, float *mMax) {
  constexpr float qmin = 0;
  constexpr float qmax = 255;
  *mMin = static_cast<float>((qmin - mean) / stdDev);
  *mMax = static_cast<float>((qmax - mean) / stdDev);
}

void MatMul::PopulaterQuantParam(const Primitive &prim,
                                 std::vector<std::vector<schema::QuantParamT>> *vecInputQuantParam,
                                 std::vector<std::vector<schema::QuantParamT>> *vecOutputQuantParam) {
  auto narrow_range = prim.GetAttr("narrow_range");
  bool narrowRangeQuantParam = GetValue<bool>(narrow_range);
  auto num_bits = prim.GetAttr("num_bits");
  int32_t numbitsRangeQuantParam = GetValue<int32_t>(num_bits);

  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quantParam;
  auto mean = prim.GetAttr("mean");
  auto std_dev = prim.GetAttr("std_dev");
  if (mean != nullptr && std_dev != nullptr) {
    auto meanQuantOaram = GetValue<double>(mean);
    double stddevQuantOaram = GetValue<double>(std_dev);
    float mMin = 0.0;
    float mMax = 0.0;
    CalQuantParam(meanQuantOaram, stddevQuantOaram, &mMin, &mMax);
    quantParam.min = mMin;
    quantParam.max = mMax;
  } else {
    auto inputMin = prim.GetAttr("input_minq");
    auto inputMax = prim.GetAttr("input_maxq");
    auto inputMinPtr = inputMin->cast<lite::tensor::TensorPtr>();
    auto inputMaxPtr = inputMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(inputMinPtr->Data());
    float *maxBuf = static_cast<float *>(inputMaxPtr->Data());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
  }
  quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                               numbitsRangeQuantParam);
  quants.emplace_back(quantParam);
  vecInputQuantParam->emplace_back(quants);

  quants.clear();
  auto filterMin = prim.GetAttr("filter_minq");
  auto filterMax = prim.GetAttr("filter_maxq");
  if (filterMin != nullptr && filterMax != nullptr) {
    auto filterMinPtr = filterMin->cast<lite::tensor::TensorPtr>();
    auto filterMaxPtr = filterMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(filterMinPtr->Data());
    float *maxBuf = static_cast<float *>(filterMaxPtr->Data());
    for (int i = 0; i < filterMinPtr->DataSize(); ++i) {
      quantParam.min = *(minBuf++);
      quantParam.max = *(maxBuf++);
      quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                                   numbitsRangeQuantParam);
      quants.emplace_back(quantParam);
    }
    vecInputQuantParam->emplace_back(quants);
  }

  quants.clear();
  auto outputMin = prim.GetAttr("output_minq");
  auto outputMax = prim.GetAttr("output_maxq");
  if (outputMin != nullptr && outputMax != nullptr) {
    auto outputMinPtr = outputMin->cast<lite::tensor::TensorPtr>();
    auto outputMaxPtr = outputMax->cast<lite::tensor::TensorPtr>();
    float *minBuf = static_cast<float *>(outputMinPtr->Data());
    float *maxBuf = static_cast<float *>(outputMaxPtr->Data());
    quantParam.min = *minBuf;
    quantParam.max = *maxBuf;
    quant::CalQuantizationParams(&quantParam, quantParam.min, quantParam.max, narrowRangeQuantParam,
                                 numbitsRangeQuantParam);
    quants.emplace_back(quantParam);
    vecOutputQuantParam->emplace_back(quants);
  }
}

int MatMul::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_MatMul;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_MatMul) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::MatMulT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->transposeA = GetValue<bool>(prim.GetAttr("transpose_a"));
    attr->transposeB = GetValue<bool>(prim.GetAttr("transpose_b"));
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  if (GetQuantType() == schema::QuantType_AwareTraining) {
    std::vector<std::vector<schema::QuantParamT>> vecInputQuantParam;
    std::vector<std::vector<schema::QuantParamT>> vecOutputQuantParam;
    PopulaterQuantParam(prim, &vecInputQuantParam, &vecOutputQuantParam);
    SetInputQuantParam(vecInputQuantParam);
    SetOutputQuantParam(vecOutputQuantParam);
  }
  return RET_OK;
}

#else

bool MatMul::GetTransposeA() const { return this->primitive_->value_as_MatMul()->transposeA(); }
bool MatMul::GetTransposeB() const { return this->primitive_->value_as_MatMul()->transposeB(); }

#endif

int MatMul::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_.at(1);
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  std::vector<int> a_shape = input0->shape();
  std::vector<int> b_shape = input1->shape();
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    MS_LOG(ERROR) << "inputs shape is invalid";
    return RET_INPUT_TENSOR_ERROR;
  }
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    if (a_shape[i] != b_shape[i]) {
      MS_LOG(ERROR) << "Op MatMul's dimensions must be equal";
      return RET_INPUT_TENSOR_ERROR;
    }
  }

  if (GetTransposeA()) {
    std::swap(a_shape[a_shape.size() - 1], a_shape[a_shape.size() - 2]);
  }
  if (GetTransposeB()) {
    std::swap(b_shape[b_shape.size() - 1], b_shape[b_shape.size() - 2]);
  }
  std::vector<int> c_shape(a_shape);
  c_shape[c_shape.size() - 1] = b_shape[b_shape.size() - 1];
  output->set_shape(c_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
