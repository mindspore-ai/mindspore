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

#include "mindspore/lite/tools/converter/quantizer/quantize_util.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include "src/ops/primitive_c.h"
#include "mindspore/lite/tools/converter/quantizer/bitpacking.h"
#include "src/common/utils.h"
#include "abstract/abstract_value.h"
#include "securec/include/securec.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
const std::vector<schema::PrimitiveType> QuantStrategy::conv_types = {
  schema::PrimitiveType_DeConv2D, schema::PrimitiveType_DeDepthwiseConv2D, schema::PrimitiveType_Conv2D,
  schema::PrimitiveType_DepthwiseConv2D};
const std::vector<schema::PrimitiveType> QuantStrategy::mul_types = {schema::PrimitiveType_MatMul,
                                                                     schema::PrimitiveType_FullConnection};
QuantStrategy::QuantStrategy(size_t weightSize, size_t convWeightQuantChannelThreshold)
    : mWeightSize(weightSize), mConvWeightQuantChannelThreshold(convWeightQuantChannelThreshold) {}

bool QuantStrategy::CanConvOpQuantized(const CNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(node->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr";
    return false;
  }
  if (!IsContain(conv_types, (schema::PrimitiveType)primitive_c->Type())) {
    return false;
  }
  if (node->size() < 3) {
    return false;
  }
  auto inputNode = node->input(2);
  if (!inputNode->isa<Parameter>()) {
    return false;
  }
  auto paramNode = inputNode->cast<ParameterPtr>();
  auto abstract_base = paramNode->abstract();
  if (abstract_base == nullptr) {
    return false;
  }
  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << paramNode->name();
    return false;
  }
  auto weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  size_t shapeSize = 1;
  for (auto dim : weight_shape) {
    shapeSize = shapeSize * dim;
  }
  if (shapeSize < mWeightSize) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }
  if (weight_shape[0] <= static_cast<int>(mConvWeightQuantChannelThreshold)) {
    MS_LOG(INFO) << "channel less mConvWeightQuantChannelThreshold!" << weight_shape[0];
    return false;
  }
  return true;
}

bool QuantStrategy::CanOpPostQuantized(AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = std::dynamic_pointer_cast<CNode>(node);
  auto type = NodePrimitiveType(cnode);
  static const std::vector<schema::PrimitiveType> int8OpList = {
    schema::PrimitiveType_Conv2D,
    schema::PrimitiveType_DepthwiseConv2D,
    schema::PrimitiveType_Add,
    schema::PrimitiveType_Mul,
    schema::PrimitiveType_Pooling,
    schema::PrimitiveType_Concat,
    schema::PrimitiveType_Split,
    schema::PrimitiveType_TupleGetItem,
    schema::PrimitiveType_Reshape,
    schema::PrimitiveType_FullConnection,
    schema::PrimitiveType_MatMul,
    schema::PrimitiveType_Crop,
    schema::PrimitiveType_DeDepthwiseConv2D,
    schema::PrimitiveType_DeConv2D,
    schema::PrimitiveType_Activation,
    schema::PrimitiveType_Transpose,
    schema::PrimitiveType_Eltwise,
    schema::PrimitiveType_Gather,
    schema::PrimitiveType_LayerNorm,
  };
  bool contain = IsContain(int8OpList, type);
  if (!contain) {
    MS_LOG(INFO) << "not quant, " << cnode->fullname_with_scope()
                 << " of type: " << schema::EnumNamePrimitiveType(type);
  }
  return contain;
}

bool QuantStrategy::CanMulOpQuantized(const CNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(node->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr";
    return false;
  }

  if (!IsContain(mul_types, (schema::PrimitiveType)primitive_c->Type())) {
    return false;
  }

  if (node->size() < 3) {
    MS_LOG(INFO) << "input size less!";
    return false;
  }

  auto inputNode1 = node->input(1);
  auto inputNode2 = node->input(2);
  if (inputNode1 == nullptr || inputNode2 == nullptr) {
    MS_LOG(INFO) << "mul input is nullptr!";
    return false;
  }

  ParameterPtr paramNode = nullptr;
  if (inputNode1->isa<Parameter>()) {
    paramNode = inputNode1->cast<ParameterPtr>();
  } else if (inputNode2->isa<Parameter>()) {
    paramNode = inputNode2->cast<ParameterPtr>();
  }

  if (paramNode == nullptr) {
    MS_LOG(INFO) << "invalid paramNode!";
    return false;
  }

  auto abstract_base = paramNode->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(INFO) << "abstract is nullptr";
    return false;
  }

  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << paramNode->name();
    return false;
  }
  auto weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  size_t shapeSize = 1;
  for (auto dim : weight_shape) {
    shapeSize = shapeSize * dim;
  }
  if (shapeSize < mWeightSize) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }

  return true;
}

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange, int quant_max,
                             int quant_min, int num_bits) {
  MS_ASSERT(quantParam != nullptr);
  if (mMin > 0.0f) {
    MS_LOG(DEBUG) << "min " << mMin << " is bigger then 0, set to 0, this may course low precision";
    mMin = 0.0f;
  }
  if (mMax < 0.0f) {
    MS_LOG(DEBUG) << "mMax " << mMax << " is smaller than 0, set to 0, this may course low precision";
    mMax = 0.0f;
  }
  if (mMin > mMax) {
    MS_LOG(ERROR) << "cal error while min" << mMin << ">" << mMax;
    return RET_PARAM_INVALID;
  }
  if (mMin == mMax) {
    if (mMin != 0.0f) {
      MS_LOG(ERROR) << "min and max should both be zero if they are equal to each other";
      return RET_ERROR;
    }
    quantParam->inited = true;
    quantParam->min = mMin;
    quantParam->max = mMax;
    quantParam->scale = 0.0f;
    quantParam->zeroPoint = 0;
    quantParam->narrowRange = narrowRange;
    quantParam->numBits = num_bits;
    return RET_OK;
  }

  auto quantMinFloat = static_cast<double>(quant_min);
  auto quantMaxFloat = static_cast<double>(quant_max);
  if (fabs(quantMaxFloat - quantMinFloat) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return RET_ERROR;
  }
  double scale = (mMax - mMin) / (quantMaxFloat - quantMinFloat);
  if (fabs(scale) <= 0.0f) {
    MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
    return RET_ERROR;
  }
  const double zeroPointFromMin = quantMinFloat - mMin / scale;
  int zeroPoint = static_cast<int32_t>(std::round(zeroPointFromMin));

  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  MS_ASSERT(zeroPoint >= quantMin);
  MS_ASSERT(zeroPoint <= quantMax);
  quantParam->inited = true;
  quantParam->min = mMin;
  quantParam->max = mMax;
  quantParam->scale = scale;
  quantParam->zeroPoint = zeroPoint;
  quantParam->narrowRange = narrowRange;
  quantParam->numBits = num_bits;

  return RET_OK;
}

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange, int numBits) {
  MS_ASSERT(quantParam != nullptr);
  if (mMin > 0.0f) {
    MS_LOG(DEBUG) << "min " << mMin << " is bigger then 0, set to 0, this may course low precision";
    mMin = 0.0f;
  }
  if (mMax < 0.0f) {
    MS_LOG(DEBUG) << "mMax " << mMax << " is smaller than 0, set to 0, this may course low precision";
    mMax = 0.0f;
  }
  if (mMin > mMax) {
    MS_LOG(ERROR) << "cal error while min" << mMin << ">" << mMax;
    return RET_PARAM_INVALID;
  }
  if (mMin == mMax) {
    if (mMin != 0.0f) {
      MS_LOG(ERROR) << "min and max should both be zero if they are equal to each other";
      return RET_ERROR;
    }
    quantParam->inited = false;
    quantParam->min = mMin;
    quantParam->max = mMax;
    quantParam->scale = 0.0f;
    quantParam->zeroPoint = 0;
    quantParam->narrowRange = narrowRange;
    quantParam->numBits = numBits;
    return RET_OK;
  }

  const int8_t quantMin = std::numeric_limits<int8_t>::min() + (narrowRange ? 1 : 0);
  const int8_t quantMax = std::numeric_limits<int8_t>::max();
  auto quantMinFloat = static_cast<double>(quantMin);
  auto quantMaxFloat = static_cast<double>(quantMax);
  if (fabs(quantMaxFloat - quantMinFloat) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return RET_ERROR;
  }
  double scale = (mMax - mMin) / (quantMaxFloat - quantMinFloat);
  if (fabs(scale) <= 0.0f) {
    MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
    return RET_ERROR;
  }
  const double zeroPointFromMin = quantMinFloat - mMin / scale;
  const double zeroPointFromMax = quantMaxFloat - mMax / scale;
  const double zpFromMinError = std::abs(quantMinFloat) + std::abs(mMin / scale);
  const double zpFromMaxError = std::abs(quantMaxFloat) + std::abs(mMax / scale);
  const double zpDouble = zpFromMinError < zpFromMaxError ? zeroPointFromMin : zeroPointFromMax;
  int zeroPoint;
  if (zpDouble < quantMinFloat) {
    zeroPoint = quantMin;
  } else if (zpDouble > quantMaxFloat) {
    zeroPoint = quantMax;
  } else {
    zeroPoint = static_cast<int32_t>(std::round(zpDouble));
  }
  if (std::abs(mMin) == std::abs(mMax)) {
    zeroPoint = 0;
  }
  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  MS_ASSERT(zeroPoint >= quantMin);
  MS_ASSERT(zeroPoint <= quantMax);
  quantParam->inited = true;
  quantParam->min = mMin;
  quantParam->max = mMax;
  quantParam->scale = scale;
  quantParam->zeroPoint = zeroPoint;
  quantParam->narrowRange = narrowRange;
  quantParam->numBits = numBits;

  return RET_OK;
}

static bool SearchLowerBound(const std::vector<float> &data, const size_t &index, const float &max_tmp, float *min_tmp,
                             size_t *min_idx) {
  size_t length = data.size();
  if (max_tmp - data.at(index) < delta) {
    return false;
  }
  if (fabs(max_tmp - *min_tmp) <= 0.0f || fabs(length - *min_idx) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return false;
  }
  float range_ratio = (data.at(index) - *min_tmp) / (max_tmp - *min_tmp);
  float index_ratio = static_cast<float>(index - *min_idx) / (length - *min_idx);
  if (fabs(index_ratio) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return false;
  }
  if (index_ratio > 0 && range_ratio / index_ratio > ratio) {
    *min_idx = index;
    *min_tmp = data.at(index);
  }
  return true;
}

static bool SearchUpperBound(const std::vector<float> &data, const size_t &index, float *max_tmp, const float &min_tmp,
                             size_t *max_idx) {
  size_t length = data.size();
  if (data.at(index) - min_tmp < delta) {
    return false;
  }
  if (fabs(*max_tmp - min_tmp) <= 0.0f || fabs(length - *max_idx) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return false;
  }
  float range_ratio = (*max_tmp - data.at(index)) / (*max_tmp - min_tmp);
  float index_ratio = static_cast<float>(index - *max_idx) / (length - *max_idx);
  if (fabs(index_ratio) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return false;
  }
  if (index_ratio > 0 && range_ratio / index_ratio > ratio) {
    *max_idx = index;
    *max_tmp = data.at(index);
  }
  return true;
}

static float CalPercentile(const std::vector<float> &datas, const int &outlier_percent) {
  const int size = datas.size();
  float val = outlier_percent / 100.0 * size;
  int index = std::ceil(val);
  float result;
  if (index - val > 0) {
    result = datas.at(index - 1);
  } else {
    result = (datas.at(index - 1) + datas.at(index)) / 2;
  }
  return result;
}

std::pair<float, float> OutlierMethod(std::vector<float> min_datas, std::vector<float> max_datas) {
  std::sort(max_datas.begin(), max_datas.end());
  std::sort(min_datas.begin(), min_datas.end());
  float min_val = CalPercentile(min_datas, percent);
  float max_val = CalPercentile(max_datas, 100 - percent);
  std::reverse(max_datas.begin(), max_datas.end());
  MS_ASSERT(min_val < max_val);
  MS_ASSERT(min_datas.size() == max_datas.size());
  float min_tmp = min_val;
  float max_tmp = max_val;
  size_t min_idx = 0;
  size_t max_idx = 0;
  size_t length = min_datas.size();
  for (size_t i = 0; i < length; i++) {
    if (!SearchLowerBound(min_datas, i, max_tmp, &min_tmp, &min_idx)) {
      break;
    }
    if (!SearchUpperBound(min_datas, i, &max_tmp, min_tmp, &max_idx)) {
      break;
    }
  }
  std::pair<float, float> result{min_tmp, max_tmp};
  return result;
}

static std::vector<float> InitClusters(float *data, size_t elem_count, size_t k) {
  std::set<float> set_unique{};
  for (size_t i = 0; i < elem_count; i++) {
    set_unique.emplace(data[i]);
  }
  std::vector<float> data_unique;
  data_unique.assign(set_unique.begin(), set_unique.end());
  std::vector<float> clusters{};
  if (set_unique.size() < k) {
    return clusters;
  }
  // init cluster
  MS_ASSERT(k != 1);
  float ratio = static_cast<float>(data_unique.size()) / (k - 1);
  std::sort(data_unique.begin(), data_unique.end());
  for (size_t i = 0; i < k; i++) {
    size_t index = std::floor(i * ratio);
    if (i * ratio - index > 0) {
      clusters.emplace_back((data_unique[index] + data_unique[index + 1]) / 2);
    } else {
      clusters.emplace_back(data_unique[index]);
    }
  }
  return clusters;
}

std::vector<int8_t> KMeans(float *data, size_t elem_count, size_t k, size_t epochs, schema::QuantParamT *quantParam) {
  MS_ASSERT(data != nullptr);
  MS_ASSERT(quantParam != nullptr);
  std::vector<float> clusters = InitClusters(data, elem_count, k);
  std::vector<int8_t> clusters_index{};
  double error{0};
  if (clusters.size() < k) {
    MS_LOG(WARNING) << "K is less than the size of data so KMeans function is not executed.";
    return clusters_index;
  }
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    double error_cur{0};
    clusters_index.clear();
    std::vector<std::vector<float>> clusters_data(clusters.size());
    for (size_t i = 0; i < elem_count; i++) {
      size_t index = 0;
      float min_distance = pow(data[i] - clusters[0], 2);
      for (size_t j = 1; j < clusters.size(); j++) {
        if (pow(data[i] - clusters[j], 2) < min_distance) {
          min_distance = pow(data[i] - clusters[j], 2);
          index = j;
        }
      }
      clusters_index.emplace_back(index + INT8_MIN);
      clusters_data[index].emplace_back(data[i]);
    }
    for (size_t j = 0; j < clusters.size(); j++) {
      if (!clusters_data[j].empty()) {
        clusters[j] = std::accumulate(clusters_data[j].begin(), clusters_data[j].end(), 0.0) / clusters_data[j].size();
      }
    }
    // compare error
    for (size_t j = 0; j < elem_count; j++) {
      error_cur += pow(data[j] - clusters[clusters_index[j]], 2);
    }
    error_cur = pow(error_cur / elem_count, 0.5);
    if (std::abs((error_cur - error) / error_cur) <= 0.0f) {
      break;
    }
    error = error_cur;
  }
  // update data
  return clusters_index;
}

schema::PrimitiveType NodePrimitiveType(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is null";
    return schema::PrimitiveType_NONE;
  }
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is null";
    return schema::PrimitiveType_NONE;
  }
  return (schema::PrimitiveType)primitive_c->Type();
}
}  // namespace mindspore::lite::quant
