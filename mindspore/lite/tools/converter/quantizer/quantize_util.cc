/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <map>
#include <fstream>
#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include "include/version.h"
#include "ops/concat.h"
#include "ops/crop.h"
#include "ops/eltwise.h"
#include "ops/fusion/activation.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/full_connection.h"
#include "ops/fusion/layer_norm_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/gather.h"
#include "ops/mat_mul.h"
#include "ops/reshape.h"
#include "ops/split.h"
#include "ops/transpose.h"
#include "ops/tuple_get_item.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "tools/converter/quantizer/bitpacking.h"
#include "src/common/utils.h"
#include "abstract/abstract_value.h"
#include "securec/include/securec.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
const std::vector<std::string> QuantStrategy::conv_types_ = {ops::kNameConv2DFusion, ops::kNameConv2dTransposeFusion};
const std::vector<std::string> QuantStrategy::mul_types_ = {ops::kNameMatMul, ops::kNameFullConnection};
QuantStrategy::QuantStrategy(size_t weight_size, size_t conv_weight_quant_channel_threshold)
    : m_weight_size_(weight_size), m_conv_weight_quant_channel_threshold_(conv_weight_quant_channel_threshold) {}

bool QuantStrategy::CanConvOpQuantized(const CNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto primitive_c = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(node->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr";
    return false;
  }
  if (!IsContain(conv_types_, primitive_c->name())) {
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
  if (shapeSize < m_weight_size_) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }
  if (weight_shape[0] <= static_cast<int>(m_conv_weight_quant_channel_threshold_)) {
    MS_LOG(INFO) << "channel less m_conv_weight_quant_channel_threshold_!" << weight_shape[0];
    return false;
  }
  return true;
}

bool QuantStrategy::CanOpPostQuantized(AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  if (!node->isa<mindspore::CNode>()) {
    return false;
  }
  auto cnode = std::dynamic_pointer_cast<mindspore::CNode>(node);
  auto type = NodePrimitiveType(cnode);
  static const std::vector<std::string> int8OpList = {
    ops::kNameAddFusion,     ops::kNameActivation,      ops::kNameAvgPoolFusion,
    ops::kNameConcat,        ops::kNameConv2DFusion,    ops::kNameConv2dTransposeFusion,
    ops::kNameCrop,          ops::kNameEltwise,         ops::kNameFullConnection,
    ops::kNameGather,        ops::kNameLayerNormFusion, ops::kNameMatMul,
    ops::kNameMaxPoolFusion, ops::kNameMulFusion,       ops::kNameReshape,
    ops::kNameSplit,         ops::kNameTranspose,       ops::kNameTupleGetItem,
  };
  bool contain = IsContain(int8OpList, type);
  if (!contain) {
    MS_LOG(INFO) << "not quant, " << cnode->fullname_with_scope() << " of type: " << type;
  }
  return contain;
}

bool QuantStrategy::CanMulOpQuantized(const CNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto primitive_c = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(node->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr";
    return false;
  }

  if (!IsContain(mul_types_, primitive_c->name())) {
    return false;
  }

  if (node->size() < 3) {
    MS_LOG(INFO) << node->fullname_with_scope() << " input size less!";
    return false;
  }

  auto inputNode1 = node->input(1);
  auto inputNode2 = node->input(2);
  if (inputNode1 == nullptr || inputNode2 == nullptr) {
    MS_LOG(INFO) << node->fullname_with_scope() << " mul input is nullptr!";
    return false;
  }

  ParameterPtr paramNode = nullptr;
  if (inputNode1->isa<Parameter>()) {
    paramNode = inputNode1->cast<ParameterPtr>();
  } else if (inputNode2->isa<Parameter>()) {
    paramNode = inputNode2->cast<ParameterPtr>();
  }

  if (paramNode == nullptr) {
    MS_LOG(INFO) << node->fullname_with_scope() << " invalid paramNode!";
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
  if (shapeSize < m_weight_size_) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }

  return true;
}

QuantParamHolderPtr GetCNodeQuantHolder(const PrimitivePtr &primitive) {
  MS_ASSERT(primitive != nullptr);
  QuantParamHolderPtr quant_params_holder = nullptr;
  auto quant_params_valueptr = primitive->GetAttr("quant_params");
  if (quant_params_valueptr == nullptr) {
    quant_params_holder = std::make_shared<QuantParamHolder>();
    primitive->AddAttr("quant_params", quant_params_holder);
  } else {
    quant_params_holder = quant_params_valueptr->cast<QuantParamHolderPtr>();
    if (quant_params_holder == nullptr) {
      quant_params_holder = std::make_shared<QuantParamHolder>();
      primitive->AddAttr("quant_params", quant_params_holder);
    }
  }
  return quant_params_holder;
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

static float CalPercentile(const std::vector<float> &data, const int &outlier_percent) {
  const int size = data.size();
  float val = outlier_percent / 100.0 * size;
  int index = std::ceil(val);
  float result;
  if (index - val > 0) {
    result = data.at(index - 1);
  } else {
    result = (data.at(index - 1) + data.at(index)) / 2;
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

std::string NodePrimitiveType(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is null";
    return "";
  }
  auto primitive_c = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is null";
    return "";
  }
  return primitive_c->name();
}

std::vector<int> DataToVector(const string &str) {
  std::vector<int> result;
  auto raw_datas = str;
  auto ind = raw_datas.find(',');
  while (ind != std::string::npos) {
    auto data = raw_datas.substr(0, ind);
    Trim(&data);
    result.push_back(std::stoul(data));
    raw_datas = raw_datas.substr(ind + 1);
    Trim(&raw_datas);
    ind = raw_datas.find(',');
  }
  if (!raw_datas.empty()) {
    result.push_back(std::stoul(raw_datas));
  }
  if (result.empty()) {
    MS_LOG(ERROR) << "result is empty";
  }
  return result;
}

std::vector<std::vector<int>> DataToVectors(const string &str) {
  std::vector<std::vector<int>> result;
  auto raw_datas = str;
  auto ind = raw_datas.find(';');
  while (ind != std::string::npos) {
    auto data = raw_datas.substr(0, ind);
    Trim(&data);
    result.push_back(DataToVector(data));
    raw_datas = raw_datas.substr(ind + 1);
    Trim(&raw_datas);
    ind = raw_datas.find(';');
  }
  if (!raw_datas.empty()) {
    result.push_back(DataToVector(raw_datas));
  }
  if (result.empty()) {
    MS_LOG(ERROR) << "result is empty";
  }
  return result;
}

void ParseInputShape(PostQuantConfig *post_quant_config, std::string raw_shape) {
  MS_ASSERT(post_quant_config != nullptr);
  auto ind = raw_shape.find('/');
  while (ind != std::string::npos) {
    auto shape = raw_shape.substr(0, ind);
    Trim(&shape);
    post_quant_config->input_shapes.push_back(DataToVectors(shape));
    raw_shape = raw_shape.substr(ind + 1);
    Trim(&raw_shape);
    ind = raw_shape.find('/');
  }
  if (!raw_shape.empty()) {
    post_quant_config->input_shapes.push_back(DataToVectors(raw_shape));
  }
}

void ParseImagePath(PostQuantConfig *post_quant_config, std::string raw_image_paths) {
  MS_ASSERT(post_quant_config != nullptr);
  auto ind = raw_image_paths.find(',');
  while (ind != std::string::npos) {
    auto image_path = raw_image_paths.substr(0, ind);
    Trim(&image_path);
    post_quant_config->image_paths.push_back(image_path);
    raw_image_paths = raw_image_paths.substr(ind + 1);
    Trim(&raw_image_paths);
    ind = raw_image_paths.find(',');
  }
  post_quant_config->image_paths.push_back(raw_image_paths);
}

void ParseBatchCount(PostQuantConfig *post_quant_config, std::string value) {
  MS_ASSERT(post_quant_config != nullptr);
  post_quant_config->batch_count = std::stoul(value);
}

void ParseThreadNum(PostQuantConfig *post_quant_config, std::string value) {
  MS_ASSERT(post_quant_config != nullptr);
  post_quant_config->thread_num = std::stoul(value);
}

void ParseMethodX(PostQuantConfig *post_quant_config, const std::string &value) {
  MS_ASSERT(post_quant_config != nullptr);
  if (value != kMethodKL && value != kMethodMaxMin && value != kMethodOutlier) {
    MS_LOG(WARNING) << "unsupported method_x: " << value << ". Use default value.";
  } else {
    post_quant_config->method_x = value;
  }
}

void ParseMixed(PostQuantConfig *post_quant_config, std::string value) {
  MS_ASSERT(post_quant_config != nullptr);
  std::for_each(value.begin(), value.end(), ::tolower);
  if (value == "true") {
    post_quant_config->mixed = true;
  }
}

void ParseMeanErrorThreshold(PostQuantConfig *post_quant_config, std::string value) {
  MS_ASSERT(post_quant_config != nullptr);
  post_quant_config->mean_error_threshold = std::stof(value);
}

void ParseBiasCorrection(PostQuantConfig *post_quant_config, std::string value) {
  MS_ASSERT(post_quant_config != nullptr);
  std::for_each(value.begin(), value.end(), ::tolower);
  if (value == "true") {
    post_quant_config->bias_correction = true;
  }
}

STATUS ParseConfigFile(std::string config_file, PostQuantConfig *post_quant_config) {
  MS_ASSERT(post_quant_config != nullptr);

  if (config_file.empty() || config_file.length() > PATH_MAX) {
    MS_LOG(ERROR) << "invalid config path!";
    return RET_PARAM_INVALID;
  }
  // check whether config file path is valid
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "New an object failed.";
    return RET_ERROR;
  }
#ifdef _WIN32
  if (_fullpath(resolved_path.get(), config_file.c_str(), 1024) != nullptr) {
    config_file = string(resolved_path.get());
  }
#else
  if (realpath(config_file.c_str(), resolved_path.get()) != nullptr) {
    config_file = string(resolved_path.get());
  }
#endif
  std::ifstream fs(config_file.c_str(), std::ifstream::in);
  if (!fs.is_open()) {
    MS_LOG(ERROR) << "config file open failed: " << config_file;
    return RET_PARAM_INVALID;
  }

  std::string INPUT_SHAPES = "input_shapes";
  std::string IMAGE_PATH = "image_path";
  std::string BATCH_COUNT = "batch_count";
  std::string THREAD_NUM = "thread_num";
  std::string METHOD_X = "method_x";
  std::string MIXED = "mixed";
  std::string MEAN_ERROR_THRESHOLD = "mean_error_threshold";
  std::string BIAS_CORRECTION = "bias_correction";

  std::map<std::string, std::function<void(PostQuantConfig *, std::string)>> value_parser;
  value_parser[INPUT_SHAPES] = ParseInputShape;
  value_parser[IMAGE_PATH] = ParseImagePath;
  value_parser[BATCH_COUNT] = ParseBatchCount;
  value_parser[THREAD_NUM] = ParseThreadNum;
  value_parser[METHOD_X] = ParseMethodX;
  value_parser[MIXED] = ParseMixed;
  value_parser[MEAN_ERROR_THRESHOLD] = ParseMeanErrorThreshold;
  value_parser[BIAS_CORRECTION] = ParseBiasCorrection;

  std::string line;
  while (std::getline(fs, line)) {
    Trim(&line);
    if (line.empty()) {
      continue;
    }
    auto index = line.find('=');
    if (index == std::string::npos) {
      MS_LOG(ERROR) << "the config file is invalid, can not find '=', please check";
      return RET_PARAM_INVALID;
    }
    auto key = line.substr(0, index);
    auto value = line.substr(index + 1);
    Trim(&key);
    Trim(&value);
    auto it = value_parser.find(key);
    if (it != value_parser.end()) {
      it->second(post_quant_config, value);
    } else {
      MS_LOG(WARNING) << "unsupported parameter: " << key;
    }
  }

  for (const auto &path : post_quant_config->image_paths) {
    MS_LOG(DEBUG) << "calibration data_path: " << path;
  }
  MS_LOG(DEBUG) << "batch_count: " << post_quant_config->batch_count << "\n"
                << "method_x: " << post_quant_config->method_x << "\n"
                << "thread_num: " << post_quant_config->thread_num << "\n"
                << "bias_correction: " << post_quant_config->bias_correction << "\n"
                << "mixed: " << post_quant_config->mixed << "\n"
                << "mean_error_threshold: " << post_quant_config->mean_error_threshold;
  post_quant_config->inited = true;
  fs.close();
  return RET_OK;
}

SessionModel CreateSessionByFuncGraph(const FuncGraphPtr &func_graph, const converter::Flags &flags, int thread_num) {
  SessionModel sm;
  auto meta_graph = Export(func_graph, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph failed";
    return sm;
  }

  // transform
  GraphDefTransform fb_transform;
  fb_transform.SetGraphDef(meta_graph);
  auto status = fb_transform.Transform(flags);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed";
    return sm;
  }
  meta_graph->version = Version();

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  auto size = builder.GetSize();
  auto *content = reinterpret_cast<const char *>(builder.GetBufferPointer());
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer return null";
    return sm;
  }
  auto model = lite::Model::Import(content, size);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return sm;
  }

  Context ctx;
  ctx.thread_num_ = thread_num;

  auto session = session::LiteSession::CreateSession(&ctx);
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed.";
    return sm;
  }

  status = session->CompileGraph(model);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CompileGraph error";
    return sm;
  }
  model->Free();
  delete meta_graph;
  sm.session = session;
  sm.model = model;
  return sm;
}

STATUS CollectCalibInputs(const std::vector<std::string> &input_dirs, size_t count_limited,
                          std::vector<std::vector<std::string>> *inputs) {
  if (inputs == nullptr) {
    MS_LOG(ERROR) << "inputs is null";
    return RET_ERROR;
  }
  auto AddImage = [&inputs](const std::string &file, size_t index) {
    if (index >= inputs->size()) {
      MS_LOG(ERROR) << "images_ size: " << inputs->size() << " but input index: " << index;
      return;
    }
    struct stat buf {};
    if (stat(file.c_str(), &buf) == 0) {
      inputs->at(index).push_back(file);
    } else {
      MS_LOG(WARNING) << "invalid image file path: " << file;
    }
  };

  inputs->resize(input_dirs.size());
  auto input_i = 0;
  bool multi_input = input_dirs.size() > 1;
  for (const auto &image_path : input_dirs) {
    DIR *root = opendir(image_path.c_str());
    if (root == nullptr) {
      MS_LOG(ERROR) << "invalid image path: " << image_path;
      return RET_PARAM_INVALID;
    }
    struct dirent *image_dir = readdir(root);
    size_t count = 0;
    while (image_dir != nullptr) {
      string file_name(image_dir->d_name);
      if (file_name != "." && file_name != "..") {
        const std::string file_path = image_path + "/" + file_name;
        if (multi_input || count == 0) {
          AddImage(file_path, input_i);
          count++;
        } else if (count < count_limited) {
          AddImage(file_path, input_i);
          count++;
        } else {
          break;
        }
      }
      image_dir = readdir(root);
    }
    std::sort(inputs->at(input_i).begin(), inputs->at(input_i).end());
    if (count_limited != 0 && count_limited < inputs->at(input_i).size()) {
      inputs->at(input_i).resize(count_limited);
    }
    closedir(root);
    input_i++;
  }
  return RET_OK;
}

STATUS CopyInputDataToTensor(size_t input_index, size_t image_index,
                             const std::vector<std::vector<std::string>> &images, mindspore::tensor::MSTensor *tensor) {
  MS_ASSERT(tensor != nullptr);
  if (input_index >= images.size()) {
    MS_LOG(ERROR) << "images_ size: " << images.size() << " but input_index: " << input_index;
    return RET_ERROR;
  }
  if (image_index >= images[input_index].size()) {
    MS_LOG(ERROR) << "images_[input_index] size: " << images[input_index].size() << " but image_index: " << image_index;
    return RET_ERROR;
  }
  string path = images[input_index][image_index];
  MS_LOG(INFO) << "read image: " << path;
  size_t size;
  char *bin_buf = ReadFile(path.c_str(), &size);
  if (bin_buf == nullptr) {
    MS_LOG(ERROR) << "ReadFile return nullptr";
    return RET_NULL_PTR;
  }
  auto data = tensor->MutableData();
  if (data == nullptr) {
    MS_LOG(ERROR) << "Get tensor MutableData return nullptr";
    return RET_NULL_PTR;
  }
  if (size != tensor->Size()) {
    MS_LOG(ERROR) << "the input data is not consistent with model input, file_size: " << size
                  << " input tensor size: " << tensor->Size();
    return RET_ERROR;
  }
  auto ret = memcpy_s(data, tensor->Size(), bin_buf, size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error: " << ret;
    delete[] bin_buf;
    return RET_ERROR;
  }
  delete[] bin_buf;
  return RET_OK;
}

FuncGraphPtr CopyFuncGraph(const FuncGraphPtr &func_graph) {
  Cloner cloner({func_graph}, true, true, true, std::make_shared<TraceCopy>(), nullptr);
  auto new_func_graph = cloner[func_graph];

  std::map<std::string, CNodePtr> old_cnode_map;
  for (const auto &cnode : func_graph->GetOrderedCnodes()) {
    old_cnode_map[cnode->fullname_with_scope()] = cnode;
  }

  for (auto &cnode : new_func_graph->GetOrderedCnodes()) {
    auto cnode_name = cnode->fullname_with_scope();
    auto old_cnode_iter = old_cnode_map.find(cnode_name);
    if (old_cnode_iter == old_cnode_map.end()) {
      MS_LOG(ERROR) << "can not find node: " << cnode_name;
      return nullptr;
    }
    auto old_cnode = old_cnode_iter->second;
    auto inputs = cnode->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
      auto input_node = inputs[i];
      if (input_node->isa<Parameter>()) {
        auto param_node = input_node->cast<ParameterPtr>();
        if (param_node->has_default()) {
          ParamValueLitePtr old_param_value = std::static_pointer_cast<ParamValueLite>(param_node->default_param());
          auto new_param_value = std::make_shared<ParamValueLite>();

          auto copyed_data = malloc(old_param_value->tensor_size());
          if (copyed_data == nullptr) {
            MS_LOG(ERROR) << "malloc data error, size: " << old_param_value->tensor_size();
            return nullptr;
          }
          memcpy(copyed_data, old_param_value->tensor_addr(), old_param_value->tensor_size());

          new_param_value->set_tensor_size(old_param_value->tensor_size());
          new_param_value->set_tensor_addr(copyed_data);
          new_param_value->set_tensor_shape(old_param_value->tensor_shape());
          new_param_value->set_format(old_param_value->format());
          new_param_value->set_tensor_type(old_param_value->tensor_type());

          param_node->set_default_param(new_param_value);
        }

        auto old_abstract_base = param_node->abstract();
        if (!utils::isa<abstract::AbstractTensorPtr>(old_abstract_base)) {
          MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, " << param_node->name();
          return nullptr;
        }
        auto old_abstract = utils::cast<abstract::AbstractTensorPtr>(old_abstract_base);
        auto new_abstract = std::make_shared<abstract::AbstractTensor>(old_abstract->element()->GetTypeTrack(),
                                                                       old_abstract->GetShapeTrack());
        param_node->set_abstract(new_abstract);
      }
    }  // end inputs loop
  }    // end cnodes loop
  return new_func_graph;
}

void GetLiteParameter(const AnfNodePtr &node, ParameterPtr *param_node, ParamValueLitePtr *param_value) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(param_node != nullptr);
  MS_ASSERT(param_value != nullptr);

  auto op_name = node->fullname_with_scope();

  *param_node = node->cast<ParameterPtr>();
  if (*param_node == nullptr) {
    MS_LOG(INFO) << op_name << " can not cast to ParameterPtr";
    return;
  }
  if (!(*param_node)->has_default()) {
    MS_LOG(INFO) << op_name << " not has_default";
    return;
  }

  *param_value = std::static_pointer_cast<ParamValueLite>((*param_node)->default_param());
  if (*param_value == nullptr) {
    MS_LOG(INFO) << "default_param can not cast to ParamValueLite";
    return;
  }
}

STATUS UpdateTensorDataAndSize(ParamValueLitePtr weight, void *quant_datas, int new_size) {
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(new_size > 0);
  delete[] reinterpret_cast<char *>(weight->tensor_addr());
  char *new_tensor_data = new (std::nothrow) char[new_size];
  if (new_tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data error";
    return RET_ERROR;
  }
  memcpy(new_tensor_data, quant_datas, new_size);

  weight->set_tensor_size(new_size);
  weight->set_tensor_addr(new_tensor_data);
  return RET_OK;
}

void GetMaxMinPerchannel(int channels, int one_filter_size, int i, int elem_count, const float *raw_datas,
                         bool channel_at_first, float *desired_max, float *desired_min) {
  float min = FLT_MAX;
  float max = -FLT_MAX;
  // find min and max
  for (int j = 0; j < one_filter_size; j++) {
    auto index = j + i * one_filter_size;
    if (!channel_at_first) {
      index = j * channels + i;
    }
    if (index >= elem_count) {
      MS_LOG(ERROR) << "over flow!";
    }
    min = std::min(min, raw_datas[index]);
    max = std::max(max, raw_datas[index]);
  }
  *desired_max = max;
  *desired_min = min;
}

}  // namespace mindspore::lite::quant
