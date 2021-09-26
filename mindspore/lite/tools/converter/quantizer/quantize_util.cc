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
#include <functional>
#include "include/version.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/full_connection.h"
#include "ops/mat_mul.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "tools/converter/quantizer/bitpacking.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "abstract/abstract_value.h"
#include "securec/include/securec.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
const std::vector<std::string> QuantStrategy::conv_types_ = {ops::kNameConv2DFusion, ops::kNameConv2dTransposeFusion};
const std::vector<std::string> QuantStrategy::mul_types_ = {ops::kNameMatMul, ops::kNameFullConnection};
constexpr int kDim2 = 2;
constexpr int kDim4 = 4;

const int kLstmInputWeightIndex = 1;
const int kLstmStateWeightIndex = 2;
const int kLstmWeightShapeSize = 3;
const int kSingleDirBiasTensorSize = 4;
const int kLstmBiasShapeSize = 2;
const int kLstmBiasIndex = 3;

QuantStrategy::QuantStrategy(size_t weight_size, size_t conv_weight_quant_channel_threshold)
    : m_weight_size_(weight_size), m_conv_weight_quant_channel_threshold_(conv_weight_quant_channel_threshold) {}

bool QuantStrategy::CanConvOpQuantized(const CNodePtr &node) const {
  MS_CHECK_TRUE_RET(node != nullptr, false);
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
  MS_ASSERT(paramNode != nullptr);
  auto abstract_base = paramNode->abstract();
  if (abstract_base == nullptr) {
    return false;
  }
  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << paramNode->name();
    return false;
  }
  auto weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  size_t shapeSize = std::accumulate(weight_shape.begin(), weight_shape.end(), 1, std::multiplies<int>());
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

bool QuantStrategy::CanOpFullQuantized(const AnfNodePtr &node) {
  MS_CHECK_TRUE_RET(node != nullptr, false);
  if (!node->isa<mindspore::CNode>()) {
    return false;
  }
  const auto cnode = std::dynamic_pointer_cast<mindspore::CNode>(node);
  MS_ASSERT(cnode != nullptr);
  auto type = NodePrimitiveType(cnode);
  static const std::set<PrimitivePtr> support_int8_ops = {prim::kPrimAddFusion,      prim::kPrimActivation,
                                                          prim::kPrimAvgPoolFusion,  prim::kPrimConcat,
                                                          prim::kPrimConv2DFusion,   prim::kPrimConv2dTransposeFusion,
                                                          prim::kPrimCrop,           prim::kPrimFullConnection,
                                                          prim::kPrimGather,         prim::kPrimLayerNormFusion,
                                                          prim::kPrimMatMul,         prim::kPrimMaxPoolFusion,
                                                          prim::kPrimMulFusion,      prim::kPrimReshape,
                                                          prim::kPrimSplit,          prim::kPrimTranspose,
                                                          prim::kPrimReduceFusion,   prim::kPrimDivFusion,
                                                          prim::kPrimSqrt,           prim::kPrimPowFusion,
                                                          prim::kPrimSubFusion,      prim::kPrimUnsqueeze,
                                                          prim::kPrimLayerNormFusion};
  // The return node does not need to be quantified.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimReturn) || opt::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
    return false;
  }
  // These operators do not need to check the data type.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimShape) || opt::CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
    return true;
  }
  auto is_support_node = CheckNodeInSet(cnode, support_int8_ops);
  if (!is_support_node && type != "Eltwise") {
    MS_LOG(WARNING) << "node:" << cnode->fullname_with_scope() << " type:" << type << " is not support quantization.";
    return false;
  }
  TypeId type_id;
  auto ret = opt::GetDataTypeFromAnfNode(cnode, &type_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fetch DataType from cnode failed.";
    return ret;
  }

  bool is_data_type_fp32 = type_id == kNumberTypeFloat32;
  if (!is_data_type_fp32) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << "  type_id is " << type_id << " , and is not float32.";
  }
  return is_data_type_fp32;
}

bool QuantStrategy::CanMulOpQuantized(const CNodePtr &node) const {
  MS_CHECK_TRUE_RET(node != nullptr, false);
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
  size_t shapeSize = std::accumulate(weight_shape.begin(), weight_shape.end(), 1, std::multiplies<int>());
  if (shapeSize < m_weight_size_) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }
  return true;
}

bool QuantStrategy::CanTensorQuantized(const AnfNodePtr &inputNode) const {
  if (inputNode == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized input is nullptr!";
    return false;
  }
  ParameterPtr paramNode = nullptr;
  if (inputNode->isa<Parameter>()) {
    paramNode = inputNode->cast<ParameterPtr>();
  }
  if (paramNode == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized invalid paramNode!";
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
  MS_ASSERT(weight_shape != nullptr);
  if (weight_shape.size() < kDim2) {  // do not quant single dim tensors
    return false;
  }
  size_t shapeSize = std::accumulate(weight_shape.begin(), weight_shape.end(), 1, std::multiplies<int>());
  if (shapeSize < m_weight_size_) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }
  if (weight_shape.size() == kDim4) {  // assume Convolution
    if (weight_shape[0] <= static_cast<int>(m_conv_weight_quant_channel_threshold_)) {
      MS_LOG(INFO) << "channel less m_conv_weight_quant_channel_threshold_!" << weight_shape[0];
      return false;
    }
  }

  return true;
}

QuantParamHolderPtr GetCNodeQuantHolder(const PrimitivePtr &primitive) {
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  QuantParamHolderPtr quant_params_holder = nullptr;
  auto quant_params_valueptr = primitive->GetAttr("quant_params");
  if (quant_params_valueptr == nullptr) {
    quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
    MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
    primitive->AddAttr("quant_params", quant_params_holder);
  } else {
    quant_params_holder = quant_params_valueptr->cast<QuantParamHolderPtr>();
    if (quant_params_holder == nullptr) {
      quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
      MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
      primitive->AddAttr("quant_params", quant_params_holder);
    }
  }
  return quant_params_holder;
}

bool TensorQuantParamsInited(const schema::TensorT &tensor) {
  if (tensor.quantParams.empty()) {
    return false;
  }

  for (auto &quant_param : tensor.quantParams) {
    if (!quant_param->inited) {
      return false;
    }
  }
  return true;
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
    quantParam->inited = true;
    quantParam->min = mMin;
    quantParam->max = mMax;
    quantParam->scale = 0.0f;
    quantParam->zeroPoint = 0;
    quantParam->narrowRange = narrowRange;
    quantParam->numBits = numBits;
    return RET_OK;
  }

  const int8_t quantMax = (1 << (static_cast<unsigned int>(numBits - 1))) - 1;
  const int8_t quantMin = -1 * (1 << (static_cast<unsigned int>(numBits - 1))) + (narrowRange ? 1 : 0);
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
  MS_ASSERT(!data.empty());
  size_t length = data.size();
  if (max_tmp - data.at(index) < delta) {
    return false;
  }
  if (fabs(max_tmp - *min_tmp) <= 0.0f || fabs(length - *min_idx) <= 0.0f) {
    MS_LOG(INFO) << "divisor cannot be 0";
    return false;
  }
  float range_ratio = (data.at(index) - *min_tmp) / (max_tmp - *min_tmp);
  float index_ratio = static_cast<float>(index - *min_idx) / (length - *min_idx);
  if (fabs(index_ratio) <= 0.0f) {
    MS_LOG(INFO) << "divisor cannot be 0";
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
  MS_ASSERT(!data.empty());
  size_t length = data.size();
  if (data.at(index) - min_tmp < delta) {
    return false;
  }
  if (fabs(*max_tmp - min_tmp) <= 0.0f || fabs(length - *max_idx) <= 0.0f) {
    MS_LOG(INFO) << "divisor cannot be 0";
    return false;
  }
  float range_ratio = (*max_tmp - data.at(index)) / (*max_tmp - min_tmp);
  float index_ratio = static_cast<float>(index - *max_idx) / (length - *max_idx);
  if (fabs(index_ratio) <= 0.0f) {
    MS_LOG(INFO) << "divisor cannot be 0";
    return false;
  }
  if (index_ratio > 0 && range_ratio / index_ratio > ratio) {
    *max_idx = index;
    *max_tmp = data.at(index);
  }
  return true;
}

static float CalPercentile(const std::vector<float> &data, const int &outlier_percent) {
  MS_ASSERT(!data.empty());
  const int size = data.size();
  float val = outlier_percent / kPercentBase * size;
  int index = std::ceil(val);
  float result;
  if (index - val > 0) {
    MS_ASSERT(index - 1 >= 0);
    result = data.at(index - 1);
  } else {
    MS_ASSERT(index - 1 >= 0);
    result = (data.at(index - 1) + data.at(index)) / 2;
  }
  return result;
}

std::pair<float, float> OutlierMethod(std::vector<float> min_datas, std::vector<float> max_datas) {
  MS_ASSERT(!min_datas.empty());
  MS_ASSERT(!max_datas.empty());
  std::sort(max_datas.begin(), max_datas.end());
  std::sort(min_datas.begin(), min_datas.end());
  float min_val = CalPercentile(min_datas, percent);
  float max_val = CalPercentile(max_datas, kPercentBase - percent);
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
  MS_ASSERT(data != nullptr);
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
  float cluster_ratio = static_cast<float>(data_unique.size()) / (k - 1);
  std::sort(data_unique.begin(), data_unique.end());
  for (size_t i = 0; i < k; i++) {
    size_t index = std::floor(i * cluster_ratio);
    if (i * cluster_ratio - index > 0) {
      clusters.emplace_back((data_unique[index] + data_unique[index + 1]) / 2);
    } else {
      clusters.emplace_back(data_unique[index]);
    }
  }
  return clusters;
}

std::vector<int8_t> KMeans(float *data, size_t elem_count, size_t k, size_t epochs, schema::QuantParamT *quantParam) {
  MS_ASSERT(data != nullptr);
  MS_CHECK_TRUE_MSG(elem_count != 0, std::vector<int8_t>{}, "elem_count is zero.");
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

  flatbuffers::FlatBufferBuilder builder(kMaxNum1024);
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
    model->Free();
    delete meta_graph;
    return sm;
  }

  status = session->CompileGraph(model);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CompileGraph error";
    model->Free();
    delete meta_graph;
    delete session;
    return sm;
  }
  model->Free();
  delete meta_graph;
  sm.session = session;
  sm.model = model;
  return sm;
}

FuncGraphPtr CopyFuncGraph(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
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
    for (const auto &input_node : inputs) {
      if (input_node->isa<Parameter>()) {
        auto param_node = input_node->cast<ParameterPtr>();
        if (!param_node->has_default()) {
          MS_LOG(ERROR) << "Param node has no default parameter: " << cnode_name;
          return nullptr;
        }
        auto old_tensor_info = std::static_pointer_cast<tensor::Tensor>(param_node->default_param());
        if (old_tensor_info == nullptr) {
          MS_LOG(ERROR) << "Default param of param node is not a tensor info:" << cnode_name;
          return nullptr;
        }
        auto new_tensor_info = lite::CreateTensorInfo(old_tensor_info->data().data(), old_tensor_info->data().nbytes(),
                                                      old_tensor_info->shape(), old_tensor_info->data_type());
        if (new_tensor_info == nullptr) {
          MS_LOG(ERROR) << "Create tensor info failed";
          return nullptr;
        }
        auto status = lite::InitParameterFromTensorInfo(param_node, new_tensor_info);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "init parameter from tensor info failed";
          return nullptr;
        }
      }
    }  // end inputs loop
  }    // end cnodes loop
  return new_func_graph;
}

void GetLiteParameter(const AnfNodePtr &node, ParameterPtr *param_node, tensor::TensorPtr *tensor_info) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr";
    return;
  }
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

  *tensor_info = std::static_pointer_cast<tensor::Tensor>((*param_node)->default_param());
  if (*tensor_info == nullptr) {
    MS_LOG(INFO) << "default_param can not cast to tensor::Tensor";
    return;
  }
}

STATUS UpdateTensorDataAndSize(const tensor::TensorPtr &weight, void *quant_datas, int new_size, TypeId new_data_type) {
  MS_CHECK_TRUE_RET(weight != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(new_size > 0, RET_NULL_PTR);
  weight->set_data_type(new_data_type);
  if (new_size != weight->data().nbytes()) {
    MS_LOG(ERROR) << "Data size of tensor info is error.";
    return RET_ERROR;
  }
  if (memcpy_s(weight->data_c(), new_size, quant_datas, new_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CalChannels(const ShapeVector &dims, int channel_cnt, bool *channel_at_first) {
  auto channels = dims[0];
  if (!(*channel_at_first)) {
    if (dims.size() != 2) {
      MS_LOG(WARNING) << "unexpected dims size: " << dims.size();
      *channel_at_first = true;
    } else {
      channels = dims[1];
    }
  } else {
    channels = channel_cnt == -1 ? channels : channel_cnt;
  }
  return channels;
}

void CalQuantAssitInfo(const PrimitivePtr &primitive, const ShapeVector &shapes, int index, bool *channel_at_first,
                       int *channel_cnt) {
  MS_ASSERT(primitive != nullptr);
  if (shapes.empty()) {
    MS_LOG(ERROR) << " shape vector is empty.";
    return;
  }
  if (primitive->name() == ops::kNameMatMul && static_cast<int>(shapes.size()) == DIMENSION_2D) {
    auto matmul_prim = primitive->cast<std::shared_ptr<ops::MatMul>>();
    MS_ASSERT(matmul_prim != nullptr);
    *channel_at_first =
      index != 1 || (matmul_prim->GetAttr(ops::kTransposeB) != nullptr && matmul_prim->get_transpose_b());
  } else if (primitive->name() == ops::kNameLSTM) {
    if (index == kLstmInputWeightIndex || index == kLstmStateWeightIndex) {
      if (shapes.size() != kLstmWeightShapeSize) {
        MS_LOG(WARNING) << "unexpected lstm shape size: " << shapes.size();
      } else {
        *channel_cnt = shapes[0] * shapes[1];
      }
    } else if (index == kLstmBiasIndex) {
      if (shapes.size() != kLstmBiasShapeSize) {
        MS_LOG(WARNING) << "unexpected lstm shape size: " << shapes.size();
      } else {
        auto tensor_elem_cnt = shapes[0] * shapes[1];
        if (tensor_elem_cnt % kSingleDirBiasTensorSize == 0) {
          *channel_cnt = kSingleDirBiasTensorSize;
        }
      }
    } else {
      MS_LOG(WARNING) << "unexpected index of lstm: " << index;
    }
  }
}

void CalQuantAssitInfo(const schema::PrimitiveT &primitive, const std::vector<int> &shapes, int index,
                       bool *channel_at_first, int *channel_cnt) {
  MS_ASSERT(primitive != nullptr);
  if (shapes.empty()) {
    MS_LOG(ERROR) << " shape vector is empty.";
    return;
  }
  if (primitive.value.type == schema::PrimitiveType_MatMul && static_cast<int>(shapes.size()) == kDim2) {
    auto matmul_prim = primitive.value.AsMatMul();
    MS_ASSERT(matmul_prim != nullptr);
    *channel_at_first = index != 1 || matmul_prim->transpose_b;
  } else if (primitive.value.type == schema::PrimitiveType_LSTM) {
    if (index == kLstmInputWeightIndex || index == kLstmStateWeightIndex) {
      if (shapes.size() != kLstmWeightShapeSize) {
        MS_LOG(WARNING) << "unexpected lstm shape size: " << shapes.size();
      } else {
        *channel_cnt = shapes[0] * shapes[1];
      }
    } else if (index == kLstmBiasIndex) {
      if (shapes.size() != kLstmBiasShapeSize) {
        MS_LOG(WARNING) << "unexpected lstm shape size: " << shapes.size();
      } else {
        auto tensor_elem_cnt = shapes[0] * shapes[1];
        if (tensor_elem_cnt % kSingleDirBiasTensorSize == 0) {
          *channel_cnt = kSingleDirBiasTensorSize;
        }
      }
    } else {
      MS_LOG(WARNING) << "unexpected index of lstm: " << index;
    }
  }
}

STATUS MixedBitQuantFilter(const tensor::TensorPtr &weight, const PrimitivePtr &primitive, QuantType quant_type,
                           WeightQuantType weight_quant_type, TypeId quant_data_type, double init_scale, int index) {
  MS_CHECK_TRUE_RET(primitive != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(weight != nullptr, RET_NULL_PTR);
  auto dims = weight->shape();
  if (weight_quant_type == FIXED_BIT_PER_CHANNEL) {
    if (dims.size() <= 1) {
      MS_LOG(WARNING) << "dims is " << dims.size() << " can not per_channel";
      weight_quant_type = FIXED_BIT_PER_LAYER;
    }
  }
  std::vector<schema::QuantParamT> quant_params;
  size_t elem_count = weight->DataSize();
  auto *raw_data = static_cast<float *>(weight->data_c());
  if (raw_data == nullptr) {
    MS_LOG(ERROR) << "rawDatas is nullptr";
    return RET_ERROR;
  }

  std::vector<int16_t> quant_data(elem_count);
  int ret = RET_OK;
  if (weight_quant_type == MIXED_BIT_PER_LAYER) {
    MixedBitWeightQuantizer quantizer(init_scale);
    quantizer.DoQuantization(static_cast<float *>(weight->data_c()), weight->shape_c(), 0, &quant_params, &quant_data);
  } else {
    MS_LOG(ERROR) << "Unsupported weight quant type:" << weight_quant_type;
  }
  auto status =
    UpdateTensorDataAndSize(weight, quant_data.data(), quant_data.size() * sizeof(int16_t), quant_data_type);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
    return RET_ERROR;
  }

  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  quant_param_holder->set_input_quant_param(index, quant_params);
  return ret;
}
bool CheckNodeInSet(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_primitive_types) {
  for (const auto &type : support_primitive_types) {
    if (opt::CheckPrimitiveType(cnode, type)) {
      return true;
    }
  }
  return false;
}
}  // namespace mindspore::lite::quant
