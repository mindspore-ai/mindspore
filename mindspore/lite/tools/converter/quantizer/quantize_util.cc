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
#include <memory>
#include <vector>
#include <set>
#include <functional>
#include "include/version.h"
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
const int kLstmInputWeightIndex = 1;
const int kLstmStateWeightIndex = 2;
const int kLstmWeightShapeSize = 3;
const int kSingleDirBiasTensorSize = 4;
const int kLstmBiasShapeSize = 2;
const int kLstmBiasIndex = 3;

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

SessionModel CreateSessionByFuncGraph(const FuncGraphPtr &func_graph, const converter::Flags &flags, int thread_num,
                                      int *size) {
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
  *size = builder.GetSize();
  auto *content = reinterpret_cast<const char *>(builder.GetBufferPointer());
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer return null";
    return sm;
  }
  auto model = lite::Model::Import(content, *size);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return sm;
  }
  Context ctx;
  ctx.thread_num_ = thread_num;
  MS_ASSERT(!ctx.device_list_.empty());
  ctx.device_list_.front().device_info_.cpu_device_info_.cpu_bind_mode_ = HIGHER_CPU;
  auto session = session::LiteSession::CreateSession(&ctx);
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed.";
    model->Free();
    delete meta_graph;
    delete model;
    return sm;
  }

  status = session->CompileGraph(model);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CompileGraph error";
    model->Free();
    delete meta_graph;
    delete session;
    delete model;
    return sm;
  }
  delete meta_graph;
  sm.session = session;
  sm.model = model;
  return sm;
}

SessionModel CreateSessionByFuncGraph(const FuncGraphPtr &func_graph, const converter::Flags &flags, int thread_num) {
  int size = 0;
  return CreateSessionByFuncGraph(func_graph, flags, thread_num, &size);
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

int UpdateTensorDataAndSize(const AnfNodePtr &node, const tensor::TensorPtr &weight, void *quant_datas, int new_size,
                            TypeId new_data_type) {
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
  // set dtype
  auto abstract_base = node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of node is nullptr, " << node->fullname_with_scope();
    return RET_NULL_PTR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of node should be anstract tensor, " << node->fullname_with_scope();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  CHECK_NULL_RETURN(abstract_tensor);
  CHECK_NULL_RETURN(abstract_tensor->element());
  abstract_tensor->element()->set_type(TypeIdToType(new_data_type));
  return RET_OK;
}

int GetMatMulPreferredDim(const PrimitivePtr &primitive, int input_index, const std::vector<int> &dims) {
  size_t last_first_index = dims.size() - 1;
  size_t last_second_index = dims.size() - 2;
  auto matmul_prim = primitive->cast<std::shared_ptr<ops::MatMul>>();
  MS_ASSERT(matmul_prim != nullptr);
  // For MatMul A
  if (input_index == 0) {
    if (matmul_prim->GetAttr(ops::kTransposeA) != nullptr && matmul_prim->get_transpose_a()) {
      return last_first_index;
    } else {
      return last_second_index;
    }
  }
  // For MatMul B
  if (input_index == 1) {
    if (matmul_prim->GetAttr(ops::kTransposeB) != nullptr && matmul_prim->get_transpose_b()) {
      return last_second_index;
    } else {
      return last_first_index;
    }
  }
  return 0;
}

int CalChannels(const std::vector<int> &dims, int channel_cnt, bool *channel_at_first) {
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

int GetPreferredDim(const PrimitivePtr &primitive, int input_index, const std::vector<int> &dims) {
  if (primitive->name() == ops::kNameMatMul) {
    return GetMatMulPreferredDim(primitive, input_index, dims);
  }
  // The first index.
  return 0;
}

std::vector<int> ConvertShapeVectorToInt32(const ShapeVector &dims) {
  std::vector<int> shape;
  for (auto dim : dims) {
    if (dim > INT32_MAX || dim < INT32_MIN) {
      MS_LOG(ERROR) << dim << " over int32 range.";
      shape.push_back(-1);
    } else {
      shape.push_back(dim);
    }
  }
  return shape;
}

void CalQuantAssitInfo(const schema::PrimitiveT &primitive, const std::vector<int> &shapes, int index,
                       bool *channel_at_first, int *channel_cnt) {
  MS_ASSERT(primitive != nullptr);
  if (shapes.empty()) {
    MS_LOG(ERROR) << " shape vector is empty.";
    return;
  }
  if (primitive.value.type == schema::PrimitiveType_MatMul && static_cast<int>(shapes.size()) == DIMENSION_2D) {
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

int MixedBitQuantFilter(const AnfNodePtr &node, const tensor::TensorPtr &weight, const PrimitivePtr &primitive,
                        QuantType quant_type, WeightQuantType weight_quant_type, TypeId quant_data_type,
                        double init_scale, int index) {
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
  if (weight_quant_type != MIXED_BIT_PER_LAYER) {
    MS_LOG(ERROR) << "Unsupported weight quant type:" << weight_quant_type;
    return RET_ERROR;
  }
  MixedBitWeightQuantizer quantizer(init_scale);
  auto ret =
    quantizer.DoQuantization(static_cast<float *>(weight->data_c()), weight->shape_c(), 0, &quant_params, &quant_data);
  if (ret == RET_NO_CHANGE) {
    const int quant_min = QuantMin(k8Bit, false, false);  // -128
    const int quant_max = QuantMax(k8Bit);                // 127
    MS_LOG(WARNING)
      << node->fullname_with_scope()
      << " mixed bit quantization search failed, the current layer rolls back to 8 bit fixed quantization.";
    return FixedBitQuantFilter<int8_t>(node, weight, primitive, QuantType_QUANT_WEIGHT, quant_max, quant_min, k8Bit,
                                       FIXED_BIT_PER_CHANNEL, kNumberTypeInt8, index);
  }
  if (ret != RET_OK) {
    return ret;
  }

  auto status =
    UpdateTensorDataAndSize(node, weight, quant_data.data(), quant_data.size() * sizeof(int16_t), quant_data_type);
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
  quant_param_holder->set_quant_type(quant_type);
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
