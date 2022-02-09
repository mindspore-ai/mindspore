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
#include "ops/fusion/mat_mul_fusion.h"
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
namespace {
constexpr int kLstmInputWeightIndex = 1;
constexpr int kLstmStateWeightIndex = 2;
constexpr int kLstmWeightShapeSize = 3;
constexpr int kSingleDirBiasTensorSize = 4;
constexpr int kLstmBiasShapeSize = 2;
constexpr int kLstmBiasIndex = 3;
constexpr size_t kBitNumPerByte = 8;

int ComputeBiasDataAndQuantParam(const std::vector<double> &bias_scales, const std::vector<double> &input_scales,
                                 const float *raw_datas, const QuantParamHolderPtr &quant_param_holder,
                                 std::vector<schema::QuantParamT> *quant_params, std::vector<int32_t> *quant_datas) {
  MS_ASSERT(raw_datas != nullptr && quant_param_holder != nullptr);
  MS_ASSERT(quant_params != nullptr && quant_datas != nullptr);
  double bias_scale_tmp;
  const constexpr double quanted_bias_abs_limit = 0.5 * INT32_MAX;
  MS_CHECK_TRUE_MSG(quant_param_holder->get_input_quant_params().size() > 1, RET_ERROR, "invalid access.");
  auto weight_quant_params = quant_param_holder->get_input_quant_params().at(1);
  auto shape_size = quant_datas->size();
  if (bias_scales.size() == shape_size) {
    for (size_t i = 0; i < shape_size; i++) {
      bias_scale_tmp = bias_scales[i];
      if (fabs(bias_scale_tmp) <= 0.0f) {
        MS_LOG(ERROR) << "divisor 'bias_scale_tmp' cannot be 0.";
        return RET_ERROR;
      }
      if (std::abs(raw_datas[i] / bias_scale_tmp) >= quanted_bias_abs_limit) {
        MS_LOG(DEBUG) << "quanted bias over flow, maybe the scale of weight: " << weight_quant_params[i].scale
                      << " is too small, need to update";
        // update filter scale and zp
        double activate_scale = input_scales[0];
        double filter_scale = std::abs(raw_datas[i]) / (activate_scale * quanted_bias_abs_limit);
        weight_quant_params[i].scale = filter_scale;
        weight_quant_params[i].zeroPoint = 0;
        quant_param_holder->set_input_quant_param(1, weight_quant_params);
        bias_scale_tmp = std::abs(raw_datas[i]) / quanted_bias_abs_limit;
        quant_params->at(i).scale = bias_scale_tmp;
        MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
      }
      auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
      quant_datas->at(i) = quant_data;
    }
    return RET_OK;
  } else if (bias_scales.size() == 1) {
    // for fc, per tensor quant
    bias_scale_tmp = quant_params->front().scale;
    float max_raw_data = 0.0f;
    for (size_t i = 0; i < shape_size; i++) {
      if (std::abs(raw_datas[i]) > max_raw_data) {
        max_raw_data = std::abs(raw_datas[i]);
      }
    }
    if (fabs(bias_scale_tmp) <= 0.0f) {
      MS_LOG(ERROR) << "divisor 'bias_scale_tmp' cannot be 0.";
      return RET_ERROR;
    }
    if (std::abs(max_raw_data / bias_scale_tmp) >= quanted_bias_abs_limit) {
      MS_LOG(DEBUG) << "quanted bias over flow, maybe the scale of weight: " << weight_quant_params[0].scale
                    << " is too small, need to update";
      double activate_scale = input_scales[0];
      MS_CHECK_TRUE_MSG(activate_scale != 0, RET_ERROR, "activate_scale == 0");
      double filter_scale = std::abs(max_raw_data) / (activate_scale * quanted_bias_abs_limit);
      weight_quant_params[0].scale = filter_scale;
      weight_quant_params[0].zeroPoint = 0;
      quant_param_holder->set_input_quant_param(1, weight_quant_params);
      bias_scale_tmp = max_raw_data / quanted_bias_abs_limit;
      quant_params->front().scale = bias_scale_tmp;
      MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
    }
    for (size_t i = 0; i < shape_size; i++) {
      auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
      quant_datas->at(i) = quant_data;
    }
    return RET_OK;
  }
  MS_LOG(ERROR) << "unexpected input_scales size: " << input_scales.size()
                << " weight_scales size: " << weight_quant_params.size();
  return RET_ERROR;
}
}  // namespace

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
    delete meta_graph;
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
    delete meta_graph;
    return sm;
  }
  auto model = lite::Model::Import(content, *size);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    delete meta_graph;
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
  auto matmul_prim = primitive->cast<std::shared_ptr<ops::MatMulFusion>>();
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
  if (primitive->name() == ops::kNameMatMulFusion) {
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
  if (primitive.value.type == schema::PrimitiveType_MatMulFusion && static_cast<int>(shapes.size()) == DIMENSION_2D) {
    auto matmul_prim = primitive.value.AsMatMulFusion();
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

std::string BoolVectorToString(const std::vector<bool> &bool_vec) {
  size_t size_in_byte = ceil(bool_vec.size() / kBitNumPerByte);
  std::string str(size_in_byte, '\0');
  auto iter = str.begin();
  size_t shift = kBitNumPerByte;
  for (bool bit : bool_vec) {
    *iter |= bit << (shift - 1);
    if (--shift == 0) {
      iter++;
      shift = kBitNumPerByte;
    }
  }
  return str;
}

int DoParameterBiasQuant(const ParameterPtr &bias, const PrimitivePtr &primitive) {
  CHECK_NULL_RETURN(bias);
  CHECK_NULL_RETURN(primitive);
  auto bias_default_param = bias->default_param();
  auto bias_param = bias_default_param->cast<tensor::TensorPtr>();
  MS_ASSERT(bias_parameter != nullptr);
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
  auto active_weight_quant_params = quant_param_holder->get_input_quant_params();

  auto active_params = active_weight_quant_params.at(FIRST_INPUT);
  auto weight_params = active_weight_quant_params.at(SECOND_INPUT);

  vector<double> input_scales;
  vector<double> filter_scales;
  vector<double> bias_scales;
  size_t sizeX = active_params.size();
  for (size_t i = 0; i < sizeX; i++) {
    input_scales.emplace_back(active_params[i].scale);
  }
  size_t sizeY = weight_params.size();
  if (sizeX != sizeY) {
    if (sizeX > 1 && sizeY > 1) {
      MS_LOG(ERROR) << "input and filter's scale count cannot match!";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < sizeY; i++) {
    filter_scales.emplace_back(weight_params[i].scale);
  }
  size_t size = std::max(sizeX, sizeY);
  for (size_t i = 0; i < size; i++) {
    auto scaleX = sizeX > 1 ? input_scales[i] : input_scales[0];
    auto scaleY = sizeY > 1 ? filter_scales[i] : filter_scales[0];
    bias_scales.push_back(scaleX * scaleY);
  }
  MS_ASSERT(!bias_scales.empty());
  size_t shape_size = bias_param->DataSize();

  // set bias quant param
  std::vector<schema::QuantParamT> quant_params;
  for (double bias_scale : bias_scales) {
    schema::QuantParamT quant_param;
    if (bias_scale == 0) {
      MS_LOG(WARNING) << "bias_scale == 0";
      quant_param.scale = 1;
    } else {
      quant_param.scale = bias_scale;
    }
    quant_param.numBits = k32Bit;
    quant_param.zeroPoint = 0;
    quant_param.inited = true;
    quant_params.emplace_back(quant_param);
  }
  // quant bias data
  std::vector<int32_t> quant_datas(shape_size);

  auto *raw_datas = static_cast<float *>(bias_param->data_c());
  if (ComputeBiasDataAndQuantParam(bias_scales, input_scales, raw_datas, quant_param_holder, &quant_params,
                                   &quant_datas) != RET_OK) {
    MS_LOG(ERROR) << "compute bias data failed.";
    return RET_ERROR;
  }
  quant_param_holder->set_input_quant_param(THIRD_INPUT, quant_params);
  auto ret = SetTensorData(bias_param, quant_datas.data(), shape_size * sizeof(int32_t));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "set tensor data failed.";
    return RET_ERROR;
  }
  // set dtype
  auto abstractBase = bias->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << bias->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << bias->name();
    return RET_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  if (abstractTensor == nullptr || abstractTensor->element() == nullptr) {
    MS_LOG(ERROR) << "abstractTensor is nullptr" << bias->name();
    return RET_NULL_PTR;
  }
  abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
  return RET_OK;
}

int DeQuantData(mindspore::tensor::MSTensor *tensor, std::vector<double> *dequant_data, int preferred_dim) {
  return DeQuantData(static_cast<int8_t *>(tensor->data()), tensor->ElementsNum(), tensor->quant_params(), dequant_data,
                     preferred_dim);
}

int DoBitPack(const size_t &bit_num, schema::TensorT *tensor_input) {
  if (bit_num > 0 && bit_num < k8Bit) {
    std::vector<int8_t> origin_data(tensor_input->data.size());
    auto status = memcpy_s(origin_data.data(), origin_data.size() * sizeof(int8_t), tensor_input->data.data(),
                           tensor_input->data.size() * sizeof(uint8_t));
    if (status != EOK) {
      MS_LOG(ERROR) << tensor_input->name << " memcpy failed. " << status;
      return RET_ERROR;
    }
    std::vector<uint8_t> pack_data{};
    BitPack::BitPacking<int8_t, uint8_t>(bit_num, origin_data, &pack_data);
    tensor_input->data.resize(pack_data.size() * sizeof(uint8_t));
    status = memcpy_s(tensor_input->data.data(), tensor_input->data.size() * sizeof(uint8_t), pack_data.data(),
                      pack_data.size() * sizeof(uint8_t));
    if (status != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed. " << status;
      return RET_ERROR;
    }
  } else if (bit_num > k8Bit && bit_num < k16Bit) {
    auto shape_size =
      std::accumulate(tensor_input->dims.begin(), tensor_input->dims.end(), size_t(1), std::multiplies<size_t>());
    std::vector<int16_t> origin_data(shape_size);
    auto status = memcpy_s(origin_data.data(), origin_data.size() * sizeof(int16_t), tensor_input->data.data(),
                           tensor_input->data.size() * sizeof(uint8_t));
    if (status != EOK) {
      MS_LOG(ERROR) << "memcpy failed. " << status;
      return RET_ERROR;
    }
    std::vector<uint16_t> pack_data{};
    BitPack::BitPacking<int16_t, uint16_t>(bit_num, origin_data, &pack_data);
    tensor_input->data.resize(pack_data.size() * sizeof(uint16_t));
    status = memcpy_s(tensor_input->data.data(), tensor_input->data.size() * sizeof(uint8_t), pack_data.data(),
                      pack_data.size() * sizeof(uint16_t));
    if (status != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed. " << status;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
