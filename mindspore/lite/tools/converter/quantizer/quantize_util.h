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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZE_UTIL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZE_UTIL_H_

#ifndef _MSC_VER
#include <dirent.h>
#endif
#include <sys/stat.h>
#include <memory>
#include <string>
#include <cmath>
#include <set>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include <map>
#include <functional>
#include "ops/mat_mul.h"
#include "ops/lstm.h"
#include "ops/fusion/full_connection.h"
#include "tools/converter/quantizer/quantizer.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "tools/converter/quantizer/huffman_encode.h"
#include "tools/converter/quantizer/bitpacking.h"
#include "tools/converter/quantizer/mixed_bit_weight_quantizer.h"
#include "src/lite_session.h"
#include "tools/converter/graphdef_transform.h"
#include "src/common/file_utils.h"
#include "src/common/quant_utils.h"

namespace mindspore::lite::quant {
enum WeightQuantType {
  FIXED_BIT_PER_CHANNEL = 0,
  FIXED_BIT_PER_LAYER = 1,
  MIXED_BIT_PER_LAYER = 2,
};
constexpr size_t k2Bit = 2;
constexpr size_t k8Bit = 8;
constexpr size_t k10Bit = 10;
constexpr size_t k16Bit = 16;
constexpr size_t k32Bit = 32;
constexpr size_t kMaxNum1024 = 1024;
constexpr float kPercentBase = 100.0;
constexpr size_t kMillisecondsBase = 10;
constexpr float delta = 0.1;
constexpr float ratio = 10.0;
constexpr int percent = 10;

struct SessionModel {
  session::LiteSession *session{nullptr};
  Model *model{nullptr};
};

QuantParamHolderPtr GetCNodeQuantHolder(const PrimitivePtr &primitive);

std::vector<int8_t> KMeans(float *data, size_t elem_count, size_t k, size_t epochs, schema::QuantParamT *quantParam);

int UpdateTensorDataAndSize(const AnfNodePtr &node, const tensor::TensorPtr &weight, void *quant_datas, int new_size,
                            TypeId new_data_type);

void CalQuantAssitInfo(const schema::PrimitiveT &primitive, const std::vector<int> &shapes, int index,
                       bool *channel_at_first, int *channel_cnt);

bool TensorQuantParamsInited(const schema::TensorT &tensor);

int MixedBitQuantFilter(const AnfNodePtr &node, const tensor::TensorPtr &weight, const PrimitivePtr &primitive,
                        QuantType quant_type, WeightQuantType weight_quant_type, TypeId quant_data_type,
                        double init_scale, int index);

int CalChannels(const std::vector<int> &dims, int channel_cnt, bool *channel_at_first);

int GetPreferredDim(const PrimitivePtr &primitive, int input_index, const std::vector<int> &dims);

std::vector<int> ConvertShapeVectorToInt32(const ShapeVector &dims);

int DoParameterBiasQuant(const ParameterPtr &bias, const PrimitivePtr &primitive);

int DeQuantData(mindspore::tensor::MSTensor *tensor, std::vector<double> *dequant_data, int preferred_dim = 0);

int DoBitPack(const size_t &bit_num, schema::TensorT *tensor_input);

template <typename T>
int DeQuantData(const int8_t *tensor_data, int64_t elements_num, std::vector<lite::LiteQuantParam> quant_params,
                std::vector<T> *dequant_data, int preferred_dim = 0) {
  if (quant_params.size() != 1) {
    MS_LOG(ERROR) << "unexpected quant_params size: " << quant_params.size() << " only support per-layer now.";
    return RET_ERROR;
  }
  auto scale = quant_params[0].scale;
  auto zp = quant_params[0].zeroPoint;
  dequant_data->resize(elements_num);
  for (int64_t i = 0; i < elements_num; i++) {
    dequant_data->at(i) = scale * (tensor_data[i] - zp);
  }
  return RET_OK;
}

template <typename T>
int FixedBitQuantFilter(const AnfNodePtr &parameter, const tensor::TensorPtr &weight, const PrimitivePtr &primitive,
                        QuantType quant_type, int quant_max, int quant_min, size_t bit_num,
                        WeightQuantType weight_quant_type, TypeId quant_data_type, int index, bool symmetry = false,
                        bool narrow_range = false, bool k_means = false) {
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(primitive != nullptr);
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

  std::vector<T> quant_data(elem_count);
  int ret = RET_OK;
  if (weight_quant_type == FIXED_BIT_PER_CHANNEL) {
    int preferred_dim = GetPreferredDim(primitive, index, ConvertShapeVectorToInt32(dims));
    ret = DoPerChannelQuant<T>(static_cast<float *>(weight->data_c()), weight->DataSize(),
                               static_cast<mindspore::schema::QuantType>(quant_type), &quant_params, quant_max,
                               quant_min, bit_num, &quant_data, ConvertShapeVectorToInt32(dims), preferred_dim,
                               symmetry, narrow_range, k_means);
    if (ret == RET_NO_CHANGE) {
      return ret;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "Do per channel quant failed.";
      return ret;
    }
  } else if (weight_quant_type == FIXED_BIT_PER_LAYER) {
    ret = DoPerLayerQuant<T>(static_cast<float *>(weight->data_c()), weight->DataSize(), &quant_params, quant_max,
                             quant_min, bit_num, &quant_data, symmetry, narrow_range, k_means);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Do per layer quant failed.";
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported weight quant type:" << weight_quant_type;
  }
  auto status =
    UpdateTensorDataAndSize(parameter, weight, quant_data.data(), quant_data.size() * sizeof(T), quant_data_type);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
    return RET_ERROR;
  }

#ifdef HUFFMAN_ENCODE
  auto huffman_encode = std::make_unique<lite::HuffmanEncode>();
  ret = huffman_encode->DoHuffmanEncode(weight, primitive, quant_datas.data(), bit_num);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do huffman encode failed.";
    return ret;
  }
#endif

  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  quant_param_holder->set_input_quant_param(index, quant_params);
  quant_param_holder->set_quant_type(quant_type);
  return ret;
}

std::string NodePrimitiveType(const CNodePtr &cnode);

SessionModel CreateSessionByFuncGraph(const FuncGraphPtr &func_graph, const converter::Flags &flags, int thread_num);
SessionModel CreateSessionByFuncGraph(const FuncGraphPtr &func_graph, const converter::Flags &flags, int thread_num,
                                      int *size);
void GetLiteParameter(const AnfNodePtr &node, ParameterPtr *param_node, tensor::TensorPtr *tensor_info);

bool CheckNodeInSet(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_primitive_types);

std::string BoolVectorToString(const std::vector<bool> &bool_vec);

template <typename T>
bool IndexingCompress(const std::set<T> &quant_data_set, const std::map<T, size_t> &unique_value_index_map,
                      size_t unique_value_bit, size_t unique_value_cnt, size_t pack_repetition_size_in_byte,
                      size_t bit_num, schema::TensorT *tensor) {
  auto quant_data_array = reinterpret_cast<T *>(tensor->data.data());
  std::vector<T> quant_data(quant_data_array, quant_data_array + tensor->data.size() / sizeof(T));

  std::vector<bool> bits(pack_repetition_size_in_byte * k8Bit);
  size_t index = 0;
  // write unique_value_cnt: bit_num bit for unsigned
  for (size_t i = 0; i < bit_num; i++) {
    bits[index++] = (unique_value_cnt >> (bit_num - i - 1)) & (0x1);
  }
  // write the unique value set: each value has bit_num bit signed
  for (auto unique_value : quant_data_set) {
    for (size_t i = 0; i < bit_num; i++) {
      bits[index++] = ((unique_value + (1 << (bit_num - 1))) >> (bit_num - i - 1)) & (0x1);
    }
  }
  // write the index: each index has unique_value_bit unsigned
  for (auto quant_value : quant_data) {
    for (size_t i = 0; i < unique_value_bit; i++) {
      bits[index++] = (unique_value_index_map.at(quant_value) >> (unique_value_bit - i - 1)) & (0x1);
    }
  }
  if (index > pack_repetition_size_in_byte * k8Bit) {
    MS_LOG(ERROR) << "unexpected index: " << index << " should not be greater than "
                  << pack_repetition_size_in_byte * k8Bit;
    return false;
  }
  // update tensor data
  auto new_data_str = BoolVectorToString(bits);
  auto ret = memcpy_s(tensor->data.data(), tensor->data.size(), new_data_str.c_str(), new_data_str.size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error";
    return false;
  }
  tensor->data.resize(new_data_str.size());

  tensor->weightQunatCompressType = schema::WeightQunatCompressType_INDEXING;
  MS_LOG(DEBUG) << "set WeightQunatCompressType_INDEXING";
  return true;
}

template <typename T>
bool SparsityCompress(const std::set<T> &quant_data_set, const std::map<T, size_t> &unique_value_index_map,
                      size_t unique_value_bit, size_t unique_value_cnt, size_t pack_sparsity_size_in_byte,
                      size_t nz_cnt, size_t coor_best_bit, size_t bit_num, schema::TensorT *tensor) {
  auto quant_data_array = reinterpret_cast<T *>(tensor->data.data());
  std::vector<T> quant_data(quant_data_array, quant_data_array + tensor->data.size() / sizeof(T));
  auto &quant_params = tensor->quantParams;
  auto elem_cnt = quant_data.size();
  auto channel_cnt = quant_params.size();
  MS_CHECK_TRUE_MSG(channel_cnt != 0, false, "div zero.");
  auto elem_perchannel = elem_cnt / channel_cnt;

  std::vector<bool> bits(pack_sparsity_size_in_byte * k8Bit);
  int index = 0;
  // coor_best_bit
  for (size_t i = 0; i < k8Bit; i++) {
    bits[index++] = (coor_best_bit >> (k8Bit - i - 1)) & 0x1;
  }
  // nz_cnt
  for (size_t i = 0; i < k32Bit; i++) {
    bits[index++] = (nz_cnt >> (k32Bit - i - 1)) & 0x1;
  }
  // unique_value cnt
  for (size_t i = 0; i < bit_num; i++) {
    bits[index++] = (unique_value_cnt >> (bit_num - i - 1)) & 0x1;
  }
  // unique_values
  for (auto unique_value : quant_data_set) {
    for (size_t i = 0; i < bit_num; i++) {
      bits[index++] = ((unique_value + (1 << (bit_num - 1))) >> (bit_num - i - 1)) & (0x1);
    }
  }
  // nz values indexing && get coor
  std::vector<size_t> coors(nz_cnt);
  size_t coors_index = 0;
  size_t prev_index = -1;
  for (size_t di = 0; di < elem_cnt; di++) {
    auto cur_channel = di / elem_perchannel;
    auto zp = quant_params[cur_channel]->zeroPoint;
    auto nz_value = quant_data[di];
    if (nz_value != zp || (di - prev_index) >= static_cast<size_t>((1 << coor_best_bit))) {
      MS_ASSERT(coors_index < nz_cnt);
      coors[coors_index++] = di - prev_index - 1;
      prev_index = di;
      for (size_t i = 0; i < unique_value_bit; i++) {
        bits[index++] = (unique_value_index_map.at(nz_value) >> (unique_value_bit - i - 1)) & (0x1);
      }
    }
  }
  // write coor
  for (auto coor : coors) {
    for (size_t i = 0; i < coor_best_bit; i++) {
      bits[index++] = (coor >> (coor_best_bit - i - 1)) & 0x1;
    }
  }
  if ((unsigned int)index > pack_sparsity_size_in_byte * k8Bit) {
    MS_LOG(ERROR) << "unexpected index: " << index << " should not be greater than "
                  << pack_sparsity_size_in_byte * k8Bit;
    return false;
  }
  auto new_data_str = BoolVectorToString(bits);
  auto ret = memcpy_s(tensor->data.data(), tensor->data.size(), new_data_str.c_str(), new_data_str.size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error";
    return false;
  }
  tensor->data.resize(new_data_str.size());

  tensor->weightQunatCompressType = schema::WeightQunatCompressType_SPARSE;
  MS_LOG(INFO) << "set WeightQunatCompressType_SPARSITY";
  return true;
}

template <typename T>
size_t CalCoorBestBit(const std::vector<T> &quant_data, size_t elem_cnt,
                      const std::vector<std::unique_ptr<schema::QuantParamT>> &quant_params, int unique_value_bit,
                      size_t *coor_best_bit) {
  MS_ASSERT(!quant_params.empty());
  size_t best_nn_cnt = 0;
  size_t min_len_in_bit = std::numeric_limits<size_t>::max();
  for (size_t bit = k2Bit; bit <= k10Bit; bit++) {
    // search
    size_t nn_cnt = 0;
    size_t prev_index = -1;
    auto channel_cnt = quant_params.size();
    MS_ASSERT(channel_cnt > 0);
    auto elem_perchannel = elem_cnt / channel_cnt;
    for (size_t i = 0; i < elem_cnt; i++) {
      auto cur_channel = i / elem_perchannel;
      auto zp = quant_params[cur_channel]->zeroPoint;
      if (quant_data[i] != zp || (i - prev_index) >= static_cast<size_t>((1 << bit))) {
        nn_cnt++;
        prev_index = i;
      }
    }

    size_t len_in_bit = nn_cnt * bit + nn_cnt * unique_value_bit;
    if (len_in_bit < min_len_in_bit) {
      min_len_in_bit = len_in_bit;
      *coor_best_bit = bit;
      best_nn_cnt = nn_cnt;
    }
  }
  return best_nn_cnt;
}

template <typename T>
bool PackRepetition(size_t bit_num, schema::TensorT *tensor) {
  if (tensor->weightQunatCompressType != schema::WeightQunatCompressType_NONE) {
    MS_LOG(INFO) << tensor->name << " is shared weight.";
    return true;
  }
  auto quant_data_array = reinterpret_cast<T *>(tensor->data.data());
  std::vector<T> quant_data(quant_data_array, quant_data_array + tensor->data.size() / sizeof(T));
  auto elem_cnt = quant_data.size();
  auto dims = tensor->dims;
  size_t elem_cnt_by_dims = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  if (elem_cnt != elem_cnt_by_dims) {
    MS_LOG(ERROR) << tensor->name << " elem_cnt: " << elem_cnt << " not equal elem_cnt_by_dims: " << elem_cnt_by_dims;
    return false;
  }

  auto &quant_params = tensor->quantParams;

  std::set<T> quant_data_set;
  for (auto quant_value : quant_data) {
    quant_data_set.insert(quant_value);
  }
  std::map<T, size_t> unique_value_index_map;
  auto index = 0;
  for (auto value : quant_data_set) {
    unique_value_index_map[value] = index++;
  }

  auto unique_value_cnt = quant_data_set.size();
  size_t unique_value_bit = ceil(log2(unique_value_cnt));
  auto pack_repetition_size_in_bit = bit_num + bit_num * unique_value_cnt + unique_value_bit * elem_cnt;
  size_t pack_repetition_size_in_byte = ceil(1.0 * pack_repetition_size_in_bit / k8Bit);
  size_t origin_size_in_byte = ceil(1.0 * bit_num * elem_cnt / k8Bit);

  size_t coor_best_bit = 0;
  auto nz_cnt = CalCoorBestBit<T>(quant_data, elem_cnt, quant_params, unique_value_bit, &coor_best_bit);
  // 1. coor_best_bit 2. nz_cnt 3. quant_data_set size 4. unique_values 5. unique_value indexing 6. nz values coord
  const auto pack_sparsity_size_in_bit =
    1 * k8Bit + 4 * k8Bit + bit_num + bit_num * unique_value_cnt + unique_value_bit * nz_cnt + nz_cnt * coor_best_bit;
  size_t pack_sparsity_size_in_byte = ceil(1.0 * pack_sparsity_size_in_bit / k8Bit);
  MS_LOG(DEBUG) << "coor_best_bit: " << coor_best_bit << " ori: " << origin_size_in_byte
                << " indexing: " << pack_repetition_size_in_byte << " sparse: " << pack_sparsity_size_in_byte;
  auto min_byte_need = std::min({origin_size_in_byte, pack_repetition_size_in_byte, pack_sparsity_size_in_byte});
  if (min_byte_need == origin_size_in_byte) {
    return false;
  } else if (min_byte_need == pack_repetition_size_in_byte) {
    MS_LOG(DEBUG) << "from " << origin_size_in_byte << " to " << pack_repetition_size_in_byte;
    return IndexingCompress<T>(quant_data_set, unique_value_index_map, unique_value_bit, unique_value_cnt,
                               pack_repetition_size_in_byte, bit_num, tensor);
  } else if (min_byte_need == pack_sparsity_size_in_byte) {
    MS_LOG(DEBUG) << "from " << origin_size_in_byte << " to " << pack_sparsity_size_in_byte;
    return SparsityCompress<T>(quant_data_set, unique_value_index_map, unique_value_bit, unique_value_cnt,
                               pack_sparsity_size_in_byte, nz_cnt, coor_best_bit, bit_num, tensor);
  } else {
    MS_LOG(DEBUG) << "unexpected: " << min_byte_need << " not in {" << origin_size_in_byte << " "
                  << pack_repetition_size_in_byte << " " << pack_sparsity_size_in_byte << "}";
  }
  return false;
}
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZE_UTIL_H_
