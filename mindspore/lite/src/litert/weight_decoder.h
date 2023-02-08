/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_WEIGHT_DECODER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_WEIGHT_DECODER_H_

#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <limits>
#include <string>
#include <cmath>
#include "nnacl/matmul_parameter.h"
#include "nnacl/gather_parameter.h"
#include "src/litert/kernel_exec.h"
#include "src/common/utils.h"
#include "src/tensor.h"
#include "src/litert/lite_model.h"

static constexpr int kPerTensor = 1;
static constexpr int kBitNumMix = 0;
static constexpr int kBitNum1 = 1;
static constexpr int kBitNum8 = 8;
static constexpr int kBitNum16 = 16;
static constexpr int kBitNum32 = 32;

namespace mindspore::lite {

class MS_API WeightDecoder {
 public:
  static int DequantNode(const OpParameter *op_parameter, const std::vector<Tensor *> &in_tensors, TypeId dst_data_type,
                         const std::string &model_version, bool float_mode);
  static int DecompressTensor(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor);

  static int CompareVersion(const std::string &version1, const std::string &version2) {
    std::istringstream iss1(version1);
    std::istringstream iss2(version2);
    std::string string1;
    std::string string2;
    while (!iss1.eof() || !iss2.eof()) {
      getline(iss1, string1, '.');
      getline(iss2, string2, '.');
      if (stoi(string1) > stoi(string2)) return 1;
      if (stoi(string1) < stoi(string2)) return -1;
      string1 = string2 = "0";
    }
    return 0;
  }

  template <typename T>
  static int GetPreferredDim(const std::vector<T *> &in_tensors, const OpParameter *op_parameter, int index,
                             const std::vector<int> &dims, const std::string &model_version) {
#ifndef WEIGHT_DECODE_CLIP
    const int first_version_offset = 15;
    if (model_version.empty() || model_version.substr(0, first_version_offset) != "MindSpore Lite " ||
        CompareVersion(model_version.substr(first_version_offset, model_version.size()), "1.6.0") == -1) {
      return IsChannelFirst(index, op_parameter) ? 0 : 1;
    }
    if (op_parameter->type_ == schema::PrimitiveType_MatMulFusion) {
      return GetMatMulPreferredDim(op_parameter, index, dims);
    } else if (op_parameter->type_ == schema::PrimitiveType_Conv2dTransposeFusion) {
      if (model_version.empty() ||
          CompareVersion(model_version.substr(first_version_offset, model_version.size()), "1.8.0") == -1) {
        return 0;
      }
      return GetDeConvPreferredDim(op_parameter, dims);
    } else if (op_parameter->type_ == schema::PrimitiveType_Gather) {
      return GetGatherPreferredDim(op_parameter, in_tensors);
    }
    // The first index.
    return 0;
#else
    MS_LOG(ERROR) << "Do not support preferred dim.";
    return RET_NOT_SUPPORT;
#endif
  }

  template <typename ST, typename DT = float>
  static DT *DequantData(const lite::Tensor *input_tensor, int preferred_dim) {
#ifndef WEIGHT_DECODE_CLIP
    const auto *quant_datas = static_cast<const ST *>(input_tensor->data());
    if (quant_datas == nullptr) {
      MS_LOG(ERROR) << "Get quant tensor failed.";
      return nullptr;
    }
    auto quant_param = input_tensor->quant_params();
    if (quant_param.size() != kPerTensor) {
      return DequantPerChannelData<ST, DT>(input_tensor, quant_datas, preferred_dim);
    } else {
      return DequantPerLayerData<ST, DT>(input_tensor, quant_datas);
    }
#else
    MS_LOG(ERROR) << "Do not support dequant data.";
    return RET_NOT_SUPPORT;
#endif
  }

#ifndef WEIGHT_DECODE_CLIP

 private:
  static int DequantTensor(Tensor *tensor, int preferred_dim, TypeId dst_data_type = kNumberTypeFloat32);

  static int UnPackToInt(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor);

  static int DecodeHuffmanCode(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor);

  static int UnPack(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor);

  static STATUS SparseDecompress(const SchemaTensorWrapper &src_tensor, Tensor *dst_tensor);

  static std::vector<bool> StringToBitVector(const std::string &str);

  static STATUS IndexingDecompress(const SchemaTensorWrapper &src_tensor, Tensor *dst_tensor);

  static bool IsChannelFirst(int index, const OpParameter *op_parameter);

  // A * stride_a + bucket_index * stride_b + C
  static int GetDataIndex(const std::vector<int> &dims, int preferred_dim, int bucket_index, int bucket_in_index);

  template <typename ST, typename DT = float>
  static DT *DequantPerLayerData(const lite::Tensor *input_tensor, const ST *quant_datas) {
    auto quant_param = input_tensor->quant_params();
    auto input_tensor_element_num = input_tensor->ElementsNum();
    MS_CHECK_GT(input_tensor_element_num, 0, nullptr);
    DT *dequant_datas = static_cast<DT *>(malloc(input_tensor_element_num * sizeof(DT)));
    if (dequant_datas == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return nullptr;
    }
    auto quant_clusters = input_tensor->quant_clusters();
    auto param = quant_param.front();
    auto scale = param.scale;
    auto zero_point = param.zeroPoint;
    for (int64_t j = 0; j < input_tensor_element_num; j++) {
      if (!quant_clusters.empty()) {
        int8_t index = quant_datas[j];
        if (index > INT8_MAX || index < INT8_MIN) {
          MS_LOG(ERROR) << "KMeans param quant is error.";
          free(dequant_datas);
          return nullptr;
        }
        if (abs(index - INT8_MIN) >= static_cast<int>(param.clusters.size())) {
          MS_LOG(ERROR) << "index exceed the boundary of param.clusters";
          free(dequant_datas);
          return nullptr;
        }
        dequant_datas[j] = static_cast<DT>(param.clusters[index - INT8_MIN]);
      } else {
#ifdef ENABLE_ARM32
        volatile float dequant_data = (quant_datas[j] - zero_point) * scale;
        dequant_datas[j] = static_cast<DT>(dequant_data);
#else
        dequant_datas[j] = static_cast<DT>((quant_datas[j] - zero_point) * scale);
#endif
      }
    }
    return dequant_datas;
  }

  template <typename ST, typename DT = float>
  static DT *DequantPerChannelData(const lite::Tensor *input_tensor, const ST *quant_datas, int preferred_dim) {
    auto quant_param = input_tensor->quant_params();
    auto input_tensor_element_num = input_tensor->ElementsNum();
    MS_CHECK_GT(input_tensor_element_num, 0, nullptr);
    DT *dequant_datas = static_cast<DT *>(malloc(input_tensor_element_num * sizeof(DT)));
    if (dequant_datas == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return nullptr;
    }
    auto shapes = input_tensor->shape();
    auto channels = quant_param.size();
    if (channels != static_cast<size_t>(shapes.at(preferred_dim))) {
      MS_LOG(ERROR) << input_tensor->tensor_name() << " shapes at preferred_dim " << preferred_dim << " is "
                    << shapes.at(preferred_dim) << " != channels " << channels;
      free(dequant_datas);
      return nullptr;
    }
    MS_CHECK_GT(channels, 0, nullptr);
    size_t per_channel_size = input_tensor_element_num / channels;
    for (size_t i = 0; i < channels; i++) {
      auto param = quant_param.at(i);
      auto scale = param.scale;
      auto zero_point = param.zeroPoint;
      auto var_corr = param.var_corr;
      auto mean_corr = param.mean_corr;
      if (var_corr < 0 || var_corr > 10) {
        MS_LOG(WARNING) << "unexpected var_corr: " << var_corr;
        var_corr = 1;
      }
      for (size_t j = 0; j < per_channel_size; j++) {
        auto index = GetDataIndex(shapes, preferred_dim, i, j);
#ifdef ENABLE_ARM32
        volatile float dequant_data = (quant_datas[index] - zero_point) * scale * var_corr + mean_corr;
        dequant_datas[index] = static_cast<DT>(dequant_data);
#else
        dequant_datas[index] = static_cast<DT>((quant_datas[index] - zero_point) * scale * var_corr + mean_corr);
#endif
      }
    }
    return dequant_datas;
  }

  static int GetMatMulPreferredDim(const OpParameter *op_parameter, int input_index, const std::vector<int> &dims);

  static int GetDeConvPreferredDim(const OpParameter *op_parameter, const std::vector<int> &dims);

  template <typename T>
  static int GetGatherPreferredDim(const OpParameter *op_parameter, const std::vector<T *> &in_tensors) {
    MS_ASSERT(op_parameter != nullptr);
    const int axis_index = 2;
    const int axis_tensor_size = 3;
    if (in_tensors.size() == axis_tensor_size && in_tensors.at(axis_index)->IsConst()) {
      if (in_tensors.at(axis_index)->data_type() == kNumberTypeInt32) {
        return static_cast<int *>(in_tensors.at(axis_index)->data())[0];
      } else if (in_tensors.at(axis_index)->data_type() == kNumberTypeInt64) {
        return static_cast<int64_t *>(in_tensors.at(axis_index)->data())[0];
      }
    }
    const auto *param = reinterpret_cast<const GatherParameter *>(op_parameter);
    return param->axis_;
  }

  static int DequantWeight(lite::Tensor *input_tensor, int preferred_dim, TypeId dst_data_type = kNumberTypeFloat32);

  static int DecodeKMeansWeight(lite::Tensor *tensor, TypeId dst_data_type);

  template <typename T>
  static int DecodeKMeansData(lite::Tensor *tensor, T **dequant_data) {
    CHECK_NULL_RETURN(dequant_data);
    *dequant_data = static_cast<T *>(malloc(tensor->ElementsNum() * sizeof(T)));
    CHECK_NULL_RETURN(*dequant_data);
    for (int64_t i = 0; i < tensor->ElementsNum(); i++) {
      auto index = static_cast<int8_t *>(tensor->data())[i] - INT8_MIN;
      (*dequant_data)[i] = static_cast<T>(tensor->quant_clusters().at(index));
    }
    return RET_OK;
  }

  template <typename T1, typename T2>
  static void UnPackData(int origin_bit, const T2 &packed_data, std::queue<bool> *unpack_bit_data, void *unpack_int,
                         size_t *count, size_t limit_size, bool is_last) {
    T2 uint_result = 0;
    T1 result;
    UnPackFromUintToOrigin<T2>(packed_data, unpack_bit_data);
    const int base = 2;
    while (static_cast<int>(unpack_bit_data->size()) >= origin_bit) {
      for (int k = 0; k < origin_bit; k++) {
        bool bit_tmp = unpack_bit_data->front();
        uint_result = (static_cast<size_t>(bit_tmp) << static_cast<unsigned int>(k)) + uint_result;
        unpack_bit_data->pop();
      }
      result = static_cast<T1>(uint_result - static_cast<T2>(pow(base, origin_bit - 1)));
      if (*count >= limit_size) {
        return;
      }
      (static_cast<T1 *>(unpack_int))[*count] = result;
      uint_result = 0;
      (*count)++;
    }
    size_t remainder = unpack_bit_data->size();
    if (is_last && remainder > 0) {
      for (size_t i = 0; i < remainder; i++) {
        bool bit = unpack_bit_data->front();
        uint_result = (static_cast<unsigned int>(bit) << i) + uint_result;
        unpack_bit_data->pop();
      }
      result = static_cast<T1>(uint_result - static_cast<T2>(pow(base, origin_bit - 1)));
      if (*count >= limit_size) {
        return;
      }
      (static_cast<T1 *>(unpack_int))[*count] = result;
    }
  }

  template <typename T1, typename T2>
  static int UnPackUtil(const SchemaTensorWrapper &src_tensor, const size_t &unpack_int_up_limit_size, int origin_bit,
                        void *unpack_int_data) {
    MS_ASSERT(src_tensor.handler() != nullptr);
    MS_ASSERT(src_tensor.data() != nullptr);
    if (src_tensor.data() == nullptr) {
      MS_LOG(ERROR) << "tensor data is null";
      return RET_NULL_PTR;
    }
    auto weight_data = src_tensor.data();
    size_t pack_size =
      src_tensor.handler()->dataType() == kNumberTypeInt8 ? src_tensor.length() : src_tensor.length() / 2;
    std::queue<bool> unpack_bit_data;
    size_t count = 0;
    for (size_t i = 0; i < pack_size; ++i) {
      T2 pack_data = (static_cast<const T2 *>(static_cast<const void *>(weight_data)))[i];
      bool is_last = i == pack_size - 1;
      if (count >= unpack_int_up_limit_size) {
        MS_LOG(ERROR) << "extend unpack_int_up_limit_size, which is " << unpack_int_up_limit_size;
        return RET_ERROR;
      }
      UnPackData<T1, T2>(origin_bit, pack_data, &unpack_bit_data, unpack_int_data, &count, unpack_int_up_limit_size,
                         is_last);
    }
    return RET_OK;
  }

  template <typename T2>
  static void UnPackFromUintToOrigin(const T2 &packed_data, std::queue<bool> *unpack_bit_data) {
    auto n = packed_data;
    size_t bit_count = 0;
    while (bit_count < sizeof(T2) * 8) {
      bool a = n % 2;
      n = n >> 1;
      bit_count++;
      unpack_bit_data->push(a);
    }
  }

  template <typename T>
  static STATUS UnIndexTensorData(const std::vector<int> &unique_values, const std::vector<size_t> &indices,
                                  void *dst_data, size_t dst_data_size) {
    std::vector<T> un_indexed_data;
    for (auto index : indices) {
      if (index >= unique_values.size()) {
        MS_LOG(ERROR) << "index: " << index << " size: " << unique_values.size();
        return RET_ERROR;
      }
      if (unique_values[index] > std::numeric_limits<T>::max() ||
          unique_values[index] < std::numeric_limits<T>::min()) {
        MS_LOG(ERROR) << "data: " << unique_values[index] << " max: " << std::numeric_limits<T>::max()
                      << " min: " << std::numeric_limits<T>::min();
        return RET_ERROR;
      }
      un_indexed_data.push_back(static_cast<T>(unique_values[index]));
    }
    if (un_indexed_data.size() * sizeof(T) != dst_data_size) {
      MS_LOG(ERROR) << "un idnexed data size: " << un_indexed_data.size() * sizeof(T)
                    << " expected by tensor: " << dst_data_size;
      return RET_ERROR;
    }
    memcpy(dst_data, un_indexed_data.data(), un_indexed_data.size() * sizeof(T));

    return RET_OK;
  }

  template <typename T>
  static STATUS UnSparseTensorData(const std::vector<int> &unique_values, const std::vector<size_t> &indices,
                                   const std::vector<size_t> &coors,
                                   const flatbuffers::Vector<flatbuffers::Offset<schema::QuantParam>> *quant_params,
                                   size_t elem_cnt, size_t coor_best_bit, void *dst_data, size_t dst_data_size) {
    std::vector<T> un_sparsed_data;
    size_t data_index = 0;
    auto nz_cnt = indices.size();
    MS_ASSERT(nz_cnt == coors.size());
    auto channel_cnt = quant_params->size();
    MS_CHECK_GT(channel_cnt, 0, RET_ERROR);
    auto elem_perchannel = elem_cnt / channel_cnt;
    MS_CHECK_GT(elem_perchannel, 0, RET_ERROR);
    for (size_t i = 0; i < nz_cnt; i++) {
      auto index = indices[i];
      if (index >= unique_values.size()) {
        MS_LOG(ERROR) << "index: " << index << " size: " << unique_values.size();
        return RET_ERROR;
      }
      auto nz = unique_values[index];
      if (nz > std::numeric_limits<T>::max() || nz < std::numeric_limits<T>::min()) {
        MS_LOG(ERROR) << "data: " << nz << " max: " << std::numeric_limits<T>::max()
                      << " min: " << std::numeric_limits<T>::min();
        return RET_ERROR;
      }
      auto coor = coors[i];
      for (size_t j = 0; j < coor; j++) {
        auto cur_channel = data_index / elem_perchannel;
        auto zp = quant_params->Get(cur_channel)->zeroPoint();
        un_sparsed_data.push_back(zp);
        data_index++;
      }
      un_sparsed_data.push_back(static_cast<T>(unique_values[index]));
      data_index++;
    }
    if (un_sparsed_data.size() * sizeof(T) > dst_data_size) {
      MS_LOG(ERROR) << "un-sparsed data size: " << un_sparsed_data.size() * sizeof(T)
                    << " tensor size: " << dst_data_size;
      return RET_ERROR;
    } else if (un_sparsed_data.size() * sizeof(T) < dst_data_size &&
               (un_sparsed_data.size() + (1 << coor_best_bit) - 1) * sizeof(T) < dst_data_size) {
      MS_LOG(ERROR) << "un-sparsed data size: " << un_sparsed_data.size() * sizeof(T)
                    << " tensor size: " << dst_data_size << " coor_best_bit: " << coor_best_bit;
      return RET_ERROR;
    }

    for (; data_index < dst_data_size / sizeof(T); data_index++) {
      auto cur_channel = data_index / elem_perchannel;
      auto zp = quant_params->Get(cur_channel)->zeroPoint();
      un_sparsed_data.push_back(static_cast<T>(zp));
    }

    memcpy(dst_data, un_sparsed_data.data(), un_sparsed_data.size() * sizeof(T));

    return RET_OK;
  }
#endif
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_WEIGHT_DECODER_H_
