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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_DEQUANT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_DEQUANT_H_

#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <cmath>
#include "nnacl/matmul_parameter.h"
#include "src/lite_kernel.h"
#include "src/common/utils.h"
#include "src/tensor.h"

static constexpr int kPerTensor = 1;

namespace mindspore::lite {
class WeightDecoder {
 public:
  static int UnPackToInt(const schema::Tensor &src_tensor, lite::Tensor *dst_tensor);

  static int DecodeHuffmanCode(const schema::Tensor &src_tensor, lite::Tensor *dst_tensor);

  static int DequantNode(OpParameter *op_parameter, const std::vector<Tensor *> &in_tensors, TypeId dst_data_type);

 private:
  static int DequantTensor(Tensor *tensor, bool channel_first = true, TypeId dst_data_type = kNumberTypeFloat32);

  template <typename ST, typename DT = float>
  static DT *DequantData(lite::Tensor *input_tensor, bool channel_first = true) {
    const auto *quant_datas = static_cast<const ST *>(input_tensor->data_c());
    if (quant_datas == nullptr) {
      MS_LOG(ERROR) << "Get quant tensor failed.";
      return nullptr;
    }
    DT *dequant_datas = static_cast<DT *>(malloc(input_tensor->ElementsNum() * sizeof(DT)));
    if (dequant_datas == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return nullptr;
    }
    auto quant_param = input_tensor->quant_params();
    if (quant_param.size() != kPerTensor) {
      auto shapes = input_tensor->shape();
      auto channels = quant_param.size();
      if (!channel_first) {
        if (static_cast<int>(shapes.size()) != 2 || shapes[1] != static_cast<int>(channels)) {
          MS_LOG(ERROR) << "shape size: " << shapes.size() << " quant params size: " << channels;
          free(dequant_datas);
          return nullptr;
        }
      }

      size_t per_channel_size = input_tensor->ElementsNum() / channels;
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
          auto index = per_channel_size * i + j;
          if (!channel_first) {
            index = channels * j + i;
          }
          auto dequant_data = (quant_datas[index] - zero_point) * scale;
          dequant_datas[index] = static_cast<DT>(dequant_data * var_corr + mean_corr);
        }
      }
    } else {
      auto quant_clusters = input_tensor->quant_clusters();
      auto param = quant_param.front();
      auto scale = param.scale;
      auto zero_point = param.zeroPoint;
      for (int64_t j = 0; j < input_tensor->ElementsNum(); j++) {
        if (!quant_clusters.empty()) {
          int8_t index = quant_datas[j];
          if (index > INT8_MAX || index < INT8_MIN) {
            MS_LOG(ERROR) << "KMeans param quant is error.";
            free(dequant_datas);
            return nullptr;
          }
          dequant_datas[j] = static_cast<DT>(param.clusters[index - INT8_MIN]);
        } else {
          dequant_datas[j] = static_cast<DT>((quant_datas[j] - zero_point) * scale);
        }
      }
    }
    return dequant_datas;
  }

  inline static bool IsChannelFirst(int index, const OpParameter *op_parameter) {
    MS_ASSERT(op_parameter != nullptr);
    if (op_parameter->type_ == schema::PrimitiveType_MatMul) {
      const auto *param = reinterpret_cast<const MatMulParameter *>(op_parameter);
      if (index == 0) {
        return !(param->a_transpose_);
      } else if (index == 1) {
        return param->b_transpose_;
      }
    }
    return true;
  }

  static int DequantWeight(lite::Tensor *input_tensor, bool channel_first, TypeId dst_data_type = kNumberTypeFloat32);

  template <typename T1, typename T2>
  static void UnPackData(int origin_bit, const T2 &packed_data, std::queue<bool> *unpack_bit_data, void *unpack_int,
                         size_t *count, bool is_last) {
    T2 uint_result = 0;
    T1 result;
    UnPackFromUintToOrigin<T2>(packed_data, unpack_bit_data);
    while (static_cast<int>(unpack_bit_data->size()) >= origin_bit) {
      for (int k = 0; k < origin_bit; k++) {
        bool bit_tmp = unpack_bit_data->front();
        uint_result = (static_cast<int>(bit_tmp) << k) + uint_result;
        unpack_bit_data->pop();
      }
      result = uint_result - static_cast<T2>(pow(2, origin_bit - 1));
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
      result = static_cast<T1>(uint_result - static_cast<T2>(pow(2, origin_bit - 1)));
      (static_cast<T1 *>(unpack_int))[*count] = result;
    }
  }

  template <typename T1, typename T2>
  static void UnPackUtil(const schema::Tensor *input_tensor, int origin_bit, void *unpack_int_data) {
    if (input_tensor == nullptr || input_tensor->data() == nullptr) {
      MS_LOG(ERROR) << "tensor data is null";
      return;
    }
    auto weight_data = input_tensor->data()->data();
    int pack_size =
      input_tensor->dataType() == kNumberTypeInt8 ? input_tensor->data()->size() : input_tensor->data()->size() / 2;
    std::queue<bool> unpack_bit_data;
    size_t count = 0;
    for (int i = 0; i < pack_size; ++i) {
      T2 pack_data = (static_cast<const T2 *>(static_cast<const void *>(weight_data)))[i];
      bool is_last = i == pack_size - 1;
      UnPackData<T1, T2>(origin_bit, pack_data, &unpack_bit_data, unpack_int_data, &count, is_last);
    }
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
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_DEQUANT_H_
