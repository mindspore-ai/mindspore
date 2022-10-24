/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_TENSOR_COMPRESSOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_TENSOR_COMPRESSOR_H_

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <limits>
#include <functional>
#include <algorithm>
#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/quantizer/bitpacking.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quantize_util.h"

using mindspore::ParameterPtr;
namespace mindspore::lite::quant {
class TensorCompressor {
 public:
  template <typename T>
  int DoSparseCompress(const ParameterPtr &weight, size_t bit_num,
                       const std::vector<schema::QuantParamT> &quant_params) {
    auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
    CHECK_NULL_RETURN(tensor_info);
    if (tensor_info->compression_type() != mindspore::kNoCompression) {
      MS_LOG(INFO) << weight->fullname_with_scope() << " is shared weight.";
      return RET_OK;
    }
    auto max_size = tensor_info->Size();
    auto quant_data_array = static_cast<T *>(tensor_info->data().data());

    std::vector<T> quant_data(quant_data_array, quant_data_array + max_size / sizeof(T));
    auto elem_cnt = quant_data.size();
    auto dims = tensor_info->shape_c();
    size_t elem_cnt_by_dims = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
    if (elem_cnt != elem_cnt_by_dims) {
      MS_LOG(ERROR) << weight->fullname_with_scope() << " elem_cnt: " << elem_cnt
                    << " not equal elem_cnt_by_dims: " << elem_cnt_by_dims;
      return RET_ERROR;
    }

    std::set<T> quant_data_set;
    for (auto quant_value : quant_data) {
      quant_data_set.insert(quant_value);
    }
    std::map<T, size_t> unique_value_index_map;
    auto index = 0;
    for (auto iter = quant_data_set.cbegin(); iter != quant_data_set.cend(); ++iter) {
      unique_value_index_map[*iter] = index++;
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
      return RET_NO_CHANGE;
    } else if (min_byte_need == pack_repetition_size_in_byte) {
      MS_LOG(DEBUG) << "from " << origin_size_in_byte << " to " << pack_repetition_size_in_byte;
      return IndexingCompress<T>(weight, quant_data_set, unique_value_index_map, unique_value_bit, unique_value_cnt,
                                 pack_repetition_size_in_byte, bit_num);
    } else if (min_byte_need == pack_sparsity_size_in_byte) {
      MS_LOG(DEBUG) << "from " << origin_size_in_byte << " to " << pack_sparsity_size_in_byte;
      return SparsityCompress<T>(weight, quant_params, quant_data_set, unique_value_index_map, unique_value_bit,
                                 unique_value_cnt, pack_sparsity_size_in_byte, nz_cnt, coor_best_bit, bit_num);
    } else {
      MS_LOG(DEBUG) << "unexpected: " << min_byte_need << " not in {" << origin_size_in_byte << " "
                    << pack_repetition_size_in_byte << " " << pack_sparsity_size_in_byte << "}";
    }
    return RET_NO_CHANGE;
  }
  int DoBitPack(const size_t &bit_num, schema::TensorT *tensor_input);

  int DoBitPack(const ParameterPtr &weight, size_t bit_num);

 private:
  template <typename T>
  int IndexingCompress(const ParameterPtr &weight, const std::set<T> &quant_data_set,
                       const std::map<T, size_t> &unique_value_index_map, size_t unique_value_bit,
                       size_t unique_value_cnt, size_t pack_repetition_size_in_byte, size_t bit_num) {
    std::vector<bool> bits(pack_repetition_size_in_byte * k8Bit);
    size_t index = 0;
    // write unique_value_cnt: bit_num bit for unsigned
    for (size_t i = 0; i < bit_num; i++) {
      bits[index++] = (unique_value_cnt >> (bit_num - i - 1)) & (0x1);
    }
    // write the unique value set: each value has bit_num bit signed
    for (auto iter = quant_data_set.cbegin(); iter != quant_data_set.cend(); ++iter) {
      for (size_t i = 0; i < bit_num; i++) {
        bits[index++] = ((*iter + (1 << (bit_num - 1))) >> (bit_num - i - 1)) & (0x1);
      }
    }

    auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
    CHECK_NULL_RETURN(tensor_info);
    auto max_size = tensor_info->Size();
    auto quant_data = static_cast<T *>(tensor_info->data().data());
    // write the index: each index has unique_value_bit unsigned
    for (size_t i = 0; i < max_size; i++) {
      auto quant_value = quant_data[i];
      for (size_t j = 0; j < unique_value_bit; j++) {
        bits[index++] = (unique_value_index_map.at(quant_value) >> (unique_value_bit - j - 1)) & (0x1);
      }
    }
    if (index > pack_repetition_size_in_byte * k8Bit) {
      MS_LOG(ERROR) << "unexpected index: " << index << " should not be greater than "
                    << pack_repetition_size_in_byte * k8Bit;
      return RET_ERROR;
    }

    auto ret = SetNewCompressionTensor(weight, bits, bit_num, tensor_info, mindspore::kIndexing);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Add New tensor failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }

  template <typename T>
  int SparsityCompress(const ParameterPtr &weight, const std::vector<schema::QuantParamT> &quant_params,
                       const std::set<T> &quant_data_set, const std::map<T, size_t> &unique_value_index_map,
                       size_t unique_value_bit, size_t unique_value_cnt, size_t pack_sparsity_size_in_byte,
                       size_t nz_cnt, size_t coor_best_bit, size_t bit_num) {
    auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
    CHECK_NULL_RETURN(tensor_info);
    auto quant_data = static_cast<T *>(tensor_info->data().data());
    int elem_cnt = tensor_info->DataSize();
    auto channel_cnt = quant_params.size();
    if (channel_cnt == 0) {
      MS_LOG(ERROR) << "quant_params is empty.";
      return RET_ERROR;
    }
    auto elem_perchannel = elem_cnt / channel_cnt;

    std::vector<bool> bits(pack_sparsity_size_in_byte * k8Bit);
    size_t index = 0;
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
    int coors_index = 0;
    int prev_index = -1;
    for (int di = 0; di < elem_cnt; di++) {
      auto cur_channel = di / elem_perchannel;
      auto zp = quant_params[cur_channel].zeroPoint;
      auto nz_value = quant_data[di];
      if (nz_value != zp || static_cast<size_t>(di - prev_index) >= (1u << coor_best_bit)) {
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
    if (index > pack_sparsity_size_in_byte * k8Bit) {
      MS_LOG(ERROR) << "unexpected index: " << index << " should not be greater than "
                    << pack_sparsity_size_in_byte * k8Bit;
      return RET_ERROR;
    }

    auto ret = SetNewCompressionTensor(weight, bits, bit_num, tensor_info, mindspore::kSparse);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Add New tensor failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }

  template <typename T>
  size_t CalCoorBestBit(const std::vector<T> &quant_data, size_t elem_cnt,
                        const std::vector<schema::QuantParamT> &quant_params, int unique_value_bit,
                        size_t *coor_best_bit) {
    MS_ASSERT(!quant_params.empty());
    size_t best_nn_cnt = 0;
    size_t min_len_in_bit = std::numeric_limits<size_t>::max();
    for (size_t bit = k2Bit; bit <= k10Bit; bit++) {
      // search
      int nn_cnt = 0;
      int prev_index = -1;
      auto channel_cnt = quant_params.size();
      MS_ASSERT(channel_cnt > 0);
      auto elem_perchannel = elem_cnt / channel_cnt;
      for (size_t i = 0; i < elem_cnt; i++) {
        auto cur_channel = i / elem_perchannel;
        auto zp = quant_params[cur_channel].zeroPoint;
        if (quant_data[i] != zp || (static_cast<int>(i) - prev_index) >= ((1 << bit))) {
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

  void WriteBufferWithAlignByte(const std::vector<bool> &bool_vec, int8_t *data);

  int SetNewCompressionTensor(const ParameterPtr &weight, const std::vector<bool> &bits, size_t bit_num,
                              const tensor::TensorPtr &tensor_info, TensorCompressionType compression_type);
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_TENSOR_COMPRESSOR_H_
