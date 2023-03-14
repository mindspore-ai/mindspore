/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_ENCODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_ENCODER_H_

#include <vector>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "tools/converter/quantizer/fse_bit_stream.h"
#include "tools/converter/quantizer/fse_chunk_end.h"
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "src/common/quant_utils.h"

namespace mindspore::lite::quant {
constexpr size_t MAX_SYMS = 65534;
constexpr size_t MAX_TABLE_LOG = 16;

typedef struct FSEQuant {
  uint16_t *symbol_table{nullptr};    // the place to store the quantized tensor
  size_t symbol_table_count{0};       // the number of symbols that exist
  float centroids_float[MAX_SYMS];    // the mean of all the numbers that got quantized into it
  int32_t centroids_int[MAX_SYMS];    // the mean of all the numbers that got quantized into it
  uint32_t frequency[MAX_SYMS];       // holds the number of times each symbol appears in `*symbol_table`
  size_t size{0};                     // the number of entries in `symbol_table`
  size_t num_chunk_ends{0};           // number of chunk ends elements
  ChunkEndData *chunk_ends{nullptr};  // chunk ends elements
} FSEQuant;

class FSEEncoder {
 public:
  FSEEncoder() = default;
  ~FSEEncoder() = default;

  int Compress(const ParameterPtr &weight, const std::vector<schema::QuantParamT> &q_param,
               mindspore::TensorCompressionType compress_type, int max_chanks = 1);

 private:
  int FSECreateStatesForEncoding(const uint32_t *frequency, size_t frequency_count, size_t table_log,
                                 uint32_t *delta_bit_count, int16_t *delta_state, uint16_t *coding_table,
                                 uint16_t *symbol_table);

  uint8_t NumOfBits(uint16_t sym, uint16_t state, const uint32_t *delta_bit_count);

  uint16_t FSEEncodeSymbolGetNewState(FSEBitStream *bs, uint16_t sym, uint16_t state, uint8_t num_of_bits,
                                      const int16_t *delta_state, const uint16_t *coding_table);

  // Encoding is therefore just a repeat of this process :
  // - get Symbol to encode
  // - look at current state value
  // - determine nbBits, flush them
  // - determine sub-Range Id
  // - look for Symbol position of same Id : you get your next state
  int FSEEncode(FSEBitStream *bs, const uint16_t *data, size_t data_count, const uint32_t *frequency,
                const size_t num_chunk_ends, ChunkEndData *chunk_ends, size_t frequency_count, size_t table_log);

  int NormalizeFrequency(FSEQuant *q, size_t *table_log);

  int SerializingToTensor(const ParameterPtr &weight, const FSEBitStream *bs, const FSEQuant &fse_quant,
                          size_t table_log, TensorCompressionType compress_type);

  int SerializingToBuffer(const FSEBitStream *bs, const FSEQuant &fse_quant, size_t table_log, size_t max_size,
                          uint8_t *out8, size_t *out_size, TensorCompressionType compress_type);

  template <typename T>
  int SqueezeQuant(const ParameterPtr &weight, const std::vector<schema::QuantParamT> &q_param, FSEQuant *quants,
                   mindspore::TensorCompressionType compress_type) {
    CHECK_NULL_RETURN(weight);
    CHECK_NULL_RETURN(quants);
    auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
    CHECK_NULL_RETURN(tensor_info);

    auto data_c = static_cast<T *>(tensor_info->data_c());
    auto data_size = tensor_info->DataSize();

    auto min_max = GetMinMaxValue(static_cast<T *>(tensor_info->data_c()), data_size);
    int qmin = min_max.first;
    int qmax = min_max.second;
    int uncompressed_frequency_count = qmax - qmin + 1;

    std::vector<int> uncompressed_frequency(uncompressed_frequency_count);
    for (int i = 0; i < uncompressed_frequency_count; i++) {
      uncompressed_frequency[i] = 0;
    }
    for (size_t i = 0; i < data_size; i++) {
      auto data = static_cast<T>(data_c[i]);
      int q = data - qmin;
      uncompressed_frequency[q] += 1;
    }

    double shannon_entropy = 0.0;
    for (int i = 0; i < uncompressed_frequency_count; ++i) {
      if (uncompressed_frequency[i] != 0) {
        auto p = 1.0 * uncompressed_frequency[i] / data_size;
        shannon_entropy -= p * log(p);
      }
    }
    MS_LOG(INFO) << weight->fullname_with_scope() << " shannon_entropy is " << shannon_entropy;

    std::vector<uint16_t> uncompressed_freqs_to_compressed_sym(uncompressed_frequency_count);
    size_t sym = 0;
    for (int i = 0; i < uncompressed_frequency_count; i++) {
      if (uncompressed_frequency[i] != 0) {
        if (sym >= MAX_SYMS) {
          return RET_ERROR;  // too many symbols!
        }
        uncompressed_freqs_to_compressed_sym[i] = sym;
        quants->frequency[sym] = uncompressed_frequency[i];
        // real = varCorr * (q - zp) * scale + meanCorr
        if (compress_type == mindspore::kFSE || compress_type == mindspore::kFSEInfer) {
          if (q_param.empty()) {
            MS_LOG(ERROR) << "q_param is empty.";
            return RET_ERROR;
          }
          quants->centroids_float[sym] = q_param.front().varCorr *
                                           static_cast<float>(i + qmin - q_param.front().zeroPoint) *
                                           (q_param.front().scale) +
                                         q_param.front().meanCorr;
        } else {
          quants->centroids_int[sym] = i + qmin;
        }
        sym++;
      }
    }
    MS_LOG(INFO) << "uncompressed frequency count:" << uncompressed_frequency_count << " sym:" << sym;
    quants->size = sym;
    quants->symbol_table_count = data_size;
    quants->symbol_table = static_cast<uint16_t *>(malloc(quants->symbol_table_count * sizeof(uint16_t)));
    if (quants->symbol_table == nullptr) {
      MS_LOG(ERROR) << "malloc memory failed.";
      return RET_ERROR;
    }
    for (size_t i = 0; i < quants->symbol_table_count; i++) {
      auto data = static_cast<T>(data_c[i]);
      int q = data - qmin;
      sym = uncompressed_freqs_to_compressed_sym[q];
      quants->symbol_table[i] = sym;
    }
    return RET_OK;
  }
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_ENCODER_H_
