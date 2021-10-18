/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_ENCODER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_ENCODER_H

#include <vector>
#include "tools/converter/quantizer/fse_bit_stream.h"
#include "tools/converter/quantizer/mixed_bit_weight_quantizer.h"
namespace mindspore::lite::quant {
constexpr int MAX_SYMS = 65534;
constexpr int MAX_TABLE_LOG = 16;
typedef struct {
  uint16_t *symbol_table;        // the place to store the quantized tensor
  int symbol_table_count;        // the number of symbols that exist
  float centroids[MAX_SYMS];     // the mean of all the numbers that got quantized into it
  uint32_t frequency[MAX_SYMS];  // holds the number of times each symbol appears in `*symbol_table`
  int size;                      // the number of entries in `symbol_table`
} FSEQuant;

class FSEEncoder {
 public:
  FSEEncoder() = default;
  ~FSEEncoder() = default;
  int Compress(schema::TensorT *tensor_input);

 private:
  int FSECreateStatesForEncoding(uint32_t *frequency, int frequency_count, int table_log, uint32_t *delta_bit_count,
                                 int16_t *delta_state, uint16_t *coding_table, uint16_t *symbol_table);

  uint16_t FSEEncodeSymbolGetNewState(BitStream *bs, uint16_t sym, uint16_t state, const uint32_t *delta_bit_count,
                                      const int16_t *delta_state, uint16_t *coding_table);

  int FSEEncode(BitStream *bs, const uint16_t *data, int data_count, uint32_t *frequency, int frequency_count,
                int table_log);

  int NormalizeFrequency(FSEQuant *q, int *table_log);

  int SerializingToOut(schema::TensorT *tensor_input, BitStream *bs, const FSEQuant &fse_quant, int table_log);

  int SerializingToTensor(schema::TensorT *tensor_input, BitStream *bs, const FSEQuant &fse_quant, int table_log,
                          uint8_t *out8, size_t max_size, size_t *offset);
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_ENCODER_H
