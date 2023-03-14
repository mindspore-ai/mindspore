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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_DECODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_DECODER_H_

#include <cstdint>
#include <vector>
#include "tools/converter/quantizer/fse_bit_stream.h"
#include "src/tensor.h"
#include "src/litert/lite_model.h"

namespace mindspore::lite::quant {
struct FSEBuffer {
  uint16_t frequency_count = 0;
  size_t table_log = 0;
  uint32_t chunk_count = 0;
  int32_t curr_chunk_index = 0;
  uint32_t *frequency = nullptr;
  void *centroids = nullptr;
  size_t centroid_size = 0;
  uint64_t *chunks = nullptr;
  size_t chunk_size = 0;
  uint64_t curr_chunk = 0;
  uint8_t curr_bit_count = 0;
  size_t chunk_ends_count = 0;
  uint64_t *chunk_ends = nullptr;
};
class FSEDecoder {
 public:
  FSEDecoder() = default;
  ~FSEDecoder() = default;

  static int DeCompress(const SchemaTensorWrapper &src_tensor, Tensor *dst_tensor,
                        schema::WeightQuantCompressType compress_type);

  static int FSECreateStatesForDecoding(const uint32_t *symbol_frequency, int symbol_frequency_count, size_t table_log,
                                        uint16_t *new_state_baseline, uint8_t *bit_count, uint16_t *symbol_table);

  static int DecodeBuffer(int8_t *buffer, size_t data_size, FSEBuffer *fse_buffer);

 private:
  template <typename C_TYPE, typename OUT_TYPE>
  static int FSEDecode(FSEBitStream *bs, OUT_TYPE *buff, int buff_count, uint32_t *frequency, int frequency_count,
                       const C_TYPE *centroids, size_t table_log) {
    MS_ASSERT(bs != nullptr);
    MS_ASSERT(buff != nullptr);
    MS_ASSERT(frequency != nullptr);
    MS_ASSERT(centroids != nullptr);
    size_t table_size = 1u << table_log;
    std::vector<uint16_t> states_table(table_size);
    std::vector<uint8_t> bit_count_table(table_size);
    std::vector<uint16_t> symbol_table(table_size);
    auto ret = FSECreateStatesForDecoding(frequency, frequency_count, table_log, states_table.data(),
                                          bit_count_table.data(), symbol_table.data());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "FSE create states for decoding failed.";
      return RET_ERROR;
    }

    auto state = bs->Pop(table_log);
    while ((bs->GetCurrChunkIndex() >= 0) || (bit_count_table[state] == 0) || (bs->GetCurrBitCount() > 0)) {
      if (buff_count == 0) {
        return RET_OK;
      }
      buff[--buff_count] = static_cast<OUT_TYPE>(centroids[symbol_table[state]]);
      state = states_table[state] + bs->Pop(bit_count_table[state]);
    }

    // Unexpected Condition!!!
    int remaining_buff_count = buff_count;
    if (remaining_buff_count < 0) {
      MS_LOG(ERROR) << "out buffer too small";
      return RET_ERROR;
    }
    if (remaining_buff_count > 0) {
      MS_LOG(ERROR) << "out buffer too large";
      return RET_ERROR;
    }
    return ret;
  }
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_DECODER_H_
