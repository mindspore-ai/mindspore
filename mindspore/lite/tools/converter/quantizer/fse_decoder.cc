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

#include <memory>
#include <vector>
#include "tools/converter/quantizer/fse_decoder.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"

namespace mindspore::lite::quant {
int FSEDecoder::FSECreateStatesForDecoding(const uint32_t *symbol_frequency, int symbol_frequency_count, int table_log,
                                           uint16_t *new_state, uint8_t *bit_count, uint16_t *symbol_table) {
  MS_ASSERT(symbol_frequency != nullptr);
  MS_ASSERT(new_state != nullptr);
  MS_ASSERT(bit_count != nullptr);
  MS_ASSERT(symbol_table != nullptr);
  const int table_size = 1 << table_log;
  int table_mask = table_size - 1;
  int step = ((table_size >> 1) + (table_size >> 3) + 3);
  int pos = 0;
  for (int sym = 0; sym < symbol_frequency_count; sym++) {
    for (uint32_t i = 0; i < symbol_frequency[sym]; i++) {
      symbol_table[pos] = sym;
      pos = (pos + step) & table_mask;
      while (pos > table_mask) {
        pos = (pos + step) & table_mask;
      }
    }
  }
  if (pos != 0) {
    return RET_ERROR;
  }
  // defensive copy to not mutate frequency:
  std::vector<uint32_t> frequency(symbol_frequency, symbol_frequency + symbol_frequency_count);

  for (int i = 0; i < table_size; i++) {
    uint16_t sym = symbol_table[i];
    uint32_t x = frequency[sym];
    frequency[sym] += 1;
#ifdef _MSC_VER
    int num = 0;
    uint32_t tmp = x;
    tmp |= 1;
    while (!(tmp & 0x80000000)) {
      num += 1;
      tmp <<= 1;
    }
    bit_count[i] = table_log - (num ^ 31);
#else
    bit_count[i] = table_log - (__builtin_clz(x) ^ 31);
#endif
    new_state[i] = (x << bit_count[i]) - table_size;
  }
  return RET_OK;
}

int FSEDecoder::FSEDecode(BitStream *bs, float *buff, int buff_count, uint32_t *frequency, int frequency_count,
                          const float *centroids, int table_log) {
  MS_ASSERT(bs != nullptr);
  MS_ASSERT(buff != nullptr);
  MS_ASSERT(frequency != nullptr);
  MS_ASSERT(centroids != nullptr);
  int table_size = 1 << table_log;
  std::vector<uint16_t> states_table(table_size);
  std::vector<uint8_t> bit_count_table(table_size);
  std::vector<uint16_t> symbol_table(table_size);
  auto ret = FSECreateStatesForDecoding(frequency, frequency_count, table_log, states_table.data(),
                                        bit_count_table.data(), symbol_table.data());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSE create states for decoding failed.";
    return RET_ERROR;
  }

  uint16_t state = bs->Pop(table_log);
  while ((bs->GetCurrChunkIndex() >= 0) || (bit_count_table[state] == 0) || (bs->GetCurrBitCount() > 0)) {
    if (buff_count == 0) {
      return RET_OK;
    }
    buff[--buff_count] = centroids[symbol_table[state]];
    state = states_table[state] + bs->Pop(bit_count_table[state]);
  }

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

int FSEDecoder::DeCompress(const schema::Tensor &src_tensor, Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
  if (dst_tensor->MutableData() == nullptr) {
    MS_LOG(ERROR) << "tensor data is nullptr.";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(src_tensor.data());
  auto total_size = src_tensor.data()->size();
  float *output = static_cast<float *>(dst_tensor->data());
  CHECK_NULL_RETURN(output);
  int out_sz = dst_tensor->ElementsNum();
  // deserialize from `data`:
  BitStream bs;

  size_t i = 0;
  auto data8 = const_cast<unsigned char *>(src_tensor.data()->data());

  int frequency_count = *(reinterpret_cast<uint16_t *>(&data8[i]));
  i += sizeof(uint16_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  int table_log = *(reinterpret_cast<uint16_t *>(&data8[i]));
  i += sizeof(uint16_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  bs.SetChunkCount(*(reinterpret_cast<uint32_t *>(&data8[i])));
  bs.SetCurrChunkIndex(bs.GetChunkCount() - 2);
  i += sizeof(uint32_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  auto *frequency = reinterpret_cast<uint32_t *>(&data8[i]);
  i += frequency_count * sizeof(uint32_t);
  // Used for 8-byte alignment
  i = ((i + 7) >> 3) << 3;
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  auto centroids = reinterpret_cast<void *>(&data8[i]);
  auto centroids_float = reinterpret_cast<float *>(centroids);
  i += frequency_count * sizeof(float);
  // Used for 8-byte alignment
  i = ((i + 7) >> 3) << 3;
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  bs.SetChunks(reinterpret_cast<uint64_t *>(&data8[i]));
  i += (bs.GetCurrChunkIndex() + 1) * sizeof(uint64_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  bs.SetCurrChunk(*(reinterpret_cast<uint64_t *>(&data8[i])));
  i += sizeof(uint64_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  bs.SetCurrBitCount(*(reinterpret_cast<uint8_t *>(&data8[i])));

  auto res = FSEDecode(&bs, output, out_sz, frequency, frequency_count, centroids_float, table_log);
  if (res != RET_OK) {
    MS_LOG(ERROR) << "FSE Decode failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
