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
#include "nnacl/op_base.h"

namespace mindspore::lite::quant {
namespace {
constexpr size_t kTableExtend = 3;
constexpr size_t kAlignOffset = 7;
}  // namespace
int FSEDecoder::FSECreateStatesForDecoding(const uint32_t *symbol_frequency, int symbol_frequency_count,
                                           size_t table_log, uint16_t *new_state_baseline, uint8_t *bit_count,
                                           uint16_t *symbol_table) {
  CHECK_NULL_RETURN(symbol_frequency);
  CHECK_NULL_RETURN(new_state_baseline);
  CHECK_NULL_RETURN(bit_count);
  CHECK_NULL_RETURN(symbol_table);
  const size_t table_size = 1u << table_log;
  const size_t table_mask = table_size - 1;
  size_t step = ((table_size >> 1) + (table_size >> kTableExtend) + kTableExtend);
  size_t pos = 0;
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
    MS_LOG(ERROR) << "pos must equal 0.";
    return RET_ERROR;
  }
  // defensive copy to not mutate frequency:
  std::vector<uint32_t> frequency(symbol_frequency, symbol_frequency + symbol_frequency_count);

  for (size_t i = 0; i < table_size; i++) {
    uint16_t sym = symbol_table[i];
    uint32_t x = frequency[sym];
    frequency[sym] += 1;
    MS_CHECK_GE(table_log, FSEBitStream::CountBits(x), RET_ERROR);
    bit_count[i] = static_cast<uint8_t>(table_log - FSEBitStream::CountBits(x));
    new_state_baselne[i] = (x << bit_count[i]) - table_size;
  }
  return RET_OK;
}

int FSEDecoder::DeCompress(const SchemaTensorWrapper &src_tensor, Tensor *dst_tensor,
                           schema::WeightQuantCompressType compress_type) {
  CHECK_NULL_RETURN(src_tensor.handler());
  CHECK_NULL_RETURN(src_tensor.data());
  CHECK_NULL_RETURN(dst_tensor);
  if (dst_tensor->MutableData() == nullptr) {
    MS_LOG(ERROR) << "tensor data is nullptr.";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(src_tensor.data());
  auto total_size = src_tensor.length();
  int out_sz = dst_tensor->ElementsNum();
  MS_CHECK_GT(out_sz, 0, RET_ERROR);
  // deserialize from `data`:
  FSEBitStream bs;

  size_t i = 0;
  auto data8 = reinterpret_cast<int8_t *>(const_cast<void *>(src_tensor.data()));
  CHECK_NULL_RETURN(data8);
  // 16bit for frequency_count
  uint16_t frequency_count = *(reinterpret_cast<uint16_t *>(&data8[i]));
  i += sizeof(uint16_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  // 16bit for table_log
  size_t table_log = *(reinterpret_cast<uint16_t *>(&data8[i]));
  i += sizeof(uint16_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  // 32bit for ChunkCount
  bs.SetChunkCount(*(reinterpret_cast<uint32_t *>(&data8[i])));
  const int offset = 2;
  // 32bit for CurrChunkIndex
  bs.SetCurrChunkIndex(bs.GetChunkCount() - offset);
  i += sizeof(uint32_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  // 32bit * frequency_count for frequency
  auto *frequency = reinterpret_cast<uint32_t *>(&data8[i]);
  i += frequency_count * sizeof(uint32_t);
  // Used for 8-byte(64bit) alignment
  i = ((i + kAlignOffset) >> kTableExtend) << kTableExtend;
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  // 32bit * frequency_count for centroids
  auto centroids = reinterpret_cast<void *>(&data8[i]);
  i += frequency_count * sizeof(float);
  // Used for 8-byte(64bit) alignment
  i = ((i + kAlignOffset) >> kTableExtend) << kTableExtend;
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  // 64bit * bs.GetCurrChunkIndex() + 1 for Chunks.
  bs.SetChunks(reinterpret_cast<uint64_t *>(&data8[i]));
  i += (bs.GetCurrChunkIndex() + 1) * sizeof(uint64_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  // 64bit for CurrChunk
  bs.SetCurrChunk(*(reinterpret_cast<uint64_t *>(&data8[i])));
  i += sizeof(uint64_t);
  if (i > total_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << total_size;
    return RET_ERROR;
  }
  // 8bit for CurrBitCount
  bs.SetCurrBitCount(*(reinterpret_cast<uint8_t *>(&data8[i])));
  int ret;
  if (compress_type == schema::WeightQuantCompressType_FSE) {
    ret = FSEDecode<float, float>(&bs, static_cast<float *>(dst_tensor->data()), out_sz, frequency, frequency_count,
                                  static_cast<float *>(centroids), table_log);
  } else {  // WeightQuantCompressType_FSE_INT
    if (src_tensor.handler()->dataType() == kNumberTypeInt8) {
      ret = FSEDecode<int, int8_t>(&bs, static_cast<int8_t *>(dst_tensor->data()), out_sz, frequency, frequency_count,
                                   static_cast<int *>(centroids), table_log);
    } else {  // kNumberTypeInt16
      ret = FSEDecode<int, int16_t>(&bs, static_cast<int16_t *>(dst_tensor->data()), out_sz, frequency, frequency_count,
                                    static_cast<int *>(centroids), table_log);
    }
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSE Decode failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
