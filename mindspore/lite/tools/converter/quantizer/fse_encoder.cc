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

#include "tools/converter/quantizer/fse_encoder.h"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <memory>
#include "mindspore/core/ir/dtype/type_id.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
#include "include/errorcode.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/common/statistic_utils.h"
#include "ir/tensor.h"

namespace mindspore::lite::quant {
namespace {
constexpr size_t kFseTableExtendSize = 3u;
constexpr size_t kFreqTableExtendSize = 2u;
constexpr size_t kAlignSize = 8u;
constexpr size_t kAlignHalfSize = 4u;
constexpr float kUpRoundOffSet = 0.5f;
constexpr size_t kMaxModelBufferSize = 1024u * 1024 * 1024 * 2;  // 2G
constexpr size_t PARALLEL_MIN_SIZE = 10000;
}  // namespace

int FSEEncoder::FSECreateStatesForEncoding(const uint32_t *frequency, size_t frequency_count, size_t table_log,
                                           uint32_t *delta_bit_count, int16_t *delta_state, uint16_t *coding_table,
                                           uint16_t *symbol_table) {
  CHECK_NULL_RETURN(frequency);
  CHECK_NULL_RETURN(delta_bit_count);
  CHECK_NULL_RETURN(delta_state);
  CHECK_NULL_RETURN(symbol_table);
  CHECK_NULL_RETURN(coding_table);
  const size_t table_size = 1u << static_cast<size_t>(table_log);
  size_t table_mask = table_size - 1;
  size_t step = ((table_size >> 1u) + (table_size >> kFseTableExtendSize) + kFseTableExtendSize);
  size_t pos = 0;
  // Separate the same symbols, coding will be better if the same characters are distributed evenly across the table.
  for (size_t sym = 0; sym < frequency_count; sym++) {
    for (uint32_t i = 0; i < frequency[sym]; i++) {
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

  std::vector<uint32_t> cfreqs(frequency_count + kFreqTableExtendSize);
  cfreqs[0] = 0;
  for (size_t i = 1; i < frequency_count + 1; i++) {
    cfreqs[i] = cfreqs[i - 1] + frequency[i - 1];
  }
  cfreqs[frequency_count + 1] = cfreqs[frequency_count] + 1;
  for (size_t i = 0; i < table_size; i++) {
    uint16_t sym = symbol_table[i];
    coding_table[cfreqs[sym]] = table_size + i;
    cfreqs[sym] += 1;
  }

  int total = 0;
  for (size_t sym = 0; sym < frequency_count; sym++) {
    if (frequency[sym] >= kFreqTableExtendSize) {
      auto bit = FSEBitStream::CountBits(frequency[sym] - 1);
      MS_CHECK_GE(table_log, bit, RET_ERROR);
      size_t max_bits_out = table_log - bit;
      int min_state_plus = frequency[sym] << max_bits_out;
      delta_bit_count[sym] = (max_bits_out << k16Bit) - min_state_plus;
      delta_state[sym] = total - static_cast<int>(frequency[sym]);
      total += static_cast<int>(frequency[sym]);
    } else {
      // we assume minimum `frequency` is 1
      delta_bit_count[sym] = (table_log << k16Bit) - (1 << table_log);
      delta_state[sym] = total - 1;
      total++;
    }
  }
  return RET_OK;
}

int FSEEncoder::Compress(const ParameterPtr &weight, const std::vector<schema::QuantParamT> &q_param,
                         mindspore::TensorCompressionType compress_type, int max_segments) {
  auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
  CHECK_NULL_RETURN(tensor_info);
  FSEQuant fse_quant;
  int ret = RET_ERROR;
  if (tensor_info->data_type() == kNumberTypeInt16) {
    ret = SqueezeQuant<int16_t>(weight, q_param, &fse_quant, compress_type);
  } else if (tensor_info->data_type() == kNumberTypeInt8) {
    ret = SqueezeQuant<int8_t>(weight, q_param, &fse_quant, compress_type);
  } else {
    MS_LOG(ERROR) << " type_id:" << tensor_info->data_type() << " don't support.";
    return ret;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Squeeze quant data failed.";
    return ret;
  }
  size_t table_log = 0;
  ret = NormalizeFrequency(&fse_quant, &table_log);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Normalize frequency failed.";
    free(fse_quant.symbol_table);
    return ret;
  }
  FSEBitStream bs;
  uint64_t bit_capacity = k16Bit * fse_quant.symbol_table_count;
  ret = bs.Create(bit_capacity);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSEBitStream Create failed.";
    free(fse_quant.symbol_table);
    return ret;
  }
  size_t num_chunk_ends = (fse_quant.symbol_table_count > PARALLEL_MIN_SIZE) ? max_segments : 1;
  fse_quant.chunk_ends = static_cast<ChunkEndData *>(malloc(num_chunk_ends * sizeof(ChunkEndData)));
  if (fse_quant.chunk_ends == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return RET_ERROR;
  }
  fse_quant.num_chunk_ends = num_chunk_ends;
  ret = FSEEncode(&bs, fse_quant.symbol_table, fse_quant.symbol_table_count, fse_quant.frequency, num_chunk_ends,
                  fse_quant.chunk_ends, fse_quant.size, table_log);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSE Encode failed.";
    free(fse_quant.symbol_table);
    return ret;
  }
  bs.Flush();
  // Serializing to out:
  ret = SerializingToTensor(weight, &bs, fse_quant, table_log, compress_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Serializing To Tensor failed.";
    free(fse_quant.symbol_table);
    return ret;
  }
  bs.Free();
  free(fse_quant.symbol_table);
  return RET_OK;
}

uint8_t FSEEncoder::NumOfBits(uint16_t sym, uint16_t state, const uint32_t *delta_bit_count) {
  // It is to determine the number of bits to flush.
  // This is basically one of 2 values, n or n+1, depending on state crossing a threshold.
  MS_ASSERT(delta_bit_count != nullptr);
  uint8_t bits_out = (state + delta_bit_count[sym]) >> k16Bit;
  return bits_out;
}

uint16_t FSEEncoder::FSEEncodeSymbolGetNewState(FSEBitStream *bs, uint16_t sym, uint16_t state, uint8_t bits_out,
                                                const int16_t *delta_state, const uint16_t *coding_table) {
  MS_ASSERT(bs != nullptr);
  MS_ASSERT(delta_state != nullptr);
  MS_ASSERT(coding_table != nullptr);
  bs->Push(state, bits_out);
  // subrangeID = state >> nbBitsOut
  return coding_table[(state >> bits_out) + delta_state[sym]];
}

int GetMaxIndex(const uint32_t *arr, size_t arr_count) {
  MS_ASSERT(arr != nullptr);
  float max = -INFINITY;
  int index = -1;
  for (size_t i = 0; i < arr_count; i++) {
    if (arr[i] > max) {
      max = arr[i];
      index = i;
    }
  }
  return index;
}

int FSEEncoder::NormalizeFrequency(FSEQuant *q, size_t *table_log) {
  CHECK_NULL_RETURN(q);
  CHECK_NULL_RETURN(table_log);
  // The higher the number, the more accurate we'll be to the shannon entropy,
  // but also the larger the table, so `+3` is a good compromise. +1 inorder to round up
  *table_log = std::min(MAX_TABLE_LOG, (FSEBitStream::CountBits((uint32_t)q->size) + 1 + kFseTableExtendSize));
  const int new_table_size = 1 << (*table_log);
  int curr_table_size = 0;
  for (size_t i = 0; i < q->size; i++) {
    curr_table_size += q->frequency[i];
  }

  if (curr_table_size == 0) {
    MS_LOG(ERROR) << "curr_table_size is 0";
    return RET_ERROR;
  }
  // normalize
  int updated_table_size = 0;
  float rat = (static_cast<float>(new_table_size)) / curr_table_size;
  for (size_t i = 0; i < q->size; i++) {
    q->frequency[i] = std::max(1, static_cast<int>(floorf(kUpRoundOffSet + rat * q->frequency[i])));
    updated_table_size += q->frequency[i];
  }

  // If the sum of the symbol frequencies is not equal to the power of two (almost always),
  // then the frequencies need to be normalized-they must be proportionally reduced (or increased) so that the power of
  // two is obtained in total.
  // shrink
  while (updated_table_size > new_table_size) {
    int max_ix = GetMaxIndex(q->frequency, q->size);
    if (max_ix < 0 || static_cast<size_t>(max_ix) > MAX_SYMS) {
      MS_LOG(ERROR) << "max_ix is invalid.";
      return RET_ERROR;
    }
    q->frequency[max_ix]--;
    updated_table_size--;
  }

  // grow
  if (updated_table_size < new_table_size) {
    int max_ix = GetMaxIndex(q->frequency, q->size);
    if (max_ix < 0 || static_cast<size_t>(max_ix) >= MAX_SYMS) {
      MS_LOG(ERROR) << "max_ix is invalid.";
      return RET_ERROR;
    }
    q->frequency[max_ix] += new_table_size - updated_table_size;
  }
  return RET_OK;
}

int FSEEncoder::FSEEncode(FSEBitStream *bs, const uint16_t *data, size_t data_count, const uint32_t *frequency,
                          const size_t num_chunk_ends, ChunkEndData *chunk_ends, size_t frequency_count,
                          size_t table_log) {
  CHECK_NULL_RETURN(bs);
  CHECK_NULL_RETURN(data);
  CHECK_NULL_RETURN(frequency);
  int table_size = 1 << table_log;
  // symbolTT.deltaNbBits stores a value which, when added with state,
  // makes the result of >> 16 produces either n or n+1, as required.
  std::vector<uint32_t> delta_number_bits(frequency_count);
  // symbolTT.deltaFindState provides the offset to find the correct segment into the table.
  std::vector<int16_t> delta_find_state(frequency_count);
  // nextStateTable with symbol
  std::vector<uint16_t> coding_table(table_size);
  // position with symbol
  std::vector<uint16_t> symtable(table_size);
  int ret = FSECreateStatesForEncoding(frequency, frequency_count, table_log, delta_number_bits.data(),
                                       delta_find_state.data(), coding_table.data(), symtable.data());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create states table for encoding failed.";
    return ret;
  }
  uint16_t state = table_size;
  // The results of the 1st symbol encoding is not flushed to the bitstream,
  // It is just to get a valid 1 st state.
  uint8_t num_of_bits = NumOfBits(data[0], state, delta_number_bits.data());
  state = FSEEncodeSymbolGetNewState(bs, data[0], state, num_of_bits, delta_find_state.data(), coding_table.data());
  bs->Empty();
  size_t idx = 0;
  if (num_chunk_ends <= 0) {
    MS_LOG(ERROR) << "num_chunk_ends is invalid.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < data_count; i++) {
    num_of_bits = NumOfBits(data[i], state, delta_number_bits.data());
    state = FSEEncodeSymbolGetNewState(bs, data[i], state, num_of_bits, delta_find_state.data(), coding_table.data());
    if ((i == data_count * (idx + 1) / num_chunk_ends - 1) && (chunk_ends != nullptr) && (idx < num_chunk_ends)) {
      chunk_ends[idx].bs_position = bs->GetCurrChunkIndex();
      chunk_ends[idx].state = state - table_size;
      chunk_ends[idx].bit_count = bs->GetCurrBitCount();
      idx++;
    }
  }
  bs->Push(state - table_size, table_log);
  return ret;
}

int FSEEncoder::SerializingToBuffer(const FSEBitStream *bs, const FSEQuant &fse_quant, size_t table_log,
                                    size_t max_size, uint8_t *out8, size_t *out_size,
                                    TensorCompressionType compress_type) {
  MSLITE_CHECK_PTR(bs);
  MSLITE_CHECK_PTR(out_size);
  MSLITE_CHECK_PTR(out8);
  size_t offset = 0;
  *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)fse_quant.size;
  offset += sizeof(uint16_t);
  CHECK_LARGE_RETURN(offset + sizeof(uint16_t), max_size);
  *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)table_log;
  offset += sizeof(uint16_t);
  int chunksc = bs->GetCurrChunkIndex() + sizeof(uint16_t);
  CHECK_LARGE_RETURN(offset + sizeof(uint32_t), max_size);
  *(reinterpret_cast<uint32_t *>(&out8[offset])) = (uint32_t)chunksc;
  offset += sizeof(uint32_t);
  for (size_t j = 0; j < fse_quant.size; j++) {
    CHECK_LARGE_RETURN(offset + sizeof(uint32_t), max_size);
    *(reinterpret_cast<uint32_t *>(&out8[offset])) = (uint32_t)fse_quant.frequency[j];
    offset += sizeof(uint32_t);
  }
  while (offset % kAlignSize != 0) {
    CHECK_LARGE_RETURN(offset + sizeof(uint16_t), max_size);
    *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)0;
    offset += sizeof(uint16_t);
  }
  for (size_t j = 0; j < fse_quant.size; j++) {
    CHECK_LARGE_RETURN(offset + sizeof(float), max_size);
    if (compress_type == mindspore::kFSE || compress_type == mindspore::kFSEInfer) {
      *(reinterpret_cast<float *>(&out8[offset])) = fse_quant.centroids_float[j];
    } else {
      *(reinterpret_cast<int32_t *>(&out8[offset])) = fse_quant.centroids_int[j];
    }
    offset += sizeof(float);
  }
  while (offset % kAlignSize != 0) {
    CHECK_LARGE_RETURN(offset + sizeof(uint16_t), max_size);
    *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)0;
    offset += sizeof(uint16_t);
  }
  for (int j = 0; j < bs->GetCurrChunkIndex() + 1; j++) {
    CHECK_LARGE_RETURN(offset + sizeof(uint64_t), max_size);
    *(reinterpret_cast<uint64_t *>(&out8[offset])) = bs->GetChunks()[j];
    offset += sizeof(uint64_t);
  }
  CHECK_LARGE_RETURN(offset + sizeof(uint64_t), max_size);
  *(reinterpret_cast<uint64_t *>(&out8[offset])) = bs->GetCurrChunk();
  offset += sizeof(uint64_t);
  CHECK_LARGE_RETURN(offset + sizeof(uint8_t), max_size);
  *(reinterpret_cast<uint8_t *>(&out8[offset])) = bs->GetCurrBitCount();
  offset += sizeof(uint8_t);
  if (offset > max_size) {
    MS_LOG(ERROR) << " too many symbol.";
    return RET_ERROR;
  }
  if ((compress_type == mindspore::kFSEInfer) && (fse_quant.num_chunk_ends > 0)) {
    while (offset % kAlignHalfSize != 0) {
      CHECK_LARGE_RETURN(offset + sizeof(uint8_t), max_size);
      *(reinterpret_cast<uint8_t *>(&out8[offset])) = (uint8_t)0;
      offset += sizeof(uint8_t);
    }
    CHECK_LARGE_RETURN(offset + sizeof(uint32_t), max_size);
    *(reinterpret_cast<uint32_t *>(&out8[offset])) = (uint32_t)fse_quant.num_chunk_ends;
    offset += sizeof(uint32_t);

    for (size_t j = 0; j < fse_quant.num_chunk_ends; j++) {
      CHECK_LARGE_RETURN(offset + sizeof(uint64_t), max_size);
      *(reinterpret_cast<uint64_t *>(&out8[offset])) = fse_quant.chunk_ends[j].ToUint64();
      offset += sizeof(uint64_t);
    }
  }
  *out_size = offset;
  return RET_OK;
}

int FSEEncoder::SerializingToTensor(const ParameterPtr &weight, const FSEBitStream *bs, const FSEQuant &fse_quant,
                                    size_t table_log, TensorCompressionType compress_type) {
  MSLITE_CHECK_PTR(weight);
  MSLITE_CHECK_PTR(bs);
  auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
  CHECK_NULL_RETURN(tensor_info);

  auto max_size = tensor_info->Size();
  if (max_size == 0 || max_size > kMaxModelBufferSize) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " malloc size:" << max_size << " is invalid.";
    return RET_ERROR;
  }
  auto *out8 = static_cast<uint8_t *>(malloc(max_size));
  MSLITE_CHECK_PTR(out8);
  size_t out_size = 0;
  auto ret = SerializingToBuffer(bs, fse_quant, table_log, max_size, out8, &out_size, compress_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " serializing to buffer failed.";
    free(out8);
    return ret;
  }

  std::shared_ptr<mindspore::tensor::Tensor> compress_tensor;
  if (compress_type == mindspore::kFSE || compress_type == mindspore::kFSEInfer) {
    compress_tensor =
      std::make_shared<mindspore::tensor::Tensor>(kNumberTypeFloat32, tensor_info->shape(), out_size, compress_type);
  } else {
    compress_tensor = std::make_shared<mindspore::tensor::Tensor>(tensor_info->data_type(), tensor_info->shape(),
                                                                  out_size, compress_type);
  }
  if (compress_tensor == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " compress_tensor is nullptr.";
    free(out8);
    return RET_ERROR;
  }

  ret = memcpy_s(compress_tensor->data_c(), out_size, out8, out_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " memcpy failed.";
    free(out8);
    return RET_ERROR;
  }
  free(out8);
  weight->set_default_param(compress_tensor);
  weight->set_abstract(compress_tensor->ToAbstract());
  auto ratio = 1.0 * tensor_info->Size() / compress_tensor->Size();
  MS_LOG(INFO) << weight->fullname_with_scope() << " origin size:" << tensor_info->Size()
               << " compress tensor size:" << compress_tensor->Size() << " compression ratio:" << ratio;
  return RET_OK;
}
}  // namespace mindspore::lite::quant
