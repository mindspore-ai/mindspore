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

#include "tools/converter/quantizer/fse_encoder.h"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include "mindspore/core/ir/dtype/type_id.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore::lite::quant {
// The function gives the index of most import `1` in the binary representation.
// e.g. for the number 00100 it gives 2.
int fse_count_bits(int32_t x) { return __builtin_clz(x) ^ 31; }

int FSEEncoder::FSECreateStatesForEncoding(uint16_t *frequency, int frequency_count, int table_log,
                                           uint32_t *delta_bit_count, int16_t *delta_state, uint16_t *coding_table,
                                           uint16_t *symbol_table) {
  MS_ASSERT(frequency != nullptr);
  MS_ASSERT(delta_bit_count != nullptr);
  MS_ASSERT(delta_state != nullptr);
  MS_ASSERT(symbol_table != nullptr);
  MS_ASSERT(coding_table != nullptr);
  int tablesize = 1 << table_log;
  int tablemask = tablesize - 1;
  int step = ((tablesize >> 1) + (tablesize >> 3) + 3);
  int pos = 0;
  // Separate the same symbols, coding will be better if the same characters are distributed evenly across the table.
  for (int sym = 0; sym < frequency_count; sym++) {
    for (int i = 0; i < frequency[sym]; i++) {
      symbol_table[pos] = sym;
      pos = (pos + step) & tablemask;
      while (pos > tablemask) pos = (pos + step) & tablemask;
    }
  }
  if (pos != 0) return 1;

  std::vector<uint16_t> cfreqs(frequency_count + 2);
  cfreqs[0] = 0;
  for (int i = 1; i < frequency_count + 1; i++) {
    cfreqs[i] = cfreqs[i - 1] + frequency[i - 1];
  }
  cfreqs[frequency_count + 1] = cfreqs[frequency_count] + 1;
  for (int i = 0; i < tablesize; i++) {
    uint16_t sym = symbol_table[i];
    coding_table[cfreqs[sym]] = tablesize + i;
    cfreqs[sym] += 1;
  }

  int total = 0;
  for (int sym = 0; sym < frequency_count; sym++) {
    if (frequency[sym] >= 2) {
      int max_bits_out = table_log - fse_count_bits(frequency[sym] - 1);
      int min_state_plus = frequency[sym] << max_bits_out;
      delta_bit_count[sym] = (max_bits_out << 16) - min_state_plus;
      delta_state[sym] = total - frequency[sym];
      total += frequency[sym];
    } else {
      // we assume minimum `frequency` is 1
      delta_bit_count[sym] = (table_log << 16) - (1 << table_log);
      delta_state[sym] = total - 1;
      total++;
    }
  }
  return 0;
}

int ConvertTensor2Quant(schema::TensorT *tensor_input, FSEQuant *quants) {
  MS_ASSERT(tensor_input != nullptr);
  MS_ASSERT(quants != nullptr);
  std::vector<int16_t> dequants;
  for (size_t i = 0; i < tensor_input->data.size() / sizeof(int16_t); ++i) {
    auto data = static_cast<int16_t>(reinterpret_cast<int16_t *>(tensor_input->data.data())[i]);
    dequants.push_back(data);
  }

  int qmin = *min_element(dequants.begin(), dequants.end());
  int qmax = *max_element(dequants.begin(), dequants.end());
  int uncompressed_frequency_count = qmax - qmin + 1;
  std::vector<int> uncompressed_frequency(uncompressed_frequency_count);
  for (int i = 0; i < uncompressed_frequency_count; i++) {
    uncompressed_frequency[i] = 0;
  }
  for (size_t i = 0; i < tensor_input->data.size() / sizeof(int16_t); i++) {
    auto data = static_cast<int16_t>(reinterpret_cast<int16_t *>(tensor_input->data.data())[i]);
    int q = data - qmin;
    uncompressed_frequency[q] += 1;
  }

  std::vector<uint16_t> uncompressed_freqs_to_compressed_sym(uncompressed_frequency_count);
  int sym = 0;
  for (int i = 0; i < uncompressed_frequency_count; i++) {
    if (uncompressed_frequency[i]) {
      if (sym >= MAX_SYMS) return 1;  // too many symbols!
      uncompressed_freqs_to_compressed_sym[i] = sym;
      quants->frequency[sym] = uncompressed_frequency[i];
      quants->centroids[sym] =
        tensor_input->quantParams.front()->varCorr *
          (tensor_input->quantParams.front()->scale - tensor_input->quantParams.front()->zeroPoint) * (i + qmin) +
        tensor_input->quantParams.front()->meanCorr;
      sym++;
    }
  }
  quants->size = sym;
  quants->symbol_table_count = tensor_input->data.size() / sizeof(int16_t);
  quants->symbol_table = static_cast<uint16_t *>(malloc(quants->symbol_table_count * sizeof(uint16_t)));
  if (quants->symbol_table == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return RET_ERROR;
  }
  for (int i = 0; i < quants->symbol_table_count; i++) {
    auto data = static_cast<int16_t>(reinterpret_cast<int16_t *>(tensor_input->data.data())[i]);
    int q = data - qmin;
    sym = uncompressed_freqs_to_compressed_sym[q];
    quants->symbol_table[i] = sym;
  }
  return RET_OK;
}

int FSEEncoder::Compress(schema::TensorT *tensor_input) {
  MS_ASSERT(tensor_input != nullptr);
  int table_log = 0;
  FSEQuant fse_quant;
  ConvertTensor2Quant(tensor_input, &fse_quant);
  NormalizeFrequency(&fse_quant, &table_log);
  BitStream bs;
  int ret;
  ret = bs.Create(16 * fse_quant.symbol_table_count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BitStream Create failed.";
    return ret;
  }
  ret = FSEEncode(&bs, fse_quant.symbol_table, fse_quant.symbol_table_count, fse_quant.frequency, fse_quant.size,
                  table_log);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSE Encode failed.";
    return RET_ERROR;
  }
  bs.Flush();
  // Serializing to out:
  ret = SerializingToOut(tensor_input, &bs, fse_quant, table_log);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Serializing To Out failed.";
    return ret;
  }
  bs.Free();
  free(fse_quant.symbol_table);
  return RET_OK;
}

uint16_t FSEEncoder::FSEEncodeSymbolGetNewState(BitStream *bs, uint16_t sym, uint16_t state,
                                                const uint32_t *delta_bit_count, const int16_t *delta_state,
                                                uint16_t *coding_table) {
  MS_ASSERT(bs != nullptr);
  MS_ASSERT(delta_bit_count != nullptr);
  MS_ASSERT(delta_state != nullptr);
  MS_ASSERT(coding_table != nullptr);
  // It is to determine the number of bits to flush.
  // This is basically one of 2 values, n or n+1, depending on state crossing a threshold.
  uint8_t bits_out = (state + delta_bit_count[sym]) >> 16;
  bs->Push(state, bits_out);
  // subrangeID = state >> nbBitsOut
  return coding_table[(state >> bits_out) + delta_state[sym]];
}

int GetMaxIndex(const uint16_t *arr, int arr_count) {
  MS_ASSERT(arr != nullptr);
  float max = -INFINITY;
  int index = -1;
  for (int i = 0; i < arr_count; i++) {
    if (arr[i] > max) {
      max = arr[i];
      index = i;
    }
  }
  return index;
}

void FSEEncoder::NormalizeFrequency(FSEQuant *q, int *table_log) {
  MS_ASSERT(q != nullptr);
  // The higher the number, the more accurate we'll be to the shannon entropy,
  // but also the larger the table, so `+3` is a good compromise.
  *table_log = std::min(MAX_TABLE_LOG, fse_count_bits((uint32_t)q->size) + 3);
  int new_table_size = 1 << (*table_log);
  int curr_table_size = 0;
  for (int i = 0; i < q->size; i++) {
    curr_table_size += q->frequency[i];
  }

  // normalize
  int updated_table_size = 0;
  float rat = (static_cast<float>(new_table_size)) / curr_table_size;
  for (int i = 0; i < q->size; i++) {
    q->frequency[i] = std::max(1, static_cast<int>(floorf(0.5 + rat * q->frequency[i])));
    updated_table_size += q->frequency[i];
  }

  // If the sum of the symbol frequencies is not equal to the power of two (almost always),
  // then the frequencies need to be normalized-they must be proportionally reduced (or increased) so that the power of
  // two is obtained in total.
  // shrink
  while (updated_table_size > new_table_size) {
    int max_ix = GetMaxIndex(q->frequency, q->size);
    q->frequency[max_ix]--;
    updated_table_size--;
  }

  // grow
  if (updated_table_size < new_table_size) {
    int max_ix = GetMaxIndex(q->frequency, q->size);
    q->frequency[max_ix] += new_table_size - updated_table_size;
  }
}

// Encoding is therefore just a repeat of this process :
// - get Symbol to encode
// - look at current state value
// - determine nbBits, flush them
// - determine sub-Range Id
// - look for Symbol position of same Id : you get your next state
int FSEEncoder::FSEEncode(BitStream *bs, const uint16_t *data, int data_count, uint16_t *frequency, int frequency_count,
                          int table_log) {
  MS_ASSERT(bs != nullptr);
  MS_ASSERT(data != nullptr);
  MS_ASSERT(frequency != nullptr);
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
  state = FSEEncodeSymbolGetNewState(bs, data[0], state, delta_number_bits.data(), delta_find_state.data(),
                                     coding_table.data());
  bs->Empty();
  for (int i = 0; i < data_count; i++) {
    state = FSEEncodeSymbolGetNewState(bs, data[i], state, delta_number_bits.data(), delta_find_state.data(),
                                       coding_table.data());
  }
  bs->Push(state - table_size, table_log);
  return ret;
}

int FSEEncoder::SerializingToOut(schema::TensorT *tensor_input, BitStream *bs, const FSEQuant &fse_quant,
                                 int table_log) {
  MS_ASSERT(tensor_input != nullptr);
  MS_ASSERT(bs != nullptr);
  auto max_size = tensor_input->data.size() * 2;
  auto *out8 = static_cast<uint8_t *>(malloc(max_size));
  if (out8 == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return RET_ERROR;
  }
  int offset = 0;
  *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)fse_quant.size;
  offset += sizeof(uint16_t);
  if (offset + sizeof(uint16_t) > max_size) {
    MS_LOG(ERROR) << "offset over max size"
                  << " offset:" << offset << " max_size:" << max_size;
  }
  *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)table_log;
  offset += sizeof(uint16_t);
  int chunksc = bs->GetCurrChunkIndex() + 2;
  if (offset + sizeof(uint32_t) > max_size) {
    MS_LOG(ERROR) << "offset over max size"
                  << " offset:" << offset << " max_size:" << max_size;
  }
  *(reinterpret_cast<uint32_t *>(&out8[offset])) = (uint32_t)chunksc;
  offset += sizeof(uint32_t);
  for (int j = 0; j < fse_quant.size; j++) {
    if (offset + sizeof(uint16_t) > max_size) {
      MS_LOG(ERROR) << "offset over max size"
                    << " offset:" << offset << " max_size:" << max_size;
    }
    *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)fse_quant.frequency[j];
    offset += sizeof(uint16_t);
  }
  while (offset % 8 != 0) {
    if (offset + sizeof(uint16_t) > max_size) {
      MS_LOG(ERROR) << "offset over max size"
                    << " offset:" << offset << " max_size:" << max_size;
    }
    *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)0;
    offset += sizeof(uint16_t);
  }
  for (int j = 0; j < fse_quant.size; j++) {
    if (offset + sizeof(float) > max_size) {
      MS_LOG(ERROR) << "offset over max size"
                    << " offset:" << offset << " max_size:" << max_size;
    }
    *(reinterpret_cast<float *>(&out8[offset])) = static_cast<float>(fse_quant.centroids[j]);
    offset += sizeof(float);
  }
  while (offset % 8 != 0) {
    if (offset + sizeof(uint16_t) > max_size) {
      MS_LOG(ERROR) << "offset over max size"
                    << " offset:" << offset << " max_size:" << max_size;
    }
    *(reinterpret_cast<uint16_t *>(&out8[offset])) = (uint16_t)0;
    offset += sizeof(uint16_t);
  }
  for (int j = 0; j < bs->GetCurrChunkIndex() + 1; j++) {
    if (offset + sizeof(uint64_t) > max_size) {
      MS_LOG(ERROR) << "offset over max size"
                    << " offset:" << offset << " max_size:" << max_size;
    }
    *(reinterpret_cast<uint64_t *>(&out8[offset])) = (uint64_t)bs->GetChunks()[j];
    offset += sizeof(uint64_t);
  }
  if (offset + sizeof(uint64_t) > max_size) {
    MS_LOG(ERROR) << "offset over max size"
                  << " offset:" << offset << " max_size:" << max_size;
  }
  *(reinterpret_cast<uint64_t *>(&out8[offset])) = (uint64_t)bs->GetCurrChunk();
  offset += sizeof(uint64_t);
  if (offset + sizeof(uint8_t) > max_size) {
    MS_LOG(ERROR) << "offset over max size"
                  << " offset:" << offset << " max_size:" << max_size;
  }
  *(reinterpret_cast<uint8_t *>(&out8[offset])) = (uint8_t)bs->GetCurrBitCount();
  offset += sizeof(uint8_t);
  if (static_cast<int>(offset) < static_cast<int>(tensor_input->data.size())) {
    tensor_input->data.resize(offset);
    if (memcpy_s(tensor_input->data.data(), offset, out8, offset) != EOK) {
      MS_LOG(ERROR) << "memcpy failed.";
    }
  }
  tensor_input->quantParams.clear();
  tensor_input->weightQunatCompressType = schema::WeightQunatCompressType_FSE;
  tensor_input->dataType = TypeId::kNumberTypeFloat32;
  free(out8);
  return RET_OK;
}
}  // namespace mindspore::lite::quant
