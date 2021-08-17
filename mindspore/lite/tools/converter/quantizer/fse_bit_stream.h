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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSEBITSTREAM_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSEBITSTREAM_H
#include <cstdint>

namespace mindspore::lite::quant {
class BitStream {
 public:
  BitStream() = default;

  ~BitStream() = default;

 public:
  int Create(int bit_capacity);
  void Free();
  void Empty();
  int64_t Pop(uint8_t bit_count);
  void Push(int64_t state, uint8_t bit_count);
  void Flush();

  int32_t GetCurrChunkIndex() { return this->curr_chunk_index_; }
  uint64_t GetCurrChunk() { return this->curr_chunk_; }
  int8_t GetCurrBitCount() { return this->curr_bit_count_; }
  uint64_t *GetChunks() { return this->chunks_; }
  int GetChunkCount() { return this->chunk_count_; }

  void SetCurrChunkIndex(int32_t curr_chunk_index) { this->curr_chunk_index_ = curr_chunk_index; }
  void SetCurrChunk(uint64_t curr_chunk) { this->curr_chunk_ = curr_chunk; }
  void SetCurrBitCount(int8_t curr_bit_count) { this->curr_bit_count_ = curr_bit_count; }
  void SetChunks(uint64_t *chunks) { this->chunks_ = chunks; }
  void SetChunkCount(int chunk_count) { this->chunk_count_ = chunk_count; }

 private:
  int32_t curr_chunk_index_{-1};  // the index of the next chunk that we will write to
  uint64_t curr_chunk_{0};
  int8_t curr_bit_count_{0};   // the number of bits that are currently written in the register.
  uint64_t *chunks_{nullptr};  // the actual memory
  int chunk_count_{0};         // the number of chunks
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_
