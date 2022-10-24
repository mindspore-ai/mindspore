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

#include "tools/converter/quantizer/fse_bit_stream.h"
#include <memory.h>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore::lite::quant {
namespace {
constexpr int8_t kMaxBitCount = 64;
constexpr int8_t kTableSize = 6;
constexpr size_t kInt32Mask = 31;
}  // namespace
int FSEBitStream::Create(uint64_t bit_capacity) {
  chunk_count_ = (bit_capacity >> kTableSize);
  chunks_ = static_cast<uint64_t *>(malloc(chunk_count_ * sizeof(uint64_t)));
  if (chunks_ == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

void FSEBitStream::Free() {
  curr_chunk_index_ = -1;
  curr_chunk_ = 0;
  curr_bit_count_ = 0;
  chunk_count_ = 0;
  if (chunks_ != nullptr) {
    free(chunks_);
    chunks_ = nullptr;
  }
}

void FSEBitStream::Empty() {
  curr_chunk_index_ = -1;
  curr_chunk_ = 0;
  curr_bit_count_ = 0;
  for (uint64_t i = 0; i < chunk_count_; i++) {
    chunks_[i] = 0;
  }
}

uint64_t FSEBitStream::Pop(uint8_t bit_count) {
  MS_ASSERT(curr_bit_count_ <= kMaxBitCount);
  uint64_t right = curr_chunk_ >> static_cast<size_t>(kMaxBitCount - curr_bit_count_);
  uint64_t res = right & ((1u << bit_count) - 1);
  curr_bit_count_ -= static_cast<int8_t>(bit_count);
  if (curr_bit_count_ > 0) {
    // most likely branch
    return res;
  }
  if (curr_bit_count_ == 0) {
    // not so often...
    if (curr_chunk_index_ > -1) {
      // rare...
      curr_bit_count_ = kMaxBitCount;
      curr_chunk_ = chunks_[curr_chunk_index_--];
    }
    return res;
  }
  // sad path :(
  curr_bit_count_ += static_cast<int8_t>(bit_count);
  curr_chunk_ = chunks_[curr_chunk_index_--];
  right |= (curr_chunk_ & ((1u << (static_cast<int8_t>(bit_count) - curr_bit_count_)) - 1)) << curr_bit_count_;
  curr_bit_count_ = kMaxBitCount - (static_cast<int8_t>(bit_count) - curr_bit_count_);
  return right;
}

void FSEBitStream::Push(int64_t state, uint8_t bit_count) {
  curr_bit_count_ += static_cast<int8_t>(bit_count);
  if (curr_bit_count_ <= kMaxBitCount) {
    // happy path, no split
    curr_chunk_ = (curr_chunk_ << bit_count) | (static_cast<size_t>(state) & ((1 << bit_count) - 1));
    if (curr_bit_count_ == kMaxBitCount) {
      // flush (rare)
      chunks_[++curr_chunk_index_] = curr_chunk_;
      curr_chunk_ = 0;
      curr_bit_count_ = 0;
    }
  } else {
    // split, rare
    int left_bits = curr_bit_count_ - kMaxBitCount;
    int right_bits = bit_count - left_bits;
    curr_chunk_ = (curr_chunk_ << right_bits) | ((static_cast<size_t>(state) >> left_bits) & ((1 << right_bits) - 1));
    // flush left
    chunks_[++curr_chunk_index_] = curr_chunk_;
    curr_chunk_ = static_cast<size_t>(state) & ((1 << left_bits) - 1);
    curr_bit_count_ = left_bits;
  }
}

void FSEBitStream::Flush() { curr_chunk_ <<= (kMaxBitCount - curr_bit_count_); }

// The function gives the index of most import `1` in the binary representation.
// e.g. for the number 00100 it gives 2.
size_t FSEBitStream::CountBits(size_t x) {
#ifdef _MSC_VER
  size_t num = 0;
  uint32_t tmp = x;
  tmp |= 1;
  while (!(tmp & INT32_MIN)) {
    num += 1;
    tmp <<= 1;
  }
  return num ^ kInt32Mask;
#else
  return __builtin_clz(x) ^ kInt32Mask;
#endif
}
}  // namespace mindspore::lite::quant
