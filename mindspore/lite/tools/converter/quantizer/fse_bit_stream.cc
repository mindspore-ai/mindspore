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
constexpr int8_t kCurrentBitCount = 64;
constexpr int8_t kTableSize = 6;
}  // namespace
int BitStream::Create(int bit_capacity) {
  chunk_count_ = (bit_capacity >> kTableSize);
  chunks_ = static_cast<uint64_t *>(calloc(chunk_count_, sizeof(uint64_t)));
  if (chunks_ == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

void BitStream::Free() {
  curr_chunk_index_ = -1;
  curr_chunk_ = 0;
  curr_bit_count_ = 0;
  chunk_count_ = 0;
  if (chunks_ != nullptr) {
    free(chunks_);
    chunks_ = nullptr;
  }
}

void BitStream::Empty() {
  curr_chunk_index_ = -1;
  curr_chunk_ = 0;
  curr_bit_count_ = 0;
  for (int i = 0; i < chunk_count_; i++) {
    chunks_[i] = 0;
  }
}

int64_t BitStream::Pop(uint8_t bit_count) {
  MS_ASSERT(curr_bit_count_ <= kCurrentBitCount);
  int64_t right = curr_chunk_ >> (kCurrentBitCount - curr_bit_count_);
  int64_t res = right & ((1 << bit_count) - 1);
  curr_bit_count_ -= bit_count;
  if (curr_bit_count_ > 0) {
    // most likely branch
    return res;
  }
  if (curr_bit_count_ == 0) {
    // not so often...
    if (curr_chunk_index_ > -1) {
      // rare...
      curr_bit_count_ = kCurrentBitCount;
      curr_chunk_ = chunks_[curr_chunk_index_--];
    }
    return res;
  }
  // sad path :(
  curr_bit_count_ += bit_count;
  curr_chunk_ = chunks_[curr_chunk_index_--];
  right |= (curr_chunk_ & ((1 << (bit_count - curr_bit_count_)) - 1)) << curr_bit_count_;
  curr_bit_count_ = kCurrentBitCount - (bit_count - curr_bit_count_);
  return right;
}

void BitStream::Push(int64_t state, uint8_t bit_count) {
  curr_bit_count_ += bit_count;
  if (curr_bit_count_ <= kCurrentBitCount) {
    // happy path, no split
    curr_chunk_ = (curr_chunk_ << bit_count) | (state & ((1 << bit_count) - 1));
    if (curr_bit_count_ == kCurrentBitCount) {
      // flush (rare)
      chunks_[++curr_chunk_index_] = curr_chunk_;
      curr_chunk_ = 0;
      curr_bit_count_ = 0;
    }
  } else {
    // split, rare
    int left_bits = curr_bit_count_ - kCurrentBitCount;
    int right_bits = bit_count - left_bits;
    curr_chunk_ = (curr_chunk_ << right_bits) | ((state >> left_bits) & ((1 << right_bits) - 1));
    // flush left
    chunks_[++curr_chunk_index_] = curr_chunk_;
    curr_chunk_ = state & ((1 << left_bits) - 1);
    curr_bit_count_ = left_bits;
  }
}

void BitStream::Flush() { curr_chunk_ <<= kCurrentBitCount - curr_bit_count_; }
}  // namespace mindspore::lite::quant
