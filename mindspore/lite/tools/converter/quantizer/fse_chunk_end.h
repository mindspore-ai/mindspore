/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_CHUNK_END_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_CHUNK_END_H_
#include <cstdint>

namespace mindspore::lite::quant {
struct ChunkEndData {  // this structure is used to assist in parallelising the decoding
  static constexpr size_t SHIFT32 = 32;
  static constexpr size_t SHIFT16 = 16;
  uint32_t bs_position;  // position in the bit stream
  uint16_t state;        // starting state
  uint16_t bit_count;    // number of bits in the last symbol
  uint64_t ToUint64() const {
    uint64_t right_half = static_cast<uint64_t>(state);
    right_half = (right_half << SHIFT16) + bit_count;
    uint64_t ret_val = static_cast<uint64_t>(bs_position);
    return (ret_val << SHIFT32) + right_half;
  }
  void FromUint64(uint64_t val) {
    bs_position = static_cast<uint32_t>(val >> SHIFT32);
    state = static_cast<uint16_t>((val >> SHIFT16) & 0xFFFF);
    bit_count = static_cast<uint16_t>(val & 0xFFFF);
  }
  ChunkEndData() : bs_position(0), state(0), bit_count(0) {}
  explicit ChunkEndData(uint64_t val) { this->FromUint64(val); }
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_CHUNK_END_H_
