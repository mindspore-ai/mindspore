/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_GENERAL_BITPACKING_H
#define MINDSPORE_GENERAL_BITPACKING_H
#include <stdint.h>
#include <stack>
#include <queue>
#include <vector>
#include <cassert>

namespace mindspore {
namespace lite {
class BitPack {
 public:
  explicit BitPack(const uint8_t &bitbum = 8);
  ~BitPack() = default;
  void BitPacking(const std::vector<uint8_t> &originDataVec, std::vector<uint8_t> &packedDataVec);
  void UnPack(uint8_t bitnum, uint8_t &packedData, std::vector<uint8_t> &originData, std::queue<bool> &unpackBitData);

 private:
  void UnPackFromUint8ToOrigin(uint8_t &n, std::queue<bool> &unpackBitData);
  void PackFromOriginToUint8(std::stack<bool> &ans, std::vector<uint8_t> &packedDataVec);
  void DoBinary(uint8_t &n, std::stack<bool> &ans, std::vector<uint8_t> &packed_data_vec);
  uint8_t bitnum;
};
}  // namespace lite
}  // namespace mindspore

#endif
