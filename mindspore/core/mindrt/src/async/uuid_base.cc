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

#include "async/uuid_base.h"
#include <memory.h>
#include <atomic>
#include <random>

namespace mindspore {
namespace uuids {
constexpr int DASH_POS0 = 4;
constexpr int DASH_POS1 = 6;
constexpr int DASH_POS2 = 8;
constexpr int DASH_POS3 = 10;
constexpr int SHIFT_BIT = 4;

const uint8_t *uuid::BeginAddress() const { return uuidData; }

const uint8_t *uuid::EndAddress() const { return uuidData + UUID_SIZE; }

std::size_t uuid::Size() { return UUID_SIZE; }

std::string uuid::ToBytes(const uuid &u) {
  BUS_ASSERT(sizeof(u) == UUID_SIZE);
  return std::string(reinterpret_cast<const char *>(u.uuidData), sizeof(u.uuidData));
}

Option<uuid> uuid::FromBytes(const std::string &s) {
  if (s.size() != UUID_SIZE) {
    return MindrtNone();
  }
  uuid u;
  memcpy(&u.uuidData, s.data(), s.size());

  return u;
}

Option<unsigned char> uuid::GetValue(char c) {
  static char const digitsBegin[] = "0123456789abcdefABCDEF";
  static const size_t digitsLen = (sizeof(digitsBegin) / sizeof(char)) - 1;
  static const char *const digitsEnd = digitsBegin + digitsLen;
  static unsigned char const values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15};
  size_t pos = std::find(digitsBegin, digitsEnd, c) - digitsBegin;
  if (pos >= digitsLen) {
    MS_LOG(ERROR) << "invalid char";
    return MindrtNone();
  }
  return values[pos];
}

Option<uuid> uuid::FromString(const std::string &s) {
  auto sBegin = s.begin();
  if (sBegin == s.end()) {
    return MindrtNone();
  }
  auto c = *sBegin;
  bool hasOpenBrace = (c == '{');
  bool hasDashes = false;
  if (hasOpenBrace) {
    ++sBegin;
  }
  uuid u;
  for (size_t i = 0; sBegin != s.end(); ++i) {
    c = *(sBegin++);
    if ((i == DASH_POS0) && (c == '-')) {
      hasDashes = true;
      c = *(sBegin++);
    } else if ((i == DASH_POS1 || i == DASH_POS2 || i == DASH_POS3) && (hasDashes == true)) {
      if (c == '-' && sBegin != s.end()) {
        c = *(sBegin++);
      } else {
        MS_LOG(ERROR) << "str invalid";
        return MindrtNone();
      }
    }
    Option<unsigned char> oc1 = GetValue(c);
    if (oc1.IsNone()) {
      return MindrtNone();
    }
    u.uuidData[i] = oc1.Get();
    if (sBegin != s.end()) {
      c = *(sBegin++);
    }
    u.uuidData[i] <<= SHIFT_BIT;
    Option<unsigned char> oc2 = GetValue(c);
    if (oc2.IsNone()) {
      return MindrtNone();
    }
    u.uuidData[i] |= oc2.Get();
  }
  if ((hasOpenBrace && (c != '}')) || (sBegin != s.end())) {
    MS_LOG(ERROR) << "No } end or leng invalid";
    return MindrtNone();
  }
  return u;
}

// To check whether uuid looks like 0000000-000-000-000-000000000000000
bool uuid::IsNilUUID() const {
  for (std::size_t i = 0; i < Size(); i++) {
    if (uuidData[i]) {
      return false;
    }
  }

  return true;
}

const uint8_t *uuid::Get() const { return uuidData; }

uint8_t *uuid::BeginAddress() { return uuidData; }

uint8_t *uuid::EndAddress() { return uuidData + UUID_SIZE; }

uuid RandomBasedGenerator::GenerateRandomUuid() {
  const int VARIANT_BIT_OFFSET = 8;
  const int VERSION_BIT_OFFSET = 6;
  const int RIGHT_SHIFT_BITS = 8;
  uuid tmpUUID;

  // This is used to generate a random number as a random seed
  std::random_device rd;

  // Mersenne Twister algorithm, as a generator engine,
  // which is used to generate a random number
  std::mt19937 gen(rd());

  // We use uniform distribution
  std::uniform_int_distribution<uint64_t> distribution((std::numeric_limits<uint64_t>::min)(),
                                                       (std::numeric_limits<uint64_t>::max)());

  uint64_t randomValue = distribution(gen);

  unsigned int i = 0;
  for (uint8_t *it = tmpUUID.BeginAddress(); it != tmpUUID.EndAddress(); ++it, ++i) {
    if (i == sizeof(uint64_t)) {
      randomValue = distribution(gen);
      i = 0;
    }

    *it = static_cast<uint8_t>((randomValue >> (i * RIGHT_SHIFT_BITS)) & 0xFF);
  }

  // use atomic ++ to replace random
  static std::atomic<uint64_t> ul(1);
  uint64_t lCount = ul.fetch_add(1);
  uint64_t offSet = distribution(gen) % RIGHT_SHIFT_BITS;
  auto ret = memcpy(tmpUUID.BeginAddress() + offSet, &lCount, sizeof(lCount));
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s error.";
    BUS_OOM_EXIT(tmpUUID.BeginAddress());
  }

  // set the variant
  *(tmpUUID.BeginAddress() + VARIANT_BIT_OFFSET) &= 0xBF;
  *(tmpUUID.BeginAddress() + VARIANT_BIT_OFFSET) |= 0x80;

  // set the uuid generation version
  *(tmpUUID.BeginAddress() + VERSION_BIT_OFFSET) &= 0x4F;
  *(tmpUUID.BeginAddress() + VERSION_BIT_OFFSET) |= 0x40;

  return tmpUUID;
}
}  // namespace uuids
}  // namespace mindspore
