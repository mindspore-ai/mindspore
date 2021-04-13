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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_UUID_BASE_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_UUID_BASE_H

#include <stdint.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include "async/option.h"

namespace mindspore {
namespace uuids {
const std::size_t UUID_SIZE = 16;

struct uuid {
 public:
  static std::size_t Size();

  static std::string ToBytes(const uuid &u);

  static Option<uuid> FromBytes(const std::string &s);

  static Option<unsigned char> GetValue(char c);

  static Option<uuid> FromString(const std::string &s);

  // To check whether uuid looks like 0000000-000-000-000-000000000000000
  bool IsNilUUID() const;

  const uint8_t *Get() const;

 private:
  const uint8_t *BeginAddress() const;

  const uint8_t *EndAddress() const;

  uint8_t *BeginAddress();

  uint8_t *EndAddress();

  friend class RandomBasedGenerator;
  friend bool operator==(uuid const &left, uuid const &right);
  friend bool operator!=(uuid const &left, uuid const &right);
  template <typename T, typename F>
  friend std::basic_ostream<T, F> &operator<<(std::basic_ostream<T, F> &s, const struct uuid &outputUuid);
  uint8_t uuidData[UUID_SIZE];
};

class RandomBasedGenerator {
 public:
  static uuid GenerateRandomUuid();
};

// operator override
inline bool operator==(uuid const &left, uuid const &right) {
  return std::equal(left.BeginAddress(), left.EndAddress(), right.BeginAddress());
}

// operator override
inline bool operator!=(uuid const &left, uuid const &right) { return !(left == right); }

// operator override
template <typename T, typename F>
std::basic_ostream<T, F> &operator<<(std::basic_ostream<T, F> &s, const struct uuid &outputUuid) {
  const int FIRST_DELIM_OFFSET = 3;
  const int SECOND_DELIM_OFFSET = 5;
  const int THIRD_DELIM_OFFSET = 7;
  const int FOURTH_DELIM_OFFSET = 9;
  const int UUID_WIDTH = 2;
  s << std::hex << std::setfill(static_cast<T>('0'));

  int i = 0;
  for (const uint8_t *ptr = outputUuid.BeginAddress(); ptr < outputUuid.EndAddress(); ++ptr, ++i) {
    s << std::setw(UUID_WIDTH) << static_cast<int>(*ptr);
    if (i == FIRST_DELIM_OFFSET || i == SECOND_DELIM_OFFSET || i == THIRD_DELIM_OFFSET || i == FOURTH_DELIM_OFFSET) {
      s << '-';
    }
  }

  s << std::setfill(static_cast<T>(' ')) << std::dec;
  return s;
}
}  // namespace uuids
}  // namespace mindspore
#endif /* UUID_BASE_HPP_ */
