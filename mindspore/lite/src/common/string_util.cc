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

#include <algorithm>
#include "src/common/string_util.h"
#include "include/ms_tensor.h"

namespace mindspore {
namespace lite {

std::vector<StringPack> ParseTensorBuffer(Tensor *tensor) {
  if (tensor->data_c() == nullptr) {
    MS_LOG(ERROR) << "Tensor data is null, cannot be parsed";
    return std::vector<StringPack>{};
  }
  return ParseStringBuffer(tensor->MutableData());
}

std::vector<StringPack> ParseStringBuffer(const void *data) {
  std::vector<StringPack> buffer;
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr";
    return buffer;
  }
  const auto *offset = reinterpret_cast<const int32_t *>(data);
  int32_t num = *offset;
  for (int i = 0; i < num; i++) {
    offset += 1;
    buffer.push_back(StringPack{(*(offset + 1)) - (*offset), reinterpret_cast<const char *>(data) + (*offset)});
  }
  return buffer;
}

int WriteStringsToTensor(Tensor *tensor, const std::vector<StringPack> &string_buffer) {
  int32_t num = string_buffer.size();
  std::vector<int32_t> offset(num + 1);
  offset[0] = 4 * (num + 2);
  for (int i = 0; i < num; i++) {
    offset[i + 1] = offset[i] + string_buffer[i].len;
  }
  std::vector<int> shape = {offset[num]};
  tensor->set_shape(shape);
  tensor->set_data_type(kObjectTypeString);
  tensor->FreeData();
  void *data = tensor->MutableData();
  if (data == nullptr) {
    return RET_ERROR;
  }

  auto *string_info = reinterpret_cast<int32_t *>(data);
  char *string_data = reinterpret_cast<char *>(data);

  string_info[0] = num;
  for (int i = 0; i <= num; i++) {
    string_info[i + 1] = offset[i];
  }
  for (int i = 0; i < num; i++) {
    memcpy(string_data + offset[i], string_buffer[i].data, string_buffer[i].len);
  }
  return RET_OK;
}

int WriteSeperatedStringsToTensor(Tensor *tensor, const std::vector<std::vector<StringPack>> &string_buffer) {
  int32_t num = string_buffer.size();
  std::vector<int32_t> offset(num + 1);
  offset[0] = 4 * (num + 2);
  std::vector<int> len(num);
  for (int i = 0; i < num; i++) {
    len[i] = 0;
    for (int j = 0; j < static_cast<int>(string_buffer[i].size()); j++) {
      len[i] += string_buffer[i][j].len;
    }
    offset[i + 1] = offset[i] + len[i];
  }

  std::vector<int> shape = {offset[num]};
  tensor->set_shape(shape);
  tensor->FreeData();
  void *data = tensor->MutableData();
  if (data == nullptr) {
    return RET_ERROR;
  }

  auto *string_info = reinterpret_cast<int32_t *>(data);
  auto *string_data = reinterpret_cast<char *>(data);

  string_info[0] = num;
  for (int i = 0; i <= num; i++) {
    string_info[i + 1] = offset[i];
  }
  for (int i = 0; i < num; i++) {
    auto *dst = string_data + offset[i];
    for (auto string_part : string_buffer[i]) {
      memcpy(dst, string_part.data, string_part.len);
      dst += string_part.len;
    }
  }
  return RET_OK;
}

int GetStringCount(const void *data) { return *(static_cast<const int32_t *>(data)); }

int GetStringCount(Tensor *tensor) { return GetStringCount(tensor->MutableData()); }

int StringsToMSTensor(const std::vector<std::string> &inputs, tensor::MSTensor *tensor) {
  if (tensor == nullptr) {
    return RET_PARAM_INVALID;
  }
  std::vector<StringPack> all_pack;
  for (auto &input : inputs) {
    StringPack pack = {static_cast<int>(input.length()), input.data()};
    all_pack.push_back(pack);
  }
  return WriteStringsToTensor(static_cast<Tensor *>(tensor), all_pack);
}

std::vector<std::string> MSTensorToStrings(const tensor::MSTensor *tensor) {
  if (tensor == nullptr) {
    return {""};
  }
  const void *ptr = static_cast<const Tensor *>(tensor)->data_c();
  std::vector<StringPack> all_pack = ParseStringBuffer(ptr);
  std::vector<std::string> result(all_pack.size());
  std::transform(all_pack.begin(), all_pack.end(), result.begin(), [](StringPack &pack) {
    std::string str(pack.data, pack.len);
    return str;
  });
  return result;
}

// Some primes between 2^63 and 2^64
static uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static uint64_t k1 = 0xb492b66fbe98f273ULL;
static uint64_t k2 = 0x9ae16a3b2f90404fULL;

uint64_t Fetch64Bit(const char *p) {
  uint64_t result = 0;
  memcpy(&result, p, sizeof(uint64_t));
  return result;
}

uint32_t Fetch32Bit(const char *p) {
  uint32_t result = 0;
  memcpy(&result, p, sizeof(uint32_t));
  return result;
}

uint64_t Rotate64(uint64_t value, int shift) {
  return shift == 0 ? value : ((value >> shift) | (value << (64 - shift)));
}

uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t multiple) {
  uint64_t a = (u ^ v) * multiple;
  a ^= (a >> 47);
  uint64_t b = (v ^ a) * multiple;
  b ^= (b >> 47);
  b *= multiple;
  return b;
}

uint64_t ShiftMix(uint64_t value) { return value ^ (value >> 47); }

uint64_t HashStringLen0to16(const char *s, size_t len) {
  if (len >= 8) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch64Bit(s) + k2;
    uint64_t b = Fetch64Bit(s + len - 8);
    uint64_t c = Rotate64(b, 37) * mul + a;
    uint64_t d = (Rotate64(a, 25) + b) * mul;
    return HashLen16(c, d, mul);
  }
  if (len >= 4) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch32Bit(s);
    return HashLen16(len + (a << 3), Fetch32Bit(s + len - 4), mul);
  }
  if (len > 0) {
    uint8_t a = s[0];
    uint8_t b = s[len >> 1];
    uint8_t c = s[len - 1];
    uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
    uint32_t z = len + (static_cast<uint32_t>(c) << 2);
    return ShiftMix(y * k2 ^ z * k0) * k2;
  }
  return k2;
}

uint64_t HashStringLen17to32(const char *s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch64Bit(s) * k1;
  uint64_t b = Fetch64Bit(s + 8);
  uint64_t c = Fetch64Bit(s + len - 8) * mul;
  uint64_t d = Fetch64Bit(s + len - 16) * k2;
  return HashLen16(Rotate64(a + b, 43) + Rotate64(c, 30) + d, a + Rotate64(b + k2, 18) + c, mul);
}

uint64_t HashStringLen33to64(const char *s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch64Bit(s) * k2;
  uint64_t b = Fetch64Bit(s + 8);
  uint64_t c = Fetch64Bit(s + len - 8) * mul;
  uint64_t d = Fetch64Bit(s + len - 16) * k2;
  uint64_t y = Rotate64(a + b, 43) + Rotate64(c, 30) + d;
  uint64_t z = HashLen16(y, a + Rotate64(b + k2, 18) + c, mul);
  uint64_t e = Fetch64Bit(s + 16) * mul;
  uint64_t f = Fetch64Bit(s + 24);
  uint64_t g = (y + Fetch64Bit(s + len - 32)) * mul;
  uint64_t h = (z + Fetch64Bit(s + len - 24)) * mul;
  return HashLen16(Rotate64(e + f, 43) + Rotate64(g, 30) + h, e + Rotate64(f + a, 18) + g, mul);
}

std::pair<uint64_t, uint64_t> HashLen32WithSeeds(const char *s, uint64_t a, uint64_t b) {
  a += Fetch64Bit(s);
  b = Rotate64(b + a + Fetch64Bit(s + 24), 21);
  uint64_t c = a;
  a += Fetch64Bit(s + 8);
  a += Fetch64Bit(s + 16);
  b += Rotate64(a, 44);
  return std::make_pair(a + Fetch64Bit(s + 24), b + c);
}

uint64_t StringHash64(const char *s, size_t len) {
  const uint64_t seed_value = 81;
  if (len <= 16) {
    return HashStringLen0to16(s, len);
  } else if (len <= 32) {
    return HashStringLen17to32(s, len);
  } else if (len <= 64) {
    return HashStringLen33to64(s, len);
  }

  uint64_t x = seed_value;
  uint64_t y = seed_value * k1 + 113;
  uint64_t tmp = y * k2 + 113;
  uint64_t z = (tmp ^ (tmp >> 47)) * k2;
  std::pair<uint64_t, uint64_t> v = std::make_pair(0, 0);
  std::pair<uint64_t, uint64_t> w = std::make_pair(0, 0);
  x = x * k2 + Fetch64Bit(s);

  const char *end = s + ((len - 1) / 64) * 64;
  const char *last64 = end + ((len - 1) & 63) - 63;
  MS_ASSERT(s + len - 64 == last64);
  do {
    x = Rotate64(x + y + v.first + Fetch64Bit(s + 8), 37) * k1;
    y = Rotate64(y + v.second + Fetch64Bit(s + 48), 42) * k1;
    x ^= w.second;
    y += v.first + Fetch64Bit(s + 40);
    z = Rotate64(z + w.first, 33) * k1;
    v = HashLen32WithSeeds(s, v.second * k1, x + w.first);
    w = HashLen32WithSeeds(s + 32, z + w.second, y + Fetch64Bit(s + 16));
    std::swap(z, x);
    s += 64;
  } while (s != end);
  uint64_t mul = k1 + ((z & 0xff) << 1);
  s = last64;
  w.first += ((len - 1) & 63);
  v.first += w.first;
  w.first += v.first;
  x = Rotate64(x + y + v.first + Fetch64Bit(s + 8), 37) * mul;
  y = Rotate64(y + v.second + Fetch64Bit(s + 48), 42) * mul;
  x ^= w.second * 9;
  y += v.first * 9 + Fetch64Bit(s + 40);
  z = Rotate64(z + w.first, 33) * mul;
  v = HashLen32WithSeeds(s, v.second * mul, x + w.first);
  w = HashLen32WithSeeds(s + 32, z + w.second, y + Fetch64Bit(s + 16));
  std::swap(z, x);
  return HashLen16(HashLen16(v.first, w.first, mul) + ShiftMix(y) * k0 + z, HashLen16(v.second, w.second, mul) + x,
                   mul);
}
}  // namespace lite
}  // namespace mindspore
