/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_SYSTEM_SHA256_H_
#define MINDSPORE_CORE_UTILS_SYSTEM_SHA256_H_

#include <string>
#include "mindapi/base/macros.h"

namespace mindspore {
namespace system {
namespace sha256 {
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ ((~x) & z); }
inline uint32_t ma(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sigma0(uint32_t x) { return (x >> 2 | x << 30) ^ (x >> 13 | x << 19) ^ (x >> 22 | x << 10); }
inline uint32_t sigma1(uint32_t x) { return (x >> 6 | x << 26) ^ (x >> 11 | x << 21) ^ (x >> 25 | x << 7); }
inline uint32_t sigma2(uint32_t x) { return (x >> 7 | x << 25) ^ (x >> 18 | x << 14) ^ (x >> 3); }
inline uint32_t sigma3(uint32_t x) { return (x >> 17 | x << 15) ^ (x >> 19 | x << 13) ^ (x >> 10); }

std::string LoadFilePath(const std::string &path);

bool Padding(std::string *message);

bool ProcessInner(const std::string &message, const int &bias, uint32_t *digest, const int &digest_size);

std::string ConvertToString(const uint32_t *input, const int &size);

std::string Encrypt(const std::string &message);

MS_CORE_API std::string GetHashFromString(const std::string &data);

MS_CORE_API std::string GetHashFromFile(const std::string &path);

#ifndef _WIN32
std::string GetHashFromDir(const std::string &dir);
#endif
}  // namespace sha256
}  // namespace system
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_SYSTEM_SHA256_H_
