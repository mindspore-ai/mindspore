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

#include "utils/system/sha256.h"
#include <dirent.h>
#include <sys/stat.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "securec/include/securec.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace system {
namespace sha256 {
constexpr int kBitNumber = 8;
constexpr int kDigestSize = 8;
constexpr int kIterationNumber = 64;
constexpr int kMessageBlockLength = 64;
const uint32_t constant[64] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

std::string LoadFilePath(const std::string &path) {
  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (path.size() >= PATH_MAX || _fullpath(real_path, path.c_str(), PATH_MAX) == nullptr) {
    return "";
  }
#else
  if (path.size() >= PATH_MAX || realpath(path.c_str(), real_path) == nullptr) {
    return "";
  }
#endif
  std::ifstream bin_stream(real_path, std::ios::binary);
  if (!bin_stream.is_open()) {
    return "";
  }
  std::string message((std::istreambuf_iterator<char>(bin_stream)), std::istreambuf_iterator<char>());
  return message;
}

bool Padding(std::string *message) {
  uint64_t bits_message = message->size() * kBitNumber;
  const int remains = message->size() % kMessageBlockLength;
  // The length of the message needs to be stored in 8 bytes, supplemented at the end of the message.
  const int size_append = 8;
  const int size_required = kMessageBlockLength - size_append;
  const int size_pad = size_required - remains + (size_required > remains ? 0 : kMessageBlockLength);
  if (size_pad < 1 || size_pad > kMessageBlockLength) {
    return false;
  }
  message->push_back(0x80);
  for (int i = 1; i < size_pad; ++i) {
    message->push_back(0x00);
  }
  for (int i = size_append - 1; i >= 0; --i) {
    message->push_back(static_cast<uint8_t>((bits_message >> static_cast<uint32_t>(i * kBitNumber)) & 0xff));
  }
  return true;
}

bool ProcessInner(const std::string &message, const int &bias, uint32_t *digest, const int &digest_size) {
  if (digest_size != 8) {  // The number of digests is fixed at 8
    return false;
  }
  uint32_t w[kIterationNumber] = {0};
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(static_cast<uint8_t>(message[bias + i * 4]) & 0xff) << 24) |
           (static_cast<uint32_t>(static_cast<uint8_t>(message[bias + i * 4 + 1]) & 0xff) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(message[bias + i * 4 + 2]) & 0xff) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(message[bias + i * 4 + 3]) & 0xff));
  }
  for (int i = 16; i < kIterationNumber; ++i) {
    w[i] = sigma3(w[i - 2]) + w[i - 7] + sigma2(w[i - 15]) + w[i - 16];
  }

  std::vector<uint32_t> hash(digest_size);
  size_t mem_size = digest_size * sizeof(uint32_t);
  auto ret = memcpy_s(hash.data(), mem_size, digest, mem_size);
  if (ret != EOK) {
    return false;
  }
  for (int i = 0; i < kIterationNumber; ++i) {
    uint32_t t1 = w[i] + constant[i] + hash[7] + sigma1(hash[4]) + ch(hash[4], hash[5], hash[6]);
    uint32_t t2 = sigma0(hash[0]) + ma(hash[0], hash[1], hash[2]);
    for (int j = digest_size - 1; j >= 0; --j) {
      if (j == 4) {
        hash[j] = hash[j - 1] + t1;
      } else if (j == 0) {
        hash[j] = t1 + t2;
      } else {
        hash[j] = hash[j - 1];
      }
    }
  }
  for (int i = 0; i < digest_size; ++i) {
    digest[i] += hash[i];
  }
  return true;
}

std::string ConvertToString(const uint32_t *input, const int &size) {
  std::ostringstream oss;
  oss << std::hex;
  for (int i = 0; i < size; ++i) {
    for (int j = static_cast<int>(sizeof(uint32_t) / sizeof(uint8_t)) - 1; j >= 0; --j) {
      auto val = static_cast<uint8_t>((input[i] >> static_cast<uint32_t>(j * kBitNumber)) & 0xff);
      oss << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(val);
    }
  }
  return oss.str();
}

std::string Encrypt(const std::string &message) {
  uint32_t digest[kDigestSize] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  for (int i = 0; i < static_cast<int>(message.size()); i += kMessageBlockLength) {
    if (!ProcessInner(message, i, digest, kDigestSize)) {
      return "";
    }
  }
  return ConvertToString(digest, kDigestSize);
}

std::string GetHashFromString(const std::string &data) {
  std::string message = data;
  if (message.empty() || !Padding(&message)) {
    return "";
  }
  return Encrypt(message);
}

std::string GetHashFromFile(const std::string &path) {
  std::string message = LoadFilePath(path);
  if (message.empty() || !Padding(&message)) {
    return "";
  }
  return Encrypt(message);
}

#ifndef _WIN32
std::string GetHashFromDir(const std::string &dir) {
  if (dir.empty()) {
    MS_LOG(ERROR) << "The directory path is empty.";
    return "";
  }
  struct stat s {};
  int ret = stat(dir.c_str(), &s);
  if (ret != 0) {
    MS_LOG(ERROR) << "stat dir \"" << dir << "\" failed, ret is : " << ret;
    return "";
  }
  if (!S_ISDIR(s.st_mode)) {
    MS_LOG(ERROR) << "The path \"" << dir << "\" is not a directory.";
    return "";
  }
  DIR *open_dir = opendir(dir.c_str());
  if (open_dir == nullptr) {
    MS_LOG(ERROR) << "open dir " << dir.c_str() << " failed";
    return "";
  }
  struct dirent *filename;
  std::vector<std::string> file_hashes;
  while ((filename = readdir(open_dir)) != nullptr) {
    std::string d_name = std::string(filename->d_name);
    if (d_name == "." || d_name == ".." || filename->d_type != DT_REG) {
      continue;
    }
    file_hashes.emplace_back(GetHashFromFile(std::string(dir) + "/" + filename->d_name));
  }
  closedir(open_dir);
  std::sort(file_hashes.begin(), file_hashes.end());
  auto dir_hash = std::accumulate(file_hashes.begin(), file_hashes.end(), std::string{});
  return dir_hash;
}
#endif
}  // namespace sha256
}  // namespace system
}  // namespace mindspore
