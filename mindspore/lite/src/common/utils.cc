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

#ifdef __ANDROID__
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
std::vector<std::string> StringSplit(std::string str, const std::string &pattern) {
  std::vector<std::string> result;
  if (str.empty()) {
    return result;
  }
  std::string::size_type pos;
  str += pattern;
  auto size = str.size();

  for (size_t i = 0; i < size; i++) {
    pos = str.find(pattern, i);
    if (pos < size) {
      std::string s = str.substr(i, pos - i);
      result.push_back(s);
      i = pos + pattern.size() - 1;
    }
  }
  return result;
}

uint64_t GetTimeUs() {
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  // USECS_IN_SEC *NSECS_IN_USEC;
  auto ret_val = static_cast<uint64_t>((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return ret_val;
}

std::string RemoveSubStr(const std::string &from, const std::string &sub_str, RemoveSubStrMode mode) {
  std::string result = from;
  if (from.empty()) {
    MS_LOG(ERROR) << "string is empty";
    return "";
  }
  if (mode == PREFIX) {
    if (from.substr(0, sub_str.length()) == sub_str) {
      result = from.substr(sub_str.size());
    }
  } else if (mode == SUFFIX) {
    if (from.rfind(sub_str) == from.size() - sub_str.size()) {
      result = from.substr(0, from.size() - sub_str.size());
    }
  } else {
    size_t index;
    while ((index = result.find(sub_str)) != std::string::npos) {
      result = result.erase(index, sub_str.size());
    }
  }

  return result;
}

std::vector<std::string> StrSplit(const std::string &str, const std::string &pattern) {
  if (str.empty()) {
    MS_LOG(ERROR) << "string is empty";
    return {};
  }
  std::string::size_type pos;
  std::vector<std::string> result;
  std::string tmpStr(str + pattern);
  std::string::size_type size = tmpStr.size();

  for (std::string::size_type i = 0; i < size; i++) {
    pos = tmpStr.find(pattern, i);
    if (pos < size) {
      std::string s = tmpStr.substr(i, pos - i);
      result.push_back(s);
      i = pos + pattern.size() - 1;
    }
  }
  return result;
}

std::vector<std::string> Tokenize(const std::string &src, const std::string &delimiters,
                                  const Option<size_t> &max_token_num) {
  if (max_token_num.IsSome() && max_token_num.Get() == 0) {
    return {};
  }

  if (src.empty()) {
    MS_LOG(ERROR) << "string is empty";
    return {};
  }

  std::vector<std::string> tokens;
  size_t offset = 0;

  while (true) {
    size_t nonDelimiter = src.find_first_not_of(delimiters, offset);
    if (nonDelimiter == std::string::npos) {
      break;
    }
    size_t delimiter = src.find_first_of(delimiters, nonDelimiter);
    if (delimiter == std::string::npos || (max_token_num.IsSome() && tokens.size() == max_token_num.Get() - 1)) {
      tokens.push_back(src.substr(nonDelimiter));
      break;
    }

    tokens.push_back(src.substr(nonDelimiter, delimiter - nonDelimiter));
    offset = delimiter;
  }
  return tokens;
}

#if defined(__ANDROID__)
uint32_t getHwCap(int hwcap_type) {
  uint32_t ret = getauxval(hwcap_type);
  return ret;
}
#endif

bool IsSupportSDot() {
  bool status = false;
#ifdef ENABLE_ARM64
#if defined(__ANDROID__)
  int hwcap_type = 16;
  uint32_t hwcap = getHwCap(hwcap_type);
  if (hwcap & HWCAP_ASIMDDP) {
    MS_LOG(DEBUG) << "Hw cap support SMID Dot Product, hwcap: 0x" << hwcap;
    status = true;
  } else {
    MS_LOG(DEBUG) << "Hw cap NOT support SIMD Dot Product, hwcap: 0x" << hwcap;
    status = false;
  }
#endif
#endif
  return status;
}

bool IsSupportFloat16() {
  bool status = false;
#ifdef ENABLE_ARM64
#if defined(__ANDROID__)
  int hwcap_type = 16;
  uint32_t hwcap = getHwCap(hwcap_type);
  if (hwcap & HWCAP_FPHP) {
    MS_LOG(DEBUG) << "Hw cap support FP16, hwcap: 0x" << hwcap;
    status = true;
  } else {
    MS_LOG(DEBUG) << "Hw cap NOT support FP16, hwcap: 0x" << hwcap;
    status = false;
  }
#endif
#endif
  return status;
}
}  // namespace lite
}  // namespace mindspore
