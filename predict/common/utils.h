/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_COMMON_UTILS_H_
#define PREDICT_COMMON_UTILS_H_

#include <stdint.h>
#include <ctime>
#include <cstdint>
#include <vector>
#include <string>
#include "common/mslog.h"
#include "common/option.h"
#include "include/errorcode.h"

namespace mindspore {
namespace predict {
const int USEC = 1000000;
const int MSEC = 1000;

uint64_t GetTimeUs();

int16_t Float32ToShort(float srcValue);

float ShortToFloat32(int16_t srcValue);

void ShortToFloat32(const int16_t *srcData, float *dstData, size_t elementSize);

void Float32ToShort(const float *srcData, int16_t *dstData, size_t elementSize);

template <typename T>
bool IsContain(const std::vector<T> &vec, T element) {
  for (auto iter = vec.begin(); iter != vec.end(); iter++) {
    if (*iter == element) {
      return true;
    }
  }
  return false;
}

const char WHITESPACE[] = "\t\n\v\f\r ";
const char STR_TRUE[] = "true";
const char STR_FALSE[] = "false";

template <typename T>
Option<std::string> ToString(T t) {
  std::ostringstream out;
  out << t;
  if (!out.good()) {
    return Option<std::string>(None());
  }

  return Option<std::string>(out.str());
}

template <>
inline Option<std::string> ToString(bool value) {
  return value ? Option<std::string>(STR_TRUE) : Option<std::string>(STR_FALSE);
}

// get the file name from a given path
// for example: "/usr/bin", we will get "bin"
inline std::string GetFileName(const std::string &path) {
  char delim = '/';

  size_t i = path.rfind(delim, path.length());
  if (i != std::string::npos) {
    return (path.substr(i + 1, path.length() - i));
  }

  return "";
}

// trim the white space character in a string
// see also: macro WHITESPACE defined above
inline void Trim(std::string *input) {
  if (input == nullptr) {
    return;
  }
  if (input->empty()) {
    return;
  }

  input->erase(0, input->find_first_not_of(WHITESPACE));
  input->erase(input->find_last_not_of(WHITESPACE) + 1);
}

// to judge whether a string is starting with  prefix
// for example: "hello world" is starting with "hello"
inline bool StartsWithPrefix(const std::string &source, const std::string &prefix) {
  if (source.length() < prefix.length()) {
    return false;
  }

  return (source.compare(0, prefix.length(), prefix) == 0);
}

// split string
std::vector<std::string> StrSplit(const std::string &str, const std::string &pattern);

// tokenize string
std::vector<std::string> Tokenize(const std::string &src, const std::string &delimiters,
                                  const Option<size_t> &maxTokenNum = Option<size_t>(None()));

enum Mode { PREFIX, SUFFIX, ANY };

// remove redundant character
std::string Remove(const std::string &from, const std::string &subStr, Mode mode = ANY);

template <typename T>
inline Option<T> GenericParseValue(const std::string &value) {
  T ret;
  std::istringstream input(value);
  input >> ret;

  if (input && input.eof()) {
    return Option<T>(ret);
  }

  return Option<T>(None());
}

template <>
inline Option<std::string> GenericParseValue(const std::string &value) {
  return Option<std::string>(value);
}

template <>
inline Option<bool> GenericParseValue(const std::string &value) {
  if (value == "true") {
    return Option<bool>(true);
  } else if (value == "false") {
    return Option<bool>(false);
  }

  return Option<bool>(None());
}
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_COMMON_UTILS_H_
