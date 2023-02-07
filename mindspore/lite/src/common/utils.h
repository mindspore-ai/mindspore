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

#ifndef MINDSPORE_LITE_SRC_COMMON_UTILS_H_
#define MINDSPORE_LITE_SRC_COMMON_UTILS_H_

#include <ctime>
#include <cstdint>
#include <vector>
#include <set>
#include <limits>
#include <cmath>
#include <string>
#include <utility>
#include "src/common/log_adapter.h"
#include "tools/common/option.h"
#include "include/errorcode.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace lite {
enum NodeType {
  NodeType_ValueNode,  // const
  NodeType_Parameter,  // var
  NodeType_CNode       // op
};

const int USEC = 1000000;
const int MSEC = 1000;
uint64_t GetTimeUs();

bool IsSupportSDot();

size_t GetMaxMallocSize();

int GetCoreNum();

#ifdef __ANDROID__
uint32_t getHwCap(int hwcap_type);
#endif

template <typename T>
bool IsContain(const std::vector<T> &vec, T element) {
  for (auto iter = vec.begin(); iter != vec.end(); iter++) {
    if (*iter == element) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool VectorErase(std::vector<T> *vec, T element) {
  bool ret = false;
  for (auto iter = vec->begin(); iter != vec->end();) {
    if (*iter == element) {
      iter = vec->erase(iter);
      ret = true;
    } else {
      iter++;
    }
  }
  return ret;
}

template <typename T>
bool VectorSetNull(std::vector<T> *vec, T element) {
  bool ret = false;
  for (size_t i = 0; i < vec->size(); i++) {
    if (vec->at(i) == element) {
      vec->at(i) = nullptr;
    }
  }
  return ret;
}

template <typename T>
bool VectorReplace(std::vector<T> *vec, T srcElement, T dstElement) {
  bool ret = false;
  for (auto iter = vec->begin(); iter != vec->end(); iter++) {
    if (*iter == srcElement) {
      if (!IsContain(*vec, dstElement)) {
        *iter = std::move(dstElement);
      } else {
        vec->erase(iter);
      }
      ret = true;
      break;
    }
  }
  return ret;
}

template <typename T>
bool CommonCheckTensorType(const std::vector<T *> &tensors, size_t index, TypeId input_type) {
  if (tensors.at(index) == nullptr) {
    MS_LOG(ERROR) << "Tensors index: " << index << " is a nullptr";
    return false;
  }
  if (tensors.at(index)->data_type() != input_type) {
    MS_LOG(ERROR) << "Invalid tensor[" << index << "] data_type: " << tensors.at(index)->data_type();
    return false;
  }
  return true;
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
  if (path.empty()) {
    MS_LOG(ERROR) << "string is empty";
    return "";
  }

  char delim = '/';

  size_t i = path.rfind(delim, path.length());
  if (i != std::string::npos && i + 1 < path.length()) {
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

bool ConvertStrToInt(const std::string &str, int *value);

bool ParseShapeStr(const std::string &shape_str, std::vector<int64_t> *shape_ptr);

// tokenize string
std::vector<std::string> Tokenize(const std::string &src, const std::string &delimiters,
                                  const Option<size_t> &max_token_num = Option<size_t>(None()));

enum RemoveSubStrMode { PREFIX, SUFFIX, ANY };

// remove redundant character
std::string RemoveSubStr(const std::string &from, const std::string &sub_str, RemoveSubStrMode mode = ANY);

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

inline size_t DataTypeSize(TypeId type) {
  switch (type) {
    case kNumberTypeFloat64:
      return sizeof(double);
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return sizeof(float);
    case kNumberTypeInt8:
      return sizeof(int8_t);
    case kNumberTypeUInt8:
      return sizeof(uint8_t);
    case kNumberTypeFloat16:
    case kNumberTypeInt16:
      return sizeof(int16_t);
    case kNumberTypeInt32:
      return sizeof(int32_t);
    case kNumberTypeInt64:
      return sizeof(int64_t);
    case kNumberTypeUInt16:
      return sizeof(uint16_t);
    case kNumberTypeUInt32:
      return sizeof(uint32_t);
    case kNumberTypeUInt64:
      return sizeof(uint64_t);
    case kNumberTypeBool:
      return sizeof(bool);
    case kObjectTypeString:
      return sizeof(char);
    case kObjectTypeTensorType:
      return 0;
    case kMetaTypeTypeType:
      return sizeof(int);
    default:
      MS_LOG(ERROR) << "Not support the type: " << type;
      return 0;
  }
}

inline bool FloatCompare(const float &a, const float &b = 0.0f) {
  return std::fabs(a - b) <= std::numeric_limits<float>::epsilon();
}

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_UTILS_H_
