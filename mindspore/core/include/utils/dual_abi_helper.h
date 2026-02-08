/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_DUAL_ABI_HELPER_H_
#define MINDSPORE_INCLUDE_API_DUAL_ABI_HELPER_H_

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mindspore {
using VecChar = std::vector<char>;
/// \brief Transform string to char.
///
/// \param[in] s Define the string to transform.
///
/// \return Vector of chars.
inline std::vector<char> StringToChar(const std::string &s) {
  if (s.empty()) {
    const auto empty = std::vector<char>();
    return empty;
  }
  return std::vector<char>(s.begin(), s.end());
}
/// \brief Transform char to string.
///
/// \param[in] c Define the chars to transform.
///
/// \return String.
inline std::string CharToString(const std::vector<char> &c) {
  if (c.empty()) {
    const auto empty = "";
    return empty;
  }
  return std::string(c.begin(), c.end());
}
/// \brief Transform pair of string to char.
///
/// \param[in] s Pair of strings to transform.
///
/// \return Pair of chars.
inline std::pair<std::vector<char>, int32_t> PairStringToChar(const std::pair<std::string, int32_t> &s) {
  return std::pair<std::vector<char>, int32_t>(std::vector<char>(s.first.begin(), s.first.end()), s.second);
}
/// \brief Transform pair of char to string.
///
/// \param[in] c Pair of chars to transform.
///
/// \return Pair of strings.
inline std::pair<std::string, int32_t> PairCharToString(const std::pair<std::vector<char>, int32_t> &c) {
  return std::pair<std::string, int32_t>(std::string(c.first.begin(), c.first.end()), c.second);
}
/// \brief Transform vector of string to chars.
///
/// \param[in] s Vector of strings.
///
/// \return Vector of vector of chars.
inline std::vector<std::vector<char>> VectorStringToChar(const std::vector<std::string> &s) {
  std::vector<std::vector<char>> ret;
  std::transform(s.begin(), s.end(), std::back_inserter(ret),
                 [](auto str) { return std::vector<char>(str.begin(), str.end()); });
  return ret;
}
/// \brief Transform vector of chars to strings.
///
/// \param[in] c Vector of vector ofof schars.
///
/// \return Vector of strings.
inline std::vector<std::string> VectorCharToString(const std::vector<std::vector<char>> &c) {
  std::vector<std::string> ret;
  std::transform(c.begin(), c.end(), std::back_inserter(ret),
                 [](auto ch) { return std::string(ch.begin(), ch.end()); });
  return ret;
}
/// \brief Transform set of strings to chars.
///
/// \param[in] s Set of strings.
///
/// \return Set of vector of chars.
inline std::set<std::vector<char>> SetStringToChar(const std::set<std::string> &s) {
  std::set<std::vector<char>> ret;
  std::transform(s.begin(), s.end(), std::inserter(ret, ret.begin()),
                 [](auto str) { return std::vector<char>(str.begin(), str.end()); });
  return ret;
}
/// \brief Transform set of chars to strings.
///
/// \param[in] c Set of chars.
///
/// \return Set of strings.
inline std::set<std::string> SetCharToString(const std::set<std::vector<char>> &c) {
  std::set<std::string> ret;
  std::transform(c.begin(), c.end(), std::inserter(ret, ret.begin()),
                 [](auto ch) { return std::string(ch.begin(), ch.end()); });
  return ret;
}
/// \brief Transform map of strings to chars.
///
/// \param[in] s Map of strings.
///
/// \return Map of vector of chars.
template <class T>
inline std::map<std::vector<char>, T> MapStringToChar(const std::map<std::string, T> &s) {
  std::map<std::vector<char>, T> ret;
  std::transform(s.begin(), s.end(), std::inserter(ret, ret.begin()), [](auto str) {
    return std::pair<std::vector<char>, T>(std::vector<char>(str.first.begin(), str.first.end()), str.second);
  });
  return ret;
}
/// \brief Transform map of chars to strings.
///
/// \param[in] c Map of vector of chars.
///
/// \return Map of strings.
template <class T>
inline std::map<std::string, T> MapCharToString(const std::map<std::vector<char>, T> &c) {
  std::map<std::string, T> ret;
  std::transform(c.begin(), c.end(), std::inserter(ret, ret.begin()), [](auto ch) {
    return std::pair<std::string, T>(std::string(ch.first.begin(), ch.first.end()), ch.second);
  });
  return ret;
}
/// \brief Transform unordered map of strings to chars.
///
/// \param[in] s Unordered_map of strings.
///
/// \return Map of vector of chars.
inline std::map<std::vector<char>, std::vector<char>> UnorderedMapStringToChar(
  const std::unordered_map<std::string, std::string> &s) {
  std::map<std::vector<char>, std::vector<char>> ret;
  std::transform(s.begin(), s.end(), std::inserter(ret, ret.begin()), [](auto str) {
    return std::pair<std::vector<char>, std::vector<char>>(std::vector<char>(str.first.begin(), str.first.end()),
                                                           std::vector<char>(str.second.begin(), str.second.end()));
  });
  return ret;
}
/// \brief Transform unordered map of chars to strings.
///
/// \param[in] c Map of vector of chars.
///
/// \return Unordered map of strings.
inline std::unordered_map<std::string, std::string> UnorderedMapCharToString(
  const std::map<std::vector<char>, std::vector<char>> &c) {
  std::unordered_map<std::string, std::string> ret;
  std::transform(c.begin(), c.end(), std::inserter(ret, ret.begin()), [](auto ch) {
    return std::pair<std::string, std::string>(std::string(ch.first.begin(), ch.first.end()),
                                               std::string(ch.second.begin(), ch.second.end()));
  });
  return ret;
}
/// \brief Transform map of strings to vector of chars.
///
/// \param[in] s Map of vector of strings.
///
/// \return Map of vector of chars.
inline std::map<std::vector<char>, std::vector<char>> MapStringToVectorChar(
  const std::map<std::string, std::string> &s) {
  std::map<std::vector<char>, std::vector<char>> ret;
  std::transform(s.begin(), s.end(), std::inserter(ret, ret.begin()), [](auto str) {
    return std::pair<std::vector<char>, std::vector<char>>(std::vector<char>(str.first.begin(), str.first.end()),
                                                           std::vector<char>(str.second.begin(), str.second.end()));
  });
  return ret;
}
/// \brief Transform map of chars to strings.
///
/// \param[in] c Map of vector of chars.
///
/// \return Map of strings.
inline std::map<std::string, std::string> MapVectorCharToString(
  const std::map<std::vector<char>, std::vector<char>> &c) {
  std::map<std::string, std::string> ret;
  std::transform(c.begin(), c.end(), std::inserter(ret, ret.begin()), [](auto ch) {
    return std::pair<std::string, std::string>(std::string(ch.first.begin(), ch.first.end()),
                                               std::string(ch.second.begin(), ch.second.end()));
  });
  return ret;
}
/// \brief Transform class index string to char.
///
/// \param[in] s Vector of pair of strings.
///
/// \return Vector of pair of vector of chars.
inline std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> ClassIndexStringToChar(
  const std::vector<std::pair<std::string, std::vector<int32_t>>> &s) {
  std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> ret;
  std::transform(s.begin(), s.end(), std::back_inserter(ret), [](auto str) {
    return std::pair<std::vector<char>, std::vector<int32_t>>(std::vector<char>(str.first.begin(), str.first.end()),
                                                              str.second);
  });
  return ret;
}
/// \brief Transform class index char to string.
///
/// \param[in] c Vector of pair of schar.
///
/// \return Vector of pair of vector of strings.
inline std::vector<std::pair<std::string, std::vector<int32_t>>> ClassIndexCharToString(
  const std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> &c) {
  std::vector<std::pair<std::string, std::vector<int32_t>>> ret;
  std::transform(c.begin(), c.end(), std::back_inserter(ret), [](auto ch) {
    return std::pair<std::string, std::vector<int32_t>>(std::string(ch.first.begin(), ch.first.end()), ch.second);
  });
  return ret;
}
/// \brief Transform pair string of int64 to pair char of int64.
///
/// \param[in] s Vector of pair of strings.
///
/// \return Vector of pair of vector of chars.
inline std::vector<std::pair<std::vector<char>, int64_t>> PairStringInt64ToPairCharInt64(
  const std::vector<std::pair<std::string, int64_t>> &s) {
  std::vector<std::pair<std::vector<char>, int64_t>> ret;
  std::transform(s.begin(), s.end(), std::back_inserter(ret), [](auto str) {
    return std::pair<std::vector<char>, int64_t>(std::vector<char>(str.first.begin(), str.first.end()), str.second);
  });
  return ret;
}
/// \brief Transform tensor map of char to string.
///
/// \param[in] c Map of Vector of chars.
///
/// \param[in] s Unordered map of strings, which will be changed after the function.
template <class T>
inline void TensorMapCharToString(const std::map<std::vector<char>, T> *c, std::unordered_map<std::string, T> *s) {
  if (c == nullptr || s == nullptr) {
    return;
  }
  for (auto ch : *c) {
    auto key = std::string(ch.first.begin(), ch.first.end());
    auto val = ch.second;
    s->insert(std::pair<std::string, T>(key, val));
  }
}
/// \brief Transform map of map of chars to map of strings.
///
/// \param[in] c Map of map of chars.
///
/// \return Map of strings.
inline std::map<std::string, std::map<std::string, std::string>> MapMapCharToString(
  const std::map<std::vector<char>, std::map<std::vector<char>, std::vector<char>>> &c) {
  std::map<std::string, std::map<std::string, std::string>> ret;
  for (auto iter : c) {
    std::map<std::string, std::string> item;
    std::transform(iter.second.begin(), iter.second.end(), std::inserter(item, item.begin()), [](auto ch) {
      return std::pair<std::string, std::string>(std::string(ch.first.begin(), ch.first.end()),
                                                 std::string(ch.second.begin(), ch.second.end()));
    });
    ret[CharToString(iter.first)] = item;
  }
  return ret;
}
/// \brief Transform map of map of strings to map of chars.
///
/// \param[in] s Map of map of strings.
///
/// \return Map of chars.
inline std::map<std::vector<char>, std::map<std::vector<char>, std::vector<char>>> MapMapStringToChar(
  const std::map<std::string, std::map<std::string, std::string>> &s) {
  std::map<std::vector<char>, std::map<std::vector<char>, std::vector<char>>> ret;
  for (auto iter : s) {
    std::map<std::vector<char>, std::vector<char>> item;
    std::transform(iter.second.begin(), iter.second.end(), std::inserter(item, item.begin()), [](auto ch) {
      return std::pair<std::vector<char>, std::vector<char>>(StringToChar(ch.first), StringToChar(ch.second));
    });
    ret[StringToChar(iter.first)] = item;
  }
  return ret;
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_DUAL_ABI_HELPER_H_
