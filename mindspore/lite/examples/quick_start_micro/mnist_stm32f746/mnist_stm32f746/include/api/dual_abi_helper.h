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
#ifndef MINDSPORE_INCLUDE_API_DUAL_ABI_HELPER_H_
#define MINDSPORE_INCLUDE_API_DUAL_ABI_HELPER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mindspore {
inline std::vector<char> StringToChar(const std::string &s) { return std::vector<char>(s.begin(), s.end()); }

inline std::string CharToString(const std::vector<char> &c) { return std::string(c.begin(), c.end()); }

inline std::optional<std::vector<char>> OptionalStringToChar(const std::optional<std::string> &s) {
  if (s == std::nullopt) return std::nullopt;
  std::optional<std::vector<char>> ret = std::vector<char>(s->begin(), s->end());
  return ret;
}

inline std::optional<std::string> OptionalCharToString(const std::optional<std::vector<char>> &c) {
  if (c == std::nullopt) return std::nullopt;
  std::optional<std::string> ret = std::string(c->begin(), c->end());
  return ret;
}

inline std::pair<std::vector<char>, int32_t> PairStringToChar(const std::pair<std::string, int32_t> &s) {
  return std::pair<std::vector<char>, int32_t>(std::vector<char>(s.first.begin(), s.first.end()), s.second);
}

inline std::pair<std::string, int32_t> PairCharToString(const std::pair<std::vector<char>, int32_t> &c) {
  return std::pair<std::string, int32_t>(std::string(c.first.begin(), c.first.end()), c.second);
}

inline std::vector<std::vector<char>> VectorStringToChar(const std::vector<std::string> &s) {
  std::vector<std::vector<char>> ret;
  std::transform(s.begin(), s.end(), std::back_inserter(ret),
                 [](auto str) { return std::vector<char>(str.begin(), str.end()); });
  return ret;
}

inline std::vector<std::string> VectorCharToString(const std::vector<std::vector<char>> &c) {
  std::vector<std::string> ret;
  std::transform(c.begin(), c.end(), std::back_inserter(ret),
                 [](auto ch) { return std::string(ch.begin(), ch.end()); });
  return ret;
}

inline std::set<std::vector<char>> SetStringToChar(const std::set<std::string> &s) {
  std::set<std::vector<char>> ret;
  std::transform(s.begin(), s.end(), std::inserter(ret, ret.begin()),
                 [](auto str) { return std::vector<char>(str.begin(), str.end()); });
  return ret;
}

inline std::set<std::string> SetCharToString(const std::set<std::vector<char>> &c) {
  std::set<std::string> ret;
  std::transform(c.begin(), c.end(), std::inserter(ret, ret.begin()),
                 [](auto ch) { return std::string(ch.begin(), ch.end()); });
  return ret;
}

inline std::map<std::vector<char>, int32_t> MapStringToChar(const std::map<std::string, int32_t> &s) {
  std::map<std::vector<char>, int32_t> ret;
  std::transform(s.begin(), s.end(), std::inserter(ret, ret.begin()), [](auto str) {
    return std::pair<std::vector<char>, int32_t>(std::vector<char>(str.first.begin(), str.first.end()), str.second);
  });
  return ret;
}

inline std::map<std::string, int32_t> MapCharToString(const std::map<std::vector<char>, int32_t> &c) {
  std::map<std::string, int32_t> ret;
  std::transform(c.begin(), c.end(), std::inserter(ret, ret.begin()), [](auto ch) {
    return std::pair<std::string, int32_t>(std::string(ch.first.begin(), ch.first.end()), ch.second);
  });
  return ret;
}

inline std::map<std::vector<char>, std::vector<char>> UnorderedMapStringToChar(
  const std::unordered_map<std::string, std::string> &s) {
  std::map<std::vector<char>, std::vector<char>> ret;
  std::transform(s.begin(), s.end(), std::inserter(ret, ret.begin()), [](auto str) {
    return std::pair<std::vector<char>, std::vector<char>>(std::vector<char>(str.first.begin(), str.first.end()),
                                                           std::vector<char>(str.second.begin(), str.second.end()));
  });
  return ret;
}

inline std::unordered_map<std::string, std::string> UnorderedMapCharToString(
  const std::map<std::vector<char>, std::vector<char>> &c) {
  std::unordered_map<std::string, std::string> ret;
  std::transform(c.begin(), c.end(), std::inserter(ret, ret.begin()), [](auto ch) {
    return std::pair<std::string, std::string>(std::string(ch.first.begin(), ch.first.end()),
                                               std::string(ch.second.begin(), ch.second.end()));
  });
  return ret;
}

inline std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> ClassIndexStringToChar(
  const std::vector<std::pair<std::string, std::vector<int32_t>>> &s) {
  std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> ret;
  std::transform(s.begin(), s.end(), std::back_inserter(ret), [](auto str) {
    return std::pair<std::vector<char>, std::vector<int32_t>>(std::vector<char>(str.first.begin(), str.first.end()),
                                                              str.second);
  });
  return ret;
}

inline std::vector<std::pair<std::string, std::vector<int32_t>>> ClassIndexCharToString(
  const std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> &c) {
  std::vector<std::pair<std::string, std::vector<int32_t>>> ret;
  std::transform(c.begin(), c.end(), std::back_inserter(ret), [](auto ch) {
    return std::pair<std::string, std::vector<int32_t>>(std::string(ch.first.begin(), ch.first.end()), ch.second);
  });
  return ret;
}

template <class T>
inline std::map<std::vector<char>, T> PadInfoStringToChar(const std::map<std::string, T> &s_pad_info) {
  std::map<std::vector<char>, T> ret;
  std::transform(s_pad_info.begin(), s_pad_info.end(), std::inserter(ret, ret.begin()), [](auto str) {
    return std::pair<std::vector<char>, T>(std::vector<char>(str.first.begin(), str.first.end()), str.second);
  });
  return ret;
}

template <class T>
inline std::map<std::string, T> PadInfoCharToString(const std::map<std::vector<char>, T> &c_pad_info) {
  std::map<std::string, T> ret;
  std::transform(c_pad_info.begin(), c_pad_info.end(), std::inserter(ret, ret.begin()), [](auto ch) {
    return std::pair<std::string, T>(std::string(ch.first.begin(), ch.first.end()), ch.second);
  });
  return ret;
}

template <class T>
inline void TensorMapCharToString(const std::map<std::vector<char>, T> *c, std::unordered_map<std::string, T> *s) {
  for (auto ch : *c) {
    auto key = std::string(ch.first.begin(), ch.first.end());
    auto val = ch.second;
    s->insert(std::pair<std::string, T>(key, val));
  }
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_DUAL_ABI_HELPER_H_
