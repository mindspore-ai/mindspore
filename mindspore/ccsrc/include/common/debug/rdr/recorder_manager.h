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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_RDR_RECORDER_MANAGER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_RDR_RECORDER_MANAGER_H_
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "utils/hash_map.h"
#include "include/common/debug/rdr/base_recorder.h"
#include "include/common/debug/env_config_parser.h"
#include "include/common/visible.h"

namespace mindspore {
// The number is the reciprocal of the golden ratio.
const unsigned int MAGIC_CONSTANT = 0x9e3779b9;
const unsigned int HASH_SHIFT_LEFT = 6;
const unsigned int HASH_SHIFT_RIGHT = 2;

template <typename T>
inline void hash_combine(std::size_t *seed, const T &val) {
  *seed ^= std::hash<T>()(val) + MAGIC_CONSTANT + (*seed << HASH_SHIFT_LEFT) + (*seed >> HASH_SHIFT_RIGHT);
}

template <typename T1, typename T2>
inline std::size_t hash_seed(const T1 &val1, const T2 &val2) {
  std::size_t seed = 0;
  hash_combine(&seed, val1);
  hash_combine(&seed, val2);
  return seed;
}

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    return hash_seed(p.first, p.second);
  }
};

class COMMON_EXPORT RecorderManager {
 public:
  static RecorderManager &Instance();

  void UpdateRdrEnable();
  bool RdrEnable() const;
  bool CheckRdrMemIsRecord() const;
  void SetRdrMemIsRecord(bool is_enable = true);

  bool RecordObject(const BaseRecorderPtr &recorder);
  BaseRecorderPtr GetRecorder(std::string module, std::string name);
  void TriggerAll();
  void Snapshot();
  void ClearAll();

 private:
  RecorderManager() {}
  ~RecorderManager() {}

  bool rdr_enable_{false};
  int rdr_mode_{Exceptional};
  bool rdr_has_record_mem_{false};

  mutable std::mutex mtx_;
  // <module, name>, BaserRecorderPtr
  mindspore::HashMap<std::pair<std::string, std::string>, BaseRecorderPtr, pair_hash> recorder_container_;
};

namespace RDR {
COMMON_EXPORT void TriggerAll();
COMMON_EXPORT void Snapshot();
COMMON_EXPORT void ResetRecorder();
}  // namespace RDR
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_RECORDER_MANAGER_H_
