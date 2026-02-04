/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_DETECTOR_H
#define MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_DETECTOR_H

#include <atomic>
#include <chrono>
#include <deque>
#include <optional>
#include <string>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include "ir/tensor.h"

namespace mindspore {
namespace silentdetect {

void SilentDetect(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr);

struct StatData {
  double avg = 0.0;
  double pre_value = 0.0;
  int count = 0;
  int none_zero_count = 0;
};

struct StrikeRecord {
  std::chrono::system_clock::time_point timestamp;
  std::string name;
  double value;
  StatData stat;
};

class SilentDetector {
 public:
  static SilentDetector &GetInstance() {
    static SilentDetector instance;
    return instance;
  }

  ~SilentDetector();

  SilentDetector(const SilentDetector &) = delete;
  SilentDetector &operator=(const SilentDetector &) = delete;
  SilentDetector(SilentDetector &&) = delete;
  SilentDetector &operator=(SilentDetector &&) = delete;
  std::optional<StrikeRecord> CheckValue(const string &name, double value);
  std::optional<StrikeRecord> CheckValueWithCoolDown(const string &name, double value, std::chrono::minutes cooldown);
  friend void SilentDetect(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr);
  static void Stop();

 private:
  SilentDetector();

#if defined(__linux__) && defined(WITH_BACKEND)
  void ProcessStrike(const StrikeRecord &record);
  void DetectStrikeout();
  void DoCheckSum();
  void ResetTcpStore();
  bool PutTcpStore(const std::string &key, const std::string &value);
  std::string GetTcpStore(const std::string &key);
  void AddTcpStore(const std::string &key, int64_t value);
#endif
  void StopStrikeoutDetector();

  // feature value detection
  std::unordered_map<std::string, StatData> check_status_;
  std::chrono::system_clock::time_point prev_strike_time_;
  std::chrono::minutes cooldown_;  // feature value abnormal cooldown and CheckSum running time
  // strikeout with checksum
  uint32_t rank_id_;
  uint32_t rank_size_;
  bool checksum_enable_;
  bool checksum_result_;
  std::deque<std::chrono::system_clock::time_point> feat_value_strikes_;
  std::shared_mutex feat_value_strikes_mutex_;
  uint32_t strikes_num_;
  std::chrono::minutes strikes_window_;
  std::chrono::system_clock::time_point prev_checksum_time_;
  std::chrono::minutes checksum_cooldown_;
  std::thread strikeout_detector_;
  std::atomic<bool> strikeout_detector_running_;
  static std::atomic<bool> instantiated_;
};

}  // namespace silentdetect
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_SILENT_DETECTOR_H
