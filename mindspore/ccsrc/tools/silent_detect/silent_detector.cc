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

#include "tools/silent_detect/silent_detector.h"

#include <string>

#include "tools/silent_detect/silent_detect_config_parser.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "tools/silent_detect/checksum/checksum_mgr.h"
#include "include/cluster/topology/cluster_context.h"
#include "include/cluster/topology/compute_graph_node.h"
#else
#include "include/cluster/topology/dummy_cluster_context.h"
#endif
#include "utils/distributed_meta.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace silentdetect {
namespace {
// TCP store keys for CheckSum
const auto kChecksumEnable = "<sdc-checksum>enable";         // global checksum enable flag
const auto kChecksumStopCnt = "<sdc-checksum>stop_cnt";      // num of ranks that stop checksum and report result
const auto kChecksumResult = "<sdc-checksum>result";         // global checksum result
const auto kChecksumResultCnt = "<sdc-checksum>result_cnt";  // num of ranks that get global checksum result

const int kNumber10 = 10;
const int kNumber100 = 100;
const double kFactorAvg = 0.99;

std::optional<double> ScalarTensorToDouble(mindspore::tensor::TensorPtr tensor, const std::string &name) {
  MS_EXCEPTION_IF_NULL(tensor);
  size_t size = SizeOf(tensor->shape());
  MS_EXCEPTION_IF_CHECK_FAIL(size == 1, "For silent detect feature value, there must be only one element, but got " +
                                          std::to_string(size) + ".");
  auto cpu_tensor = tensor->cpu();
  auto data_type = cpu_tensor->data_type();
  auto data = cpu_tensor->data_c();
  switch (data_type) {
    case TypeId::kNumberTypeBFloat16:
      return static_cast<double>(*static_cast<const bfloat16 *>(data));
    case TypeId::kNumberTypeFloat16:
      return static_cast<double>(*static_cast<const float16 *>(data));
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
      return static_cast<double>(*static_cast<const float *>(data));
    case TypeId::kNumberTypeDouble:
    case TypeId::kNumberTypeFloat64:
      return *static_cast<const double *>(data);
    default:
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Skip the unsupported data type: " << TypeIdToString(data_type)
                                      << ", tensor name is " << name;
      return std::nullopt;
  }
}

std::string ToString(const StatData &data) {
  std::stringstream ss;
  ss << "StatData{"
     << "avg: " << data.avg << ", pre_value: " << data.pre_value << ", count: " << data.count
     << ", none_zero_count: " << data.none_zero_count << "}";
  return ss.str();
}

std::string ToString(const StrikeRecord &record) {
  std::stringstream ss;
  ss << "StrikeRecord{"
     << "timestamp: " << std::chrono::system_clock::to_time_t(record.timestamp) << ", name: " << record.name
     << ", value: " << record.value << ", stat: " << ToString(record.stat) << "}";
  return ss.str();
}

uint32_t GetRankID() {
  uint32_t rank_id = 0;
  if (mindspore::DistributedMeta::GetInstance()->initialized()) {
    rank_id = mindspore::DistributedMeta::GetInstance()->global_rank_id();
  }
  return rank_id;
}

uint32_t GetRankSize() {
  uint32_t rank_size = 1;
  if (mindspore::DistributedMeta::GetInstance()->initialized()) {
    rank_size = mindspore::DistributedMeta::GetInstance()->global_rank_size();
  }
  return rank_size;
}
}  // namespace

void SilentDetect(std::string name, mindspore::tensor::TensorPtr tensor) {
  auto res = ScalarTensorToDouble(tensor, name);
  if (!res.has_value()) {
    return;
  }
  auto current_val = res.value();
  auto cooldown = SilentDetector::GetInstance().cooldown_;
  auto strike_record = SilentDetector::GetInstance().CheckValueWithCoolDown(name, current_val, cooldown);
  if (strike_record.has_value()) {
    MS_LOG(WARNING) << "Silent detect strike detected: " << ToString(strike_record.value());
#if defined(__linux__) && defined(WITH_BACKEND)
    if (SilentDetector::GetInstance().strikeout_detector_running_) {
      SilentDetector::GetInstance().ProcessStrike(strike_record.value());
    }
#endif
  }
}

std::atomic<bool> SilentDetector::instantiated_{false};

SilentDetector::SilentDetector() {
  instantiated_ = true;
  rank_id_ = GetRankID();
  rank_size_ = GetRankSize();
  prev_strike_time_ = std::chrono::system_clock::time_point::min();
  prev_checksum_time_ = std::chrono::system_clock::time_point::min();
  cooldown_ = std::chrono::minutes(SilentDetectConfigParser::GetInstance().GetCooldown());
  strikes_num_ = static_cast<uint32_t>(SilentDetectConfigParser::GetInstance().GetStrikesNum());
  strikes_window_ = std::chrono::minutes(SilentDetectConfigParser::GetInstance().GetStrikesWindow());
  checksum_cooldown_ = std::chrono::minutes(SilentDetectConfigParser::GetInstance().GetChecksumCooldown());
  checksum_enable_ = false;
  checksum_result_ = false;
#if defined(__linux__) && defined(WITH_BACKEND)
  if (SilentDetectConfigParser::GetInstance().IsEnable() && SilentDetectConfigParser::GetInstance().IsWithChecksum()) {
    MS_LOG(WARNING) << "Feature value detection works with CheckSum, strikes_num: " << strikes_num_
                    << ", strikes_window: " << strikes_window_.count() << " min, cooldown: " << cooldown_.count()
                    << " min, checksum_cooldown: " << checksum_cooldown_.count() << " min.";
    ResetTcpStore();
    // start a thread to check feature value strikes
    strikeout_detector_running_ = true;
    strikeout_detector_ = std::thread([this]() {
      while (strikeout_detector_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(kNumber10));
        DetectStrikeout();
      }
    });
  }
#endif
}

SilentDetector::~SilentDetector() { StopStrikeoutDetector(); }

void SilentDetector::Stop() {
  if (instantiated_) {
    GetInstance().StopStrikeoutDetector();
  }
}

void SilentDetector::StopStrikeoutDetector() {
  if (strikeout_detector_running_) {
    strikeout_detector_running_ = false;
    if (strikeout_detector_.joinable()) {
      strikeout_detector_.join();
    }
  }
}

std::optional<StrikeRecord> SilentDetector::CheckValueWithCoolDown(const std::string &name, double value,
                                                                   std::chrono::minutes cooldown) {
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Silent detect receives data, name is " << name << ", value is " << value;
  auto &stat = check_status_[name];
  auto strike_record = CheckValue(name, value);
  stat.pre_value = value;
  stat.count += 1;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "After silent detect, name is " << name << ", stat is " << ToString(stat);
  if (!strike_record.has_value()) {
    return std::nullopt;
  }
  auto strike_time = strike_record.value().timestamp;
  if (prev_strike_time_ + cooldown < strike_time) {
    prev_strike_time_ = strike_time;
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Strike happened. record is " << ToString(strike_record.value());
    return strike_record;
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Strike will not be recorded repeatedly during the " << cooldown.count()
                                  << " mins cooling-off period, and the exception info is "
                                  << ToString(strike_record.value());
  return std::nullopt;
}

std::optional<StrikeRecord> SilentDetector::CheckValue(const std::string &name, double value) {
  static const int alpha1 = SilentDetectConfigParser::GetInstance().GetUpperThresh1();
  static const int alpha2 = SilentDetectConfigParser::GetInstance().GetUpperThresh2();

  auto &stat = check_status_[name];

  if (value == 0.0) {
    return std::nullopt;
  }

  if (std::isnan(value) || std::isinf(value)) {
    StrikeRecord record{std::chrono::system_clock::now(), name, value, stat};
    return record;
  }

  double thres = value;
  double thres2 = value;
  if (stat.none_zero_count >= kNumber10 && stat.avg != 0.0) {
    const double factor = 1.0 - std::pow(kFactorAvg, stat.none_zero_count);
    thres = stat.avg * static_cast<double>(alpha1) / factor;
    thres2 = stat.avg * static_cast<double>(alpha2) / factor;
  }

  if (value > thres && std::abs(value - stat.pre_value) > thres) {
    StrikeRecord record{std::chrono::system_clock::now(), name, value, stat};
    return record;
  }

  if (value <= thres2) {
    stat.none_zero_count += 1;
    stat.avg = stat.avg * kFactorAvg + value * (1.0 - kFactorAvg);
  }
  return std::nullopt;
}

#if defined(__linux__) && defined(WITH_BACKEND)
void SilentDetector::ProcessStrike(const StrikeRecord &record) {
  if (checksum::CheckSumMgr::GetInstance().IsCheckSumEnable()) {
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Feature value strikes will not be counted during CheckSum.";
    return;
  }
  std::unique_lock<std::shared_mutex> lock(feat_value_strikes_mutex_);
  auto now = record.timestamp;
  feat_value_strikes_.push_back(record.timestamp);
  MS_LOG(WARNING) << "Detect feature value strike, count: " << feat_value_strikes_.size()
                  << ", number to strikeout: " << strikes_num_;
  // remove expired records
  while (feat_value_strikes_.size() > strikes_num_ && now - feat_value_strikes_.front() > strikes_window_) {
    feat_value_strikes_.pop_front();
  }
}

void SilentDetector::ResetTcpStore() {
  // last stage is the first one sampling feature value
  if (rank_id_ != rank_size_ - 1 || rank_size_ <= 1) {
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(PutTcpStore(kChecksumEnable, "false"),
                             std::string("SilentDetector puts TCPStore key '") + kChecksumEnable + "' failed.");
  MS_EXCEPTION_IF_CHECK_FAIL(PutTcpStore(kChecksumStopCnt, "0"),
                             std::string("SilentDetector puts TCPStore key '") + kChecksumStopCnt + "' failed.");
  MS_EXCEPTION_IF_CHECK_FAIL(PutTcpStore(kChecksumResult, "false"),
                             std::string("SilentDetector puts TCPStore key '") + kChecksumResult + "' failed.");
  MS_EXCEPTION_IF_CHECK_FAIL(PutTcpStore(kChecksumResultCnt, "0"),
                             std::string("SilentDetector puts TCPStore key '") + kChecksumResultCnt + "' failed.");
  return;
}

bool SilentDetector::PutTcpStore(const std::string &key, const std::string &value) {
  if (!strikeout_detector_running_ || rank_size_ <= 1) {
    return true;
  }
  auto node = mindspore::distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  auto cgn = std::dynamic_pointer_cast<mindspore::distributed::cluster::topology::ComputeGraphNode>(node);
  MS_EXCEPTION_IF_NULL(cgn);
  return cgn->PutMetadata(key, value);
}

std::string SilentDetector::GetTcpStore(const std::string &key) {
  if (!strikeout_detector_running_) {
    return "";
  }
  if (rank_size_ <= 1) {
    if (key == kChecksumEnable) {
      return checksum_enable_ ? "true" : "false";
    }
    if (key == kChecksumResult) {
      return checksum_result_ ? "true" : "false";
    }
    if (key == kChecksumStopCnt || key == kChecksumResultCnt) {
      return std::to_string(rank_size_);
    }
    return "";
  }
  auto node = mindspore::distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  auto cgn = std::dynamic_pointer_cast<mindspore::distributed::cluster::topology::ComputeGraphNode>(node);
  MS_EXCEPTION_IF_NULL(cgn);
  return cgn->GetMetadata(key);
}

void SilentDetector::AddTcpStore(const std::string &key, int64_t value) {
  if (!strikeout_detector_running_ || rank_size_ <= 1) {
    return;
  }
  auto node = mindspore::distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  auto cgn = std::dynamic_pointer_cast<mindspore::distributed::cluster::topology::ComputeGraphNode>(node);
  MS_EXCEPTION_IF_NULL(cgn);
  cgn->AddMetadata(key, value);
}

// do CheckSum for cooldown time and get global result
void SilentDetector::DoCheckSum() {
  checksum::CheckSumMgr::GetInstance().CheckSumStart();
  std::this_thread::sleep_for(cooldown_);
  checksum::CheckSumMgr::GetInstance().CheckSumStop();
  checksum_enable_ = false;
  {
    std::unique_lock<std::shared_mutex> lock(feat_value_strikes_mutex_);
    feat_value_strikes_.clear();
  }
  // report globally if CheckSum detect error on current rank
  bool result = checksum::CheckSumMgr::GetInstance().GetCheckSumResult();
  if (result) {
    MS_LOG(WARNING) << "CheckSum detects MatMul error on rank " << rank_id_;
    PutTcpStore(kChecksumResult, "true");
  }
  AddTcpStore(kChecksumStopCnt, 1);
  // wait all ranks stop CheckSum and report result
  int iter = 0;
  static const int max_iter = 600;
  while (strikeout_detector_running_ && GetTcpStore(kChecksumStopCnt) != std::to_string(rank_size_) &&
         iter < max_iter) {
    ++iter;
    std::this_thread::sleep_for(std::chrono::milliseconds(kNumber100));
  }
  result = GetTcpStore(kChecksumResult) == "true";
  MS_LOG(WARNING) << "Global CheckSum result is " << result;
  if (result) {
    MS_LOG(WARNING) << "SilentCheck detects SDC error, which means training may be unstable. "
                       "Search 'CheckSum detects MatMul error on rank' in training logs to get abnormal ranks.";
  }
  AddTcpStore(kChecksumResultCnt, 1);
  // wait all ranks get global result
  iter = 0;
  while (strikeout_detector_running_ && GetTcpStore(kChecksumResultCnt) != std::to_string(rank_size_) &&
         iter < max_iter) {
    ++iter;
    std::this_thread::sleep_for(std::chrono::milliseconds(kNumber100));
  }
  // wait all ranks before resetting TCPStore
  std::this_thread::sleep_for(std::chrono::seconds(kNumber10));
  ResetTcpStore();
}

void SilentDetector::DetectStrikeout() {
  if (!strikeout_detector_running_ || std::chrono::system_clock::now() < prev_checksum_time_ + checksum_cooldown_) {
    return;
  }
  // detect strikeout and broadcast
  {
    std::shared_lock<std::shared_mutex> lock(feat_value_strikes_mutex_);
    if (feat_value_strikes_.size() >= strikes_num_) {
      checksum_enable_ = true;
      MS_LOG(WARNING) << "Feature value detection strikes out on rank " << rank_id_ << ", strikes_num: " << strikes_num_
                      << ", CheckSum will be enabled globally.";
      PutTcpStore(kChecksumEnable, "true");
    }
  }
  checksum_enable_ = GetTcpStore(kChecksumEnable) == "true";
  // CheckSum if global strikeout
  if (!checksum_enable_) {
    return;
  }
  MS_LOG(WARNING) << "Feature value detection strikes out! "
                  << "Search 'strikes out on rank' in training logs to get abnormal ranks.";
  prev_checksum_time_ = std::chrono::system_clock::now();
  DoCheckSum();
  MS_LOG(WARNING) << "CheckSum will not be enabled again within " << checksum_cooldown_.count()
                  << " min after timestamp " << std::chrono::system_clock::to_time_t(prev_checksum_time_);
}
#endif
}  // namespace silentdetect
}  // namespace mindspore
