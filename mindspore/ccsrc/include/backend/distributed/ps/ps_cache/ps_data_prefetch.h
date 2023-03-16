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
#ifndef MINDSPORE_CCSRC_PS_PS_CACHE_PS_DATA_PS_DATA_PREFETCH_H_
#define MINDSPORE_CCSRC_PS_PS_CACHE_PS_DATA_PS_DATA_PREFETCH_H_

#include <map>
#include <string>
#include <memory>
#include <atomic>
#include <condition_variable>
#include "include/backend/distributed/ps/ps_cache/ps_data_channel.h"

#define EXPORT __attribute__((visibility("default")))

namespace mindspore {
namespace ps {
class EXPORT PsDataPrefetch {
 public:
  EXPORT static PsDataPrefetch &GetInstance();

  EXPORT bool cache_enable() const { return cache_enable_; }
  EXPORT void set_cache_enable(bool cache_enable) { cache_enable_ = cache_enable; }
  EXPORT void CreateDataChannel(const std::string &channel_name, size_t step_num);
  EXPORT bool PrefetchData(const std::string &channel_name, void *data, const size_t data_size,
                           const std::string &data_type);
  EXPORT bool FinalizeData(const std::string &channel_name);
  EXPORT void NotifyFinalize();
  EXPORT bool QueryData(const std::string &channel_name, void **data_ptr) const;
  EXPORT size_t data_size(const std::string &channel_name) const;
  EXPORT bool TryWakeChannel(const std::string &channel_name) const;

 private:
  PsDataPrefetch() : cache_enable_(false), data_ready_(false) {}
  virtual ~PsDataPrefetch() = default;
  PsDataPrefetch(const PsDataPrefetch &) = delete;
  PsDataPrefetch &operator=(const PsDataPrefetch &) = delete;
  std::shared_ptr<PsDataChannel> ps_data_channel(const std::string &channel_name) const;
  void WakeAllChannel() const;
  std::map<std::string, std::shared_ptr<PsDataChannel>> ps_data_channel_map_;
  bool cache_enable_;
  bool data_ready_;
  std::mutex data_mutex_;
  std::condition_variable data_prefetch_;
  std::condition_variable data_process_;
  std::atomic_bool need_wait_{true};
  std::atomic_bool invalid_data_type_{false};
  // Ensure that the Finalize function is multithreaded safe.
  std::mutex finalize_mutex_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_PS_DATA_PS_DATA_PREFETCH_H_
