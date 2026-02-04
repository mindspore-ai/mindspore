/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#include "tools/data_dump/debugger/tensor_summary.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include "base/float16.h"
#include "openssl/md5.h"
#include "openssl/sha.h"

namespace mindspore {

MeanCalculator::MeanCalculator() : mean(0.0), count(0) {}

void MeanCalculator::ProcessElement(double value) {
  count += 1;
  mean += value;
}

double MeanCalculator::GetMean() const { return mean / count; }

void L2Calculator::ProcessElement(double value) { squre_sum += value * value; }

void L2Calculator::ProcessElement(const L2Calculator &other) { this->squre_sum += other.squre_sum; }

double L2Calculator::GetL2Value() const { return std::sqrt(squre_sum); }

HashThreadPool::HashThreadPool() : running_(true) {
  uint64_t num_logical_processors = std::thread::hardware_concurrency();
  if (num_logical_processors == 0) {
    MS_LOG(ERROR) << "Dump Hash value: cannot get num_logical_processors";
  } else {
    MS_VLOG(VL_DUMP) << "Dump Hash value: num_logical_processors:" << num_logical_processors;
  }
  uint64_t thread_num = std::max<uint64_t>(num_logical_processors / 8, 2) - 1;  // set device_num to 8
  MS_VLOG(VL_DUMP) << "Dump Hash value: thread pool size:" << thread_num;
  for (uint64_t i = 0; i < thread_num; ++i) {
    threads_.emplace_back([this] {
      while (running_) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->mutex_);
          this->condition_.wait(lock, [this] { return !this->running_ || !this->tasks_.empty(); });
          if (!this->running_ && this->tasks_.empty()) {
            return;
          }
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }
        task();
      }
    });
  }
}

HashThreadPool::~HashThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
  }
  condition_.notify_all();
  for (std::thread &thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

// add task to ThreadPool
template <typename T>
auto HashThreadPool::add_task(TensorSummary<T> *single_thread_task, bool calc_hash) {
  MS_EXCEPTION_IF_CHECK_FAIL(running_, "Hash threadPool is not running.");
  auto task = std::make_shared<std::packaged_task<void()>>(
    std::bind(&TensorSummary<T>::TensorStatisticsSingleThread, single_thread_task, calc_hash));
  std::future<void> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace([task]() { (*task)(); });
  }
  condition_.notify_one();
  return res;
}

template <typename T>
TensorSummary<T>::TensorSummary(const void *current_tensor_ptr, const void *const previous_tensor_ptr,
                                uint64_t num_elements, uint64_t prev_num_elements)
    : current_tensor_ptr_(static_cast<const T *>(current_tensor_ptr)),
      prev_tensor_ptr_(static_cast<const T *>(previous_tensor_ptr)),
      num_elements_(num_elements),
      prev_num_elements_(prev_num_elements),
      min_(std::numeric_limits<double>::infinity()),
      max_(-std::numeric_limits<double>::infinity()),
      avg_(0.0),
      is_bool_(false),
      neg_zero_count_(0),
      pos_zero_count_(0),
      pos_inf_count_(0),
      neg_inf_count_(0),
      inf_count_(0),
      nan_count_(0),
      zero_count_(0),
      sha1_("") {}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Calculates statistics on chunks of data.
 */
template <typename T>
void TensorSummary<T>::TensorStatistics(DbgDataType dtype_value, bool calc_hash) {
  if (dtype_value == DT_BOOL) {
    is_bool_ = true;
  }
  const uint64_t default_elements_per_slice = 20480;
  if (num_elements_ <= default_elements_per_slice) {
    return TensorStatisticsSingleThread(calc_hash);
  }

  // Use multithread to calculate statistic on chunks of data
  uint64_t slice_num = ((num_elements_ - 1) / default_elements_per_slice) + 1;
  void *previous_tensor_ptr = nullptr;
  size_t offset = 0;
  std::vector<std::unique_ptr<TensorSummary<T>>> summary_vec;
  std::vector<std::future<void>> summary_future_vec;
  HashThreadPool &pool = HashThreadPool::GetInstance();
  for (uint64_t i = 0; i < slice_num; i++) {
    uint64_t num_elements_for_slice;
    if (i == slice_num - 1) {
      num_elements_for_slice = num_elements_ - offset;
    } else {
      num_elements_for_slice = default_elements_per_slice;
    }
    (void)summary_vec.emplace_back(
      std::make_unique<TensorSummary<T>>(current_tensor_ptr_ + offset, previous_tensor_ptr, num_elements_for_slice, 0));
    (void)summary_future_vec.emplace_back(pool.add_task(summary_vec[i].get(), calc_hash));
    offset += num_elements_for_slice;
  }

  // Aggregate results of all chunks
  num_elements_ = 0;  // Let current tensor weight 0 in the aggregation
  double sum = 0;
  std::string sha1_combine_ = "";
  for (unsigned int i = 0; i < summary_future_vec.size(); i++) {
    summary_future_vec[i].wait();
    summary_future_vec[i].get();
    auto &cur_summary = *(summary_vec[i]);
    num_elements_ += cur_summary.num_elements_;
    min_ = std::isnan(cur_summary.min_) ? cur_summary.min_ : std::min(min_, cur_summary.min_);
    max_ = std::isnan(cur_summary.max_) ? cur_summary.max_ : std::max(max_, cur_summary.max_);
    sum += cur_summary.avg_ * cur_summary.num_elements_;
    neg_zero_count_ += cur_summary.neg_zero_count_;
    pos_zero_count_ += cur_summary.pos_zero_count_;
    neg_inf_count_ += cur_summary.neg_inf_count_;
    pos_inf_count_ += cur_summary.pos_inf_count_;
    inf_count_ += cur_summary.inf_count_;
    nan_count_ += cur_summary.nan_count_;
    zero_count_ += cur_summary.zero_count_;
    l2_calc_.ProcessElement(cur_summary.l2_calc_);
    sha1_combine_ += cur_summary.sha1_;
  }
  avg_ = sum / num_elements_;
  if (calc_hash) {
    char *sha1_value = sha1_combine_.data();
    TensorHashValue("sha1", reinterpret_cast<unsigned char *>(sha1_value), sha1_combine_.size(), &sha1_);
  }
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Process all the elements of the chunked data and calculates the statistics.
 */
template <typename T>
void TensorSummary<T>::TensorStatisticsSingleThread(bool calc_hash) {
  MeanCalculator mean_calc = MeanCalculator();
  for (size_t i = 0; i < num_elements_; ++i) {
    auto current_value = static_cast<double>(current_tensor_ptr_[i]);
    l2_calc_.ProcessElement(current_value);
    if (std::isnan(current_value)) {
      nan_count_ += 1;
      max_ = current_value;
      min_ = current_value;
      mean_calc.ProcessElement(current_value);
      continue;
    }
    if (std::isinf(current_value)) {
      if (current_value > 0) {
        pos_inf_count_ += 1;
      } else {
        neg_inf_count_ += 1;
      }
    }
    if (current_value == 0.0) {
      zero_count_ += 1;
    }
    // only considering tensor elements with value
    if (std::signbit(current_value) && !(current_value == 0.0)) {
      neg_zero_count_ += 1;
    } else if (!(current_value == 0.0)) {
      pos_zero_count_ += 1;
    }
    max_ = std::max(max_, current_value);
    min_ = std::min(min_, current_value);
    mean_calc.ProcessElement(current_value);
  }
  avg_ = mean_calc.GetMean();
  if (calc_hash) {
    TensorHashValue("sha1", reinterpret_cast<unsigned char *>(const_cast<T *>(current_tensor_ptr_)),
                    num_elements_ * sizeof(T), &sha1_);
  }
}

const std::unordered_map<std::string, unsigned char *(*)(const unsigned char *, size_t, unsigned char *)>
  hash_func_map = {{"md5", MD5}, {"sha1", SHA1}};
const std::unordered_map<std::string, int> hash_digest_len_map = {{"md5", MD5_DIGEST_LENGTH},
                                                                  {"sha1", SHA_DIGEST_LENGTH}};
void TensorHashValue(std::string hash_type, const unsigned char *data, size_t len, std::string *output) {
  MS_EXCEPTION_IF_NULL(data);
  int hash_bit_wide = 2;
  int length = hash_digest_len_map.at(hash_type);
  unsigned char digest[length];
  hash_func_map.at(hash_type)(data, len, digest);
  std::stringstream ss;
  for (int i = 0; i < length; i++) {
    ss << std::hex << std::setw(hash_bit_wide) << std::setfill('0') << static_cast<int>(digest[i]);
  }
  *output = ss.str();
}

template class TensorSummary<uint8_t>;
template class TensorSummary<int8_t>;
template class TensorSummary<uint16_t>;
template class TensorSummary<int16_t>;
template class TensorSummary<uint32_t>;
template class TensorSummary<int32_t>;
template class TensorSummary<uint64_t>;
template class TensorSummary<int64_t>;
template class TensorSummary<float16>;
template class TensorSummary<bfloat16>;
template class TensorSummary<float>;
template class TensorSummary<double>;
template class TensorSummary<bool>;
}  // namespace mindspore
