/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_OPTIMIZER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_OPTIMIZER_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/utils/thread_pool.h"
namespace mindspore {
namespace kernel {
namespace sparse_optimizer_cpu {
template <typename T>
struct SparseGradient {
  float *value_{nullptr};
  T *indices_{nullptr};
  size_t indices_size_{0};
};

template <typename T>
struct ReduceSparseGradientParam {
  SparseGradient<T> *input_grad_{nullptr};
  SparseGradient<T> *workspace_grad_{nullptr};
  SparseGradient<T> *output_grad_{nullptr};
  size_t max_index_{0};
  size_t value_stride_{0};
  bool use_sort_reduce_{false};
};

template <typename T>
struct MultiThreadComputeParams {
  float *var_{nullptr};
  float *accum_{nullptr};
  float *linear_{nullptr};
  float *m_{nullptr};
  float *m_t_{nullptr};
  float *v_{nullptr};
  float lr_{0};
  float l1_{0};
  float l2_{0};
  float lr_power_{0};
  float beta1_{0};
  float beta2_{0};
  float epsilon_{0};
  SparseGradient<T> sparse_grad_;
  size_t var_first_dim_size_{0};
  size_t var_outer_dim_size_{0};
  bool use_nesterov_;
  bool update_slots_;
};

template <typename T>
using MultiThreadComputeFunc = std::function<void(MultiThreadComputeParams<T> *param, size_t start, size_t end)>;

template <typename T>
struct BucketSparseGradient {
  float *value_;
  T *indices_;
  T *global_indices_;
  size_t indices_size_;
};

template <typename T>
struct MultiThreadReduceSparseGradientParam {
  SparseGradient<T> *input_grad_{nullptr};
  SparseGradient<T> *workspace_grad_{nullptr};
  SparseGradient<T> *output_grad_{nullptr};
  size_t max_index_{0};
  size_t value_stride_{0};
  size_t thread_num_{0};
  bool use_sort_reduce_{false};
};

class SparseOptimizerCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseOptimizerCpuKernelMod() = default;
  ~SparseOptimizerCpuKernelMod() override = default;

  template <typename T>
  static void BucketReduceSparseGradient(const ReduceSparseGradientParam<T> &param) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(param.input_grad_);
    size_t thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
    if (param.input_grad_->indices_size_ < thread_num) {
      thread_num = param.input_grad_->indices_size_;
    }
    MultiThreadReduceSparseGradientParam<T> multi_thread_param(
      {param.input_grad_, param.workspace_grad_, param.output_grad_, param.max_index_, param.value_stride_, thread_num,
       param.use_sort_reduce_});
    std::vector<std::shared_ptr<SparseGradient<T>>> segments;
    std::vector<std::shared_ptr<std::vector<size_t>>> segment_bucket_sizes;
    SplitAndCalculateSegmentBucketSize(multi_thread_param, &segments, &segment_bucket_sizes);

    std::vector<std::shared_ptr<BucketSparseGradient<T>>> buckets;
    GatherSegmentIndicesToOutputBucket(multi_thread_param, segments, segment_bucket_sizes, &buckets);

    std::vector<std::shared_ptr<SparseGradient<T>>> reduced_buckets;
    ReduceBucketSparseGradientToWorkspace(multi_thread_param, buckets, &reduced_buckets);

    MergeReduceSparseGradient(multi_thread_param, reduced_buckets);
    MS_LOG(DEBUG) << "End";
  }

 protected:
  template <typename T>
  void MultiThreadCompute(const MultiThreadComputeFunc<T> &func, MultiThreadComputeParams<T> *params,
                          size_t total_compute_size) const {
    std::vector<common::Task> tasks;
    auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
    tasks.reserve(max_thread_num);
    size_t start = 0;
    size_t once_compute_size = (total_compute_size + max_thread_num - 1) / max_thread_num;
    while (start < total_compute_size) {
      size_t end = (start + once_compute_size) > total_compute_size ? total_compute_size : (start + once_compute_size);
      auto task = [&func, &params, start, end]() {
        func(params, start, end);
        return common::SUCCESS;
      };
      (void)tasks.emplace_back(task);
      start += once_compute_size;
    }
    ParallelLaunch(tasks);
  }

 private:
  template <typename T>
  static void CalculateEachBucketSize(const std::shared_ptr<SparseGradient<T>> &sparse_grad, size_t max_index,
                                      std::vector<size_t> *each_bucket_size) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(sparse_grad);
    MS_EXCEPTION_IF_NULL(sparse_grad->indices_);
    MS_EXCEPTION_IF_NULL(each_bucket_size);
    size_t bucket_num = each_bucket_size->size();
    if (bucket_num < 1) {
      MS_LOG(EXCEPTION) << "For 'SparseOptimizer', the 'bucket_num' must be at least 1, but got " << bucket_num;
    }
    for (size_t i = 0; i < sparse_grad->indices_size_; ++i) {
      T index = sparse_grad->indices_[i];
      if (index >= 0 && LongToSize(index) < max_index) {
        auto bucket_id = index % bucket_num;
        each_bucket_size->at(bucket_id)++;
      }
    }
    MS_LOG(DEBUG) << "End";
  }

  template <typename T>
  static void SplitAndCalculateSegmentBucketSize(
    const MultiThreadReduceSparseGradientParam<T> &param, std::vector<std::shared_ptr<SparseGradient<T>>> *segments_ptr,
    std::vector<std::shared_ptr<std::vector<size_t>>> *segment_bucket_sizes_ptr) {
    MS_EXCEPTION_IF_NULL(param.input_grad_);
    MS_EXCEPTION_IF_NULL(segment_bucket_sizes_ptr);
    MS_EXCEPTION_IF_NULL(segments_ptr);
    auto &segments = *segments_ptr;
    auto &segment_bucket_sizes = *segment_bucket_sizes_ptr;
    auto input_grad = param.input_grad_;
    if (param.thread_num_ < 1) {
      MS_EXCEPTION(ArgumentError) << "Input param thread num must > 0!";
    }
    size_t thread_indices_size = input_grad->indices_size_ / param.thread_num_;
    size_t left_indices_size = input_grad->indices_size_ % param.thread_num_;
    std::vector<common::Task> tasks;
    tasks.reserve(param.thread_num_);
    segments.reserve(param.thread_num_);

    size_t current_indices_offset = 0;
    for (size_t i = 0; i < param.thread_num_; ++i) {
      (void)segment_bucket_sizes.emplace_back(std::make_shared<std::vector<size_t>>(param.thread_num_, 0));
      size_t indices_size = thread_indices_size;
      if (i < left_indices_size) {
        indices_size += 1;
      }
      (void)segments.emplace_back(std::make_shared<SparseGradient<T>>());
      segments[i]->value_ = input_grad->value_ + current_indices_offset * param.value_stride_;
      segments[i]->indices_ = input_grad->indices_ + current_indices_offset;
      segments[i]->indices_size_ = indices_size;
      auto task = [&segments, &param, &segment_bucket_sizes, i]() {
        CalculateEachBucketSize<T>(segments[i], param.max_index_, segment_bucket_sizes[i].get());
        return common::SUCCESS;
      };
      (void)tasks.emplace_back(task);
      current_indices_offset += indices_size;
    }
    ParallelLaunch(tasks);
  }

  template <typename T>
  static void CopySegmentIndicesToBucket(const MultiThreadReduceSparseGradientParam<T> &param,
                                         const std::shared_ptr<SparseGradient<T>> &segment, size_t bucket_offset,
                                         const std::vector<std::shared_ptr<BucketSparseGradient<T>>> &buckets) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(segment);
    MS_EXCEPTION_IF_NULL(segment->indices_);
    if (param.thread_num_ == 0) {
      MS_EXCEPTION(ArgumentError) << "Input param thread num must > 0!";
    }
    std::vector<size_t> bucket_data_num(param.thread_num_, 0);
    for (size_t i = 0; i < segment->indices_size_; ++i) {
      T index = segment->indices_[i];
      if (index >= 0 && LongToSize(index) < param.max_index_) {
        auto bucket_id = index % param.thread_num_;
        auto bucket_index = bucket_data_num[bucket_id];
        buckets[bucket_id]->indices_[bucket_index] = index;
        buckets[bucket_id]->global_indices_[bucket_index] = bucket_offset + i;
        bucket_data_num[bucket_id]++;
      }
    }
    MS_LOG(DEBUG) << "End";
  }

  template <typename T>
  static void GatherSegmentIndicesToOutputBucket(
    const MultiThreadReduceSparseGradientParam<T> &param,
    const std::vector<std::shared_ptr<SparseGradient<T>>> &segments,
    const std::vector<std::shared_ptr<std::vector<size_t>>> &segment_bucket_sizes,
    std::vector<std::shared_ptr<BucketSparseGradient<T>>> *buckets_ptr) {
    MS_EXCEPTION_IF_NULL(param.output_grad_);
    MS_EXCEPTION_IF_NULL(param.output_grad_->value_);
    MS_EXCEPTION_IF_NULL(param.output_grad_->indices_);
    MS_EXCEPTION_IF_NULL(buckets_ptr);
    auto &buckets = *buckets_ptr;
    size_t thread_num = param.thread_num_;
    if (thread_num != segment_bucket_sizes.size()) {
      MS_EXCEPTION(ArgumentError) << "Input param thread num not equal to segment size!";
    }
    std::vector<size_t> bucket_data_size(thread_num, 0);
    for (size_t i = 0; i < thread_num; ++i) {
      for (size_t j = 0; j < thread_num; ++j) {
        bucket_data_size[j] += segment_bucket_sizes[i]->at(j);
      }
    }
    size_t current_indices_offset = 0;
    for (size_t i = 0; i < thread_num; ++i) {
      (void)buckets.emplace_back(std::make_shared<BucketSparseGradient<T>>());
      buckets[i]->value_ = param.output_grad_->value_ + current_indices_offset * param.value_stride_;
      buckets[i]->indices_ = param.output_grad_->indices_ + current_indices_offset;
      buckets[i]->global_indices_ = param.workspace_grad_->indices_ + current_indices_offset;
      buckets[i]->indices_size_ = bucket_data_size[i];
      current_indices_offset += bucket_data_size[i];
    }
    std::vector<size_t> tmp_bucket_data_size(thread_num, 0);
    std::vector<std::vector<std::shared_ptr<BucketSparseGradient<T>>>> each_thread_buckets;
    for (size_t i = 0; i < thread_num; ++i) {
      std::vector<std::shared_ptr<BucketSparseGradient<T>>> thread_buckets;
      for (size_t j = 0; j < thread_num; ++j) {
        (void)thread_buckets.emplace_back(std::make_shared<BucketSparseGradient<T>>());
        thread_buckets[j]->indices_ = buckets[j]->indices_ + tmp_bucket_data_size[j];
        thread_buckets[j]->global_indices_ = buckets[j]->global_indices_ + tmp_bucket_data_size[j];
        thread_buckets[j]->value_ = buckets[j]->value_ + tmp_bucket_data_size[j] * param.value_stride_;
        thread_buckets[j]->indices_size_ = segment_bucket_sizes[i]->at(j);
        tmp_bucket_data_size[j] += segment_bucket_sizes[i]->at(j);
      }
      (void)each_thread_buckets.emplace_back(thread_buckets);
    }
    std::vector<common::Task> tasks;
    tasks.reserve(thread_num);
    current_indices_offset = 0;
    for (size_t i = 0; i < thread_num; ++i) {
      auto task = [&param, &segments, &each_thread_buckets, i, current_indices_offset]() {
        CopySegmentIndicesToBucket<T>(param, segments[i], current_indices_offset, each_thread_buckets[i]);
        return common::SUCCESS;
      };
      (void)tasks.emplace_back(task);
      current_indices_offset += segments[i]->indices_size_;
    }
    ParallelLaunch(tasks);
  }

  template <typename T>
  static void SortAndReduceBucketSparseGradient(const MultiThreadReduceSparseGradientParam<T> &param,
                                                const std::shared_ptr<BucketSparseGradient<T>> &bucket,
                                                const std::shared_ptr<SparseGradient<T>> &reduced_bucket) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(bucket);
    MS_EXCEPTION_IF_NULL(bucket->value_);
    MS_EXCEPTION_IF_NULL(bucket->indices_);
    MS_EXCEPTION_IF_NULL(reduced_bucket);
    MS_EXCEPTION_IF_NULL(reduced_bucket->value_);
    MS_EXCEPTION_IF_NULL(reduced_bucket->indices_);
    std::vector<std::pair<T, T>> sorted_indices;
    sorted_indices.reserve(bucket->indices_size_);
    for (size_t i = 0; i < bucket->indices_size_; ++i) {
      T index = bucket->indices_[i];
      T global_index = bucket->global_indices_[i];
      (void)sorted_indices.emplace_back(std::pair<T, T>(index, global_index));
    }
    std::sort(sorted_indices.begin(), sorted_indices.end());

    float *global_value = param.input_grad_->value_;
    size_t unique_indices_size = 0;
    size_t max_length = reduced_bucket->indices_size_ * param.value_stride_;
    T last_index{0};
    size_t value_offset{0};
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
      T index = sorted_indices[i].first;
      T global_index = sorted_indices[i].second;
      auto global_value_offset = static_cast<size_t>(global_index) * param.value_stride_;
      if (i == 0 || index != last_index) {
        if (i != 0) {
          unique_indices_size++;
        }
        reduced_bucket->indices_[unique_indices_size] = index;
        value_offset = unique_indices_size * param.value_stride_;
        auto ret_code = memcpy_s(reduced_bucket->value_ + value_offset, (max_length - value_offset) * sizeof(float),
                                 global_value + global_value_offset, param.value_stride_ * sizeof(float));
        if (ret_code != EOK) {
          MS_LOG(EXCEPTION) << "For 'SparseOptimizer', failed to copy data. Error no: " << ret_code;
        }
      } else {
        for (size_t j = 0; j < param.value_stride_; ++j) {
          *(reduced_bucket->value_ + value_offset + j) += *(global_value + global_value_offset + j);
        }
      }
      last_index = index;
    }
    reduced_bucket->indices_size_ = unique_indices_size;
    MS_LOG(DEBUG) << "End";
  }

  template <typename T>
  static void ReduceBucketSparseGradient(const MultiThreadReduceSparseGradientParam<T> &param,
                                         const std::shared_ptr<BucketSparseGradient<T>> &bucket,
                                         const std::shared_ptr<SparseGradient<T>> &reduced_bucket) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(bucket);
    MS_EXCEPTION_IF_NULL(bucket->value_);
    MS_EXCEPTION_IF_NULL(bucket->indices_);
    MS_EXCEPTION_IF_NULL(reduced_bucket);
    MS_EXCEPTION_IF_NULL(reduced_bucket->value_);
    MS_EXCEPTION_IF_NULL(reduced_bucket->indices_);

    float *global_value = param.input_grad_->value_;
    std::unordered_map<T, size_t> index_map;
    size_t unique_indices_size = 0;
    size_t max_length = reduced_bucket->indices_size_ * param.value_stride_;
    for (size_t i = 0; i < bucket->indices_size_; ++i) {
      T index = bucket->indices_[i];
      T global_index = bucket->global_indices_[i];
      auto iter = index_map.find(index);
      if (iter == index_map.end()) {
        reduced_bucket->indices_[unique_indices_size] = index;
        size_t start_index = unique_indices_size * param.value_stride_;
        index_map[index] = start_index;
        auto ret_code =
          memcpy_s(reduced_bucket->value_ + start_index, (max_length - start_index) * sizeof(float),
                   global_value + global_index * param.value_stride_, param.value_stride_ * sizeof(float));
        if (ret_code != EOK) {
          MS_LOG(EXCEPTION) << "For 'SparseOptimizer', failed to copy data. Error no: " << ret_code;
        }
        unique_indices_size++;
      } else {
        size_t start_index = iter->second;
        size_t end_index = start_index + param.value_stride_;
        for (size_t j = start_index, k = global_index * param.value_stride_; j < end_index; ++j, ++k) {
          reduced_bucket->value_[j] += global_value[k];
        }
      }
    }
    reduced_bucket->indices_size_ = unique_indices_size;
    MS_LOG(DEBUG) << "End";
  }

  template <typename T>
  static void ReduceBucketSparseGradientToWorkspace(
    const MultiThreadReduceSparseGradientParam<T> &param,
    const std::vector<std::shared_ptr<BucketSparseGradient<T>>> &buckets,
    std::vector<std::shared_ptr<SparseGradient<T>>> *reduced_buckets_ptr) {
    MS_EXCEPTION_IF_NULL(param.workspace_grad_);
    MS_EXCEPTION_IF_NULL(param.workspace_grad_->value_);
    MS_EXCEPTION_IF_NULL(param.workspace_grad_->indices_);
    MS_EXCEPTION_IF_NULL(reduced_buckets_ptr);
    auto &reduced_buckets = *reduced_buckets_ptr;
    size_t thread_num = buckets.size();
    std::vector<common::Task> tasks;
    tasks.reserve(thread_num);

    size_t current_indices_offset = 0;
    for (size_t i = 0; i < thread_num; ++i) {
      (void)reduced_buckets.emplace_back(std::make_shared<SparseGradient<T>>());
      reduced_buckets[i]->value_ = param.workspace_grad_->value_ + current_indices_offset * param.value_stride_;
      reduced_buckets[i]->indices_ = param.workspace_grad_->indices_ + current_indices_offset;
      reduced_buckets[i]->indices_size_ = buckets[i]->indices_size_;
      auto task = [&param, &buckets, &reduced_buckets, i]() {
        if (param.use_sort_reduce_) {
          SortAndReduceBucketSparseGradient<T>(param, buckets[i], reduced_buckets[i]);
        } else {
          ReduceBucketSparseGradient<T>(param, buckets[i], reduced_buckets[i]);
        }
        return common::SUCCESS;
      };
      (void)tasks.emplace_back(task);
      current_indices_offset += buckets[i]->indices_size_;
    }
    ParallelLaunch(tasks);
  }

  template <typename T>
  static void MergeReduceSparseGradient(const MultiThreadReduceSparseGradientParam<T> &param,
                                        const std::vector<std::shared_ptr<SparseGradient<T>>> &reduced_buckets) {
    MS_EXCEPTION_IF_NULL(param.output_grad_);
    auto output_grad = param.output_grad_;
    MS_EXCEPTION_IF_NULL(output_grad->value_);
    MS_EXCEPTION_IF_NULL(output_grad->indices_);
    size_t stride_data_size = param.value_stride_ * sizeof(float);
    size_t unique_indices_size = 0;
    for (size_t i = 0; i < reduced_buckets.size(); ++i) {
      auto &bucket = reduced_buckets[i];
      MS_EXCEPTION_IF_NULL(bucket);
      if (bucket->indices_size_ == 0) {
        continue;
      }
      auto ret_code = memcpy_s(output_grad->value_ + unique_indices_size * param.value_stride_,
                               (output_grad->indices_size_ - unique_indices_size) * stride_data_size, bucket->value_,
                               bucket->indices_size_ * stride_data_size);
      if (ret_code != EOK) {
        MS_LOG(EXCEPTION) << "For 'SparseOptimizer', failed to copy data. Error no: " << ret_code;
      }
      ret_code = memcpy_s(output_grad->indices_ + unique_indices_size,
                          (output_grad->indices_size_ - unique_indices_size) * sizeof(T), bucket->indices_,
                          bucket->indices_size_ * sizeof(T));
      if (ret_code != EOK) {
        MS_LOG(EXCEPTION) << "For 'SparseOptimizer', failed to copy data. Error no: " << ret_code;
      }
      unique_indices_size += bucket->indices_size_;
    }
    output_grad->indices_size_ = unique_indices_size;
  }

 protected:
  TypeId indices_data_type_{kNumberTypeInt32};
  size_t indices_size_{0};
  size_t var_first_dim_size_{0};
  size_t var_outer_dim_size_{1};
};
}  // namespace sparse_optimizer_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_OPTIMIZER_CPU_KERNEL_H_
