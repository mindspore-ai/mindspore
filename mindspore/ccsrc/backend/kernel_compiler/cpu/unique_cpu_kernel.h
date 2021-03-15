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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_CPU_KERNEL_H_
#include <algorithm>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
template <typename DataType, typename IndexType>
struct UniqueParam {
  DataType *input_{nullptr};
  IndexType *input_idx_{nullptr};
  DataType *output_{nullptr};
  IndexType *inverse_idx_{nullptr};
  DataType *workspace_{nullptr};
  IndexType *workspace_idx_{nullptr};
  IndexType input_size_{0};
  IndexType output_size_{0};
  size_t thread_num_{0};
  bool need_sort_{true};
};

class UniqueCPUKernel : public CPUKernel {
 public:
  UniqueCPUKernel() = default;
  ~UniqueCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  void InitInputOutputSize(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename DataType, typename IndexType>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 protected:
  virtual void CheckParam(const CNodePtr &kernel_node);
  size_t input_size_{0};
  TypeId dtype_{kTypeUnknown};
  size_t output_size_{0};
  CNodeWeakPtr node_wpt_;

  template <typename DataType>
  static size_t BucketId(DataType data, size_t bucket_num) {
    if (bucket_num < 1) {
      return static_cast<size_t>(data);
    }
    return static_cast<size_t>(data) % bucket_num;
  }

  template <typename DataType, typename IndexType>
  static void CalculateEachBucketSize(const std::shared_ptr<UniqueParam<DataType, IndexType>> &params,
                                      std::vector<IndexType> *each_bucket_size) {
    MS_EXCEPTION_IF_NULL(params);
    MS_EXCEPTION_IF_NULL(params->input_);
    MS_EXCEPTION_IF_NULL(each_bucket_size);
    size_t bucket_num = each_bucket_size->size();
    if (params->input_size_ < 1) {
      return;
    }
    for (IndexType i = 0; i < params->input_size_; ++i) {
      auto bucket_id = BucketId(params->input_[i], bucket_num);
      each_bucket_size->at(bucket_id)++;
    }
  }

  template <typename DataType, typename IndexType>
  static void SplitAndCalculateBucketSize(
    const std::shared_ptr<UniqueParam<DataType, IndexType>> &params,
    std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> *segments_ptr,
    std::vector<std::shared_ptr<std::vector<IndexType>>> *segment_bucket_sizes_ptr) {
    MS_EXCEPTION_IF_NULL(params);
    MS_EXCEPTION_IF_NULL(params->input_);
    MS_EXCEPTION_IF_NULL(segments_ptr);
    MS_EXCEPTION_IF_NULL(segment_bucket_sizes_ptr);
    auto &segments = *segments_ptr;
    auto &segment_bucket_sizes = *segment_bucket_sizes_ptr;

    IndexType input_size = params->input_size_;
    size_t thread_num = params->thread_num_;
    if (thread_num < 1) {
      MS_LOG(EXCEPTION) << "Thread num must > 0 !";
    }
    IndexType thread_data_size = input_size / thread_num;
    size_t left_data_size = input_size % thread_num;
    segments.reserve(thread_num);
    segment_bucket_sizes.reserve(thread_num);
    IndexType current_offset = 0;
    std::vector<common::Task> tasks;
    tasks.reserve(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      segment_bucket_sizes.emplace_back(std::make_shared<std::vector<IndexType>>(thread_num, 0));
      IndexType data_size = thread_data_size;
      if (i < left_data_size) {
        data_size += 1;
      }
      segments.emplace_back(std::make_shared<UniqueParam<DataType, IndexType>>());
      segments[i]->input_ = params->input_ + current_offset;
      segments[i]->input_size_ = data_size;
      segments[i]->thread_num_ = thread_num;
      auto task = [&segments, &segment_bucket_sizes, i]() {
        CalculateEachBucketSize<DataType, IndexType>(segments[i], segment_bucket_sizes[i].get());
        return common::SUCCESS;
      };
      tasks.emplace_back(task);
      current_offset += data_size;
    }
    common::ThreadPool::GetInstance().SyncRun(tasks);
  }

  template <typename DataType, typename IndexType>
  static void SegmentToBuckets(const std::shared_ptr<UniqueParam<DataType, IndexType>> &segment,
                               IndexType segment_offset,
                               const std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> &buckets) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(segment);
    MS_EXCEPTION_IF_NULL(segment->input_);
    std::vector<IndexType> bucket_data_num(segment->thread_num_, 0);
    auto bucket_size = buckets.size();
    if (segment->input_size_ < 1) {
      return;
    }
    for (IndexType i = 0; i < segment->input_size_; ++i) {
      DataType data = segment->input_[i];
      auto bucket_id = BucketId(data, segment->thread_num_);
      auto bucket_index = bucket_data_num[bucket_id];
      if (bucket_id >= bucket_size) {
        MS_LOG(ERROR) << "Error bucket id!";
        continue;
      }
      auto &bucket = buckets[bucket_id];
      MS_EXCEPTION_IF_NULL(bucket);
      if (bucket_index >= bucket->input_size_) {
        MS_LOG(ERROR) << "Error bucket index!";
        continue;
      }
      bucket->input_[bucket_index] = data;
      bucket->workspace_idx_[bucket_index] = segment_offset + i;
      bucket_data_num[bucket_id]++;
    }
    MS_LOG(DEBUG) << "End";
  }

  template <typename DataType, typename IndexType>
  static void GatherSegmentsToBuckets(const std::shared_ptr<UniqueParam<DataType, IndexType>> &params,
                                      std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> *segments_ptr,
                                      std::vector<std::shared_ptr<std::vector<IndexType>>> *segment_bucket_sizes_ptr,
                                      std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> *buckets_ptr) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(params);
    MS_EXCEPTION_IF_NULL(params->workspace_);
    MS_EXCEPTION_IF_NULL(params->inverse_idx_);
    MS_EXCEPTION_IF_NULL(params->workspace_idx_);
    MS_EXCEPTION_IF_NULL(params->output_);
    MS_EXCEPTION_IF_NULL(params->input_idx_);
    MS_EXCEPTION_IF_NULL(segments_ptr);
    MS_EXCEPTION_IF_NULL(segment_bucket_sizes_ptr);
    MS_EXCEPTION_IF_NULL(buckets_ptr);
    auto &segments = *segments_ptr;
    auto &segment_bucket_sizes = *segment_bucket_sizes_ptr;
    auto &buckets = *buckets_ptr;
    auto thread_num = segments.size();
    buckets.reserve(thread_num);
    std::vector<IndexType> bucket_data_size(thread_num, 0);
    for (size_t i = 0; i < thread_num; ++i) {
      for (size_t j = 0; j < thread_num; ++j) {
        bucket_data_size[j] += segment_bucket_sizes[i]->at(j);
      }
    }

    IndexType current_offset = 0;
    for (size_t i = 0; i < thread_num; ++i) {
      auto bucket = std::make_shared<UniqueParam<DataType, IndexType>>();
      bucket->input_ = params->output_ + current_offset;
      bucket->input_idx_ = params->inverse_idx_ + current_offset;
      bucket->workspace_idx_ = params->workspace_idx_ + current_offset;
      bucket->output_ = params->workspace_ + current_offset;
      bucket->inverse_idx_ = params->input_idx_ + current_offset;
      bucket->input_size_ = bucket_data_size[i];
      current_offset += bucket_data_size[i];
      buckets.emplace_back(bucket);
    }
    std::vector<IndexType> tmp_bucket_data_size(thread_num, 0);
    std::vector<std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>>> thread_buckets;
    for (size_t i = 0; i < thread_num; ++i) {
      std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> local_buckets;
      for (size_t j = 0; j < thread_num; ++j) {
        auto bucket = std::make_shared<UniqueParam<DataType, IndexType>>();
        bucket->input_ = buckets[j]->input_ + tmp_bucket_data_size[j];
        bucket->input_size_ = buckets[j]->input_size_ - tmp_bucket_data_size[j];
        bucket->workspace_idx_ = buckets[j]->workspace_idx_ + tmp_bucket_data_size[j];
        local_buckets.emplace_back(bucket);
        tmp_bucket_data_size[j] += segment_bucket_sizes[i]->at(j);
      }
      thread_buckets.emplace_back(local_buckets);
    }
    std::vector<common::Task> tasks;
    tasks.reserve(thread_num);
    current_offset = 0;
    for (size_t i = 0; i < thread_num; ++i) {
      MS_EXCEPTION_IF_NULL(segments[i]);
      auto task = [&segments, &thread_buckets, current_offset, i]() {
        SegmentToBuckets<DataType, IndexType>(segments[i], current_offset, thread_buckets[i]);
        return common::SUCCESS;
      };
      tasks.emplace_back(task);
      current_offset += segments[i]->input_size_;
    }
    common::ThreadPool::GetInstance().SyncRun(tasks);
    MS_LOG(DEBUG) << "End";
  }

  template <typename DataType, typename IndexType>
  static void Unique(const std::shared_ptr<UniqueParam<DataType, IndexType>> &params) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(params);
    DataType *input = params->input_;
    IndexType *input_idx = params->input_idx_;
    DataType *output = params->output_;
    IndexType *inverse_idx = params->inverse_idx_;
    MS_EXCEPTION_IF_NULL(input);
    MS_EXCEPTION_IF_NULL(input_idx);
    MS_EXCEPTION_IF_NULL(output);
    MS_EXCEPTION_IF_NULL(inverse_idx);
    IndexType j = 0;
    if (params->input_size_ < 1) {
      return;
    }
    if (params->need_sort_) {
      for (IndexType i = 0; i < params->input_size_; ++i) {
        input_idx[i] = i;
      }
      std::sort(input_idx, input_idx + params->input_size_,
                [&](IndexType left, IndexType right) { return input[left] < input[right]; });
      DataType last = input[0];
      for (IndexType i = 0; i < params->input_size_; ++i) {
        auto curr = input[input_idx[i]];
        if (i == 0 || curr != last) {
          if (i != 0) {
            j++;
          }
          output[j] = curr;
          inverse_idx[input_idx[i]] = j;
          last = curr;
        } else {
          inverse_idx[input_idx[i]] = j;
        }
      }
      params->output_size_ = j + 1;
    } else {
      std::unordered_map<DataType, IndexType> uniq;
      uniq.reserve(params->input_size_);
      for (IndexType i = 0; i < params->input_size_; ++i) {
        auto it = uniq.emplace(input[i], j);
        inverse_idx[i] = it.first->second;
        if (it.second) {
          ++j;
        }
      }
      for (const auto &it : uniq) {
        output[it.second] = it.first;
      }
      params->output_size_ = j;
    }
    MS_LOG(DEBUG) << "End";
  }

  template <typename DataType, typename IndexType>
  static void UniqueEachBucket(const std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> &buckets) {
    MS_LOG(DEBUG) << "Start";
    size_t thread_num = buckets.size();
    std::vector<common::Task> tasks;
    tasks.reserve(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      auto task = [&buckets, i]() {
        Unique<DataType, IndexType>(buckets[i]);
        return common::SUCCESS;
      };
      tasks.emplace_back(task);
    }
    common::ThreadPool::GetInstance().SyncRun(tasks);
    MS_LOG(DEBUG) << "End";
  }

  template <typename DataType, typename IndexType>
  static void TransformBucketReverseIndices(const std::shared_ptr<UniqueParam<DataType, IndexType>> &bucket,
                                            const std::shared_ptr<UniqueParam<DataType, IndexType>> &result,
                                            IndexType offset) {
    MS_EXCEPTION_IF_NULL(bucket);
    MS_EXCEPTION_IF_NULL(bucket->inverse_idx_);
    MS_EXCEPTION_IF_NULL(bucket->workspace_idx_);
    MS_EXCEPTION_IF_NULL(result);
    MS_EXCEPTION_IF_NULL(result->inverse_idx_);
    if (bucket->input_size_ < 1) {
      return;
    }
    for (IndexType i = 0; i < bucket->input_size_; ++i) {
      auto origin_idx = bucket->workspace_idx_[i];
      if (origin_idx >= 0 && origin_idx < result->input_size_) {
        result->inverse_idx_[origin_idx] = bucket->inverse_idx_[i] + offset;
      }
    }
  }

  template <typename DataType, typename IndexType>
  static void MergeBuckets(const std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> &buckets,
                           const std::shared_ptr<UniqueParam<DataType, IndexType>> &result) {
    MS_LOG(DEBUG) << "Start";
    MS_EXCEPTION_IF_NULL(result);
    MS_EXCEPTION_IF_NULL(result->output_);
    size_t thread_num = buckets.size();
    std::vector<IndexType> bucket_offsets(thread_num);
    IndexType current_size = 0;
    for (size_t i = 0; i < thread_num; ++i) {
      auto bucket = buckets[i];
      MS_EXCEPTION_IF_NULL(bucket);
      MS_EXCEPTION_IF_NULL(bucket->output_);
      bucket_offsets[i] = current_size;
      auto ret_code = memcpy_s(result->output_ + current_size, (result->input_size_ - current_size) * sizeof(DataType),
                               bucket->output_, bucket->output_size_ * sizeof(DataType));
      if (ret_code != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy data!";
      }
      current_size += bucket->output_size_;
    }
    result->output_size_ = current_size;

    std::vector<common::Task> tasks;
    tasks.reserve(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      auto task = [&buckets, i, result, &bucket_offsets]() {
        TransformBucketReverseIndices<DataType, IndexType>(buckets[i], result, bucket_offsets[i]);
        return common::SUCCESS;
      };
      tasks.emplace_back(task);
    }
    common::ThreadPool::GetInstance().SyncRun(tasks);
    MS_LOG(DEBUG) << "End";
  }

  template <typename DataType, typename IndexType>
  static void BucketUnique(const std::shared_ptr<UniqueParam<DataType, IndexType>> &params) {
    MS_EXCEPTION_IF_NULL(params);
    std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> segments;
    std::vector<std::shared_ptr<UniqueParam<DataType, IndexType>>> buckets;
    std::vector<std::shared_ptr<std::vector<IndexType>>> segment_bucket_sizes;
    SplitAndCalculateBucketSize(params, &segments, &segment_bucket_sizes);
    GatherSegmentsToBuckets(params, &segments, &segment_bucket_sizes, &buckets);
    UniqueEachBucket(buckets);
    MergeBuckets(buckets, params);
  }
};

MS_REG_CPU_KERNEL(
  Unique, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  UniqueCPUKernel);

MS_REG_CPU_KERNEL(
  Unique, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  UniqueCPUKernel);

MS_REG_CPU_KERNEL(
  Unique,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
  UniqueCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_CPU_KERNEL_H_
