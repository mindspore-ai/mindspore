/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_task_list.h"
#include "utils/ms_utils.h"
#include "minddata/mindrecord/include/common/shard_utils.h"

namespace mindspore {
namespace mindrecord {
ShardTaskList::ShardTaskList() : categories(1), lazy_load_(false) {}

ShardTaskList::ShardTaskList(const ShardTaskList &other)
    : categories(other.categories),
      permutation_(other.permutation_),
      sample_ids_(other.sample_ids_),
      task_list_(other.task_list_),
      sample_meta_list_(other.sample_meta_list_),
      lazy_load_(other.lazy_load_) {}

ShardTaskList &ShardTaskList::operator=(const ShardTaskList &other) {
  ShardTaskList tmp(other);
  std::swap(categories, tmp.categories);
  permutation_.swap(tmp.permutation_);
  sample_ids_.swap(tmp.sample_ids_);
  task_list_.swap(tmp.task_list_);
  sample_meta_list_.swap(tmp.sample_meta_list_);
  lazy_load_ = tmp.lazy_load_;
  return *this;
}

void ShardTaskList::InitSampleIds() {
  // no-op if there already exists sample ids.  Do not clobber previous list
  if (sample_ids_.empty()) {
    sample_ids_ = std::vector<int64_t>(task_list_.size());
    for (auto i = 0; i < task_list_.size(); i++) {
      sample_ids_[i] = i;
    }
  }
}

void ShardTaskList::MakePerm() {
  int64_t perm_size = sample_ids_.size();
  permutation_ = std::vector<int64_t>(perm_size);
  for (int64_t i = 0; i < perm_size; i++) {
    permutation_[i] = i;
  }
}

// Swap the new_tasks with orig_tasks
void ShardTaskList::TaskListSwap(ShardTaskList &orig_tasks, ShardTaskList &new_tasks) {
  // When swapping, if the orig_tasks contains fields that need to be preserved after the swap, then swapping with a
  // new_tasks that does not have those fields will result in clobbering/losing the data after the swap.
  // The task_list_ should not be lost/clobbered.
  // This function can be called in the middle of mindrecord's epoch, when orig_tasks.task_list_ is still being
  // used by mindrecord op's worker threads. So don't touch its task_list_ since this field should be preserved anyways.

  std::swap(orig_tasks.categories, new_tasks.categories);
  std::swap(orig_tasks.permutation_, new_tasks.permutation_);
  std::swap(orig_tasks.sample_ids_, new_tasks.sample_ids_);
}

void ShardTaskList::PopBack() {
  task_list_.pop_back();
  if (lazy_load_ == false) {
    sample_meta_list_.pop_back();
  }
}

int64_t ShardTaskList::Size() const { return static_cast<int64_t>(task_list_.size()); }

int64_t ShardTaskList::SizeOfRows() const {
  int64_t size_of_rows = 0;
  if (task_list_.size() == 0) {
    return size_of_rows;
  }

  if (lazy_load_ == false) {
    // 1 task is 1 page,blob index start from 2
    auto sum_num_rows = [](int64_t x, SampleMeta y) { return x + std::get<0>(y)[0]; };
    size_of_rows = std::accumulate(sample_meta_list_.begin(), sample_meta_list_.end(), 0, sum_num_rows);
  } else {
    MS_LOG(WARNING) << "In lazy load mode, size of rows will be " << size_of_rows << " which is not correctly.";
  }
  return size_of_rows;
}

ShardTask ShardTaskList::GetTaskByID(int64_t id) {
  if (lazy_load_ == false) {
    return {std::get<0>(task_list_[id]), std::get<1>(task_list_[id]), std::get<0>(sample_meta_list_[id]),
            std::get<1>(sample_meta_list_[id])};
  } else {
    return {std::get<0>(task_list_[id]), std::get<1>(task_list_[id]), {}, json()};
  }
}

int64_t ShardTaskList::GetTaskSampleByID(int64_t id) { return sample_ids_[id]; }

int64_t ShardTaskList::GetRandomTaskID() {
  std::mt19937 gen = GetRandomDevice();
  std::uniform_int_distribution<> dis(0, sample_ids_.size() - 1);
  return dis(gen);
}

ShardTask ShardTaskList::GetRandomTask() {
  std::mt19937 gen = GetRandomDevice();
  std::uniform_int_distribution<> dis(0, task_list_.size() - 1);
  size_t random = dis(gen);
  if (lazy_load_ == false) {
    return {std::get<0>(task_list_[random]), std::get<1>(task_list_[random]), std::get<0>(sample_meta_list_[random]),
            std::get<1>(sample_meta_list_[random])};
  } else {
    return {std::get<0>(task_list_[random]), std::get<1>(task_list_[random]), {}, json()};
  }
}

ShardTaskList ShardTaskList::Combine(std::vector<ShardTaskList> &category_tasks, bool replacement, int64_t num_elements,
                                     int64_t num_samples) {
  ShardTaskList res;
  if (category_tasks.empty()) {
    return res;
  }
  auto total_categories = category_tasks.size();
  res.categories = static_cast<int64_t>(total_categories);
  if (replacement == false) {
    auto minTasks = category_tasks[0].Size();
    for (int64_t i = 1; i < total_categories; i++) {
      minTasks = std::min(minTasks, category_tasks[i].Size());
    }
    int64_t count = 0;
    for (int64_t task_no = 0; task_no < minTasks; task_no++) {
      for (int64_t i = 0; i < total_categories; i++) {
        if (num_samples != 0 && count == num_samples) {
          break;
        }
        res.InsertTask(std::move(category_tasks[i].GetTaskByID(task_no)));
        count++;
      }
    }
  } else {
    auto maxTasks = category_tasks[0].Size();
    for (int64_t i = 1; i < total_categories; i++) {
      maxTasks = std::max(maxTasks, category_tasks[i].Size());
    }
    if (num_elements != std::numeric_limits<int64_t>::max()) {
      maxTasks = static_cast<decltype(maxTasks)>(num_elements);
    }
    int64_t count = 0;
    for (int64_t i = 0; i < total_categories; i++) {
      for (int64_t j = 0; j < maxTasks; j++) {
        if (num_samples != 0 && count == num_samples) {
          break;
        }
        res.InsertTask(category_tasks[i].GetRandomTask());
        count++;
      }
    }
  }

  return res;
}
}  // namespace mindrecord
}  // namespace mindspore
