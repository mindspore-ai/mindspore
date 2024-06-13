/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "minddata/mindrecord/include/common/shard_utils.h"

namespace mindspore {
namespace mindrecord {
// the shuffle size when mindrecord is slow load mode
const int64_t ShuffleSize = 50000000;

GeneratorIds::GeneratorIds() : partitioned_shard_sample_count_(), partition_index_(0), partition_sample_index_(0) {}

void GeneratorIds::SetShardSampleCount(const std::vector<PartitionedShardSampleCount> &partitioned_shard_sample_count) {
  partitioned_shard_sample_count_ = partitioned_shard_sample_count;
  partition_index_ = 0;
  partition_sample_index_ = 0;
}

void GeneratorIds::ResetShardIndexAndID() {
  partition_index_ = 0;
  partition_sample_index_ = 0;
}

std::vector<int64_t> GeneratorIds::GetNextSampleIds(const bool &need_shuffle, const uint32_t &seed) {
  // partitioned_shard_sample_count_ is:
  // CommonTask, 0, 16680, 17777
  // CommonTask, 0, 0, 15
  std::vector<int64_t> ids;
  for (int32_t i = partition_index_; i < partitioned_shard_sample_count_.size(); i++) {
    for (int64_t j = partitioned_shard_sample_count_[i].start + partition_sample_index_;
         j < partitioned_shard_sample_count_[i].end; j++) {
      ids.push_back(j);
      partition_sample_index_++;
      if (ids.size() >= ShuffleSize) {
        if (need_shuffle) {
          std::shuffle(ids.begin(), ids.end(), std::default_random_engine(seed));
        }
        return ids;
      }
    }
    partition_index_++;
    partition_sample_index_ = 0;
  }
  if (need_shuffle) {
    std::shuffle(ids.begin(), ids.end(), std::default_random_engine(seed));
  }
  return ids;
}

ShardTaskList::ShardTaskList()
    : categories(1), padded_sample_(0), need_shuffle_(false), shuffle_seed_(0), load_mode_(LoadMode::kFast) {}

ShardTaskList::ShardTaskList(const ShardTaskList &other)
    : categories(other.categories),
      permutation_(other.permutation_),
      sample_ids_(other.sample_ids_),
      task_list_(other.task_list_),
      sample_meta_list_(other.sample_meta_list_),
      shard_sample_count_(other.shard_sample_count_),
      padded_sample_(other.padded_sample_),
      file_ids_(other.file_ids_),
      shuffled_shard_sample_count_(other.shuffled_shard_sample_count_),
      partitioned_shard_sample_count_(other.partitioned_shard_sample_count_),
      need_shuffle_(other.need_shuffle_),
      shuffle_seed_(other.shuffle_seed_),
      generator_ids_(other.generator_ids_),
      load_mode_(other.load_mode_) {}

ShardTaskList &ShardTaskList::operator=(const ShardTaskList &other) {
  ShardTaskList tmp(other);
  std::swap(categories, tmp.categories);
  permutation_.swap(tmp.permutation_);
  sample_ids_.swap(tmp.sample_ids_);
  task_list_.swap(tmp.task_list_);
  sample_meta_list_.swap(tmp.sample_meta_list_);
  shard_sample_count_.swap(tmp.shard_sample_count_);
  padded_sample_ = tmp.padded_sample_;
  file_ids_.swap(tmp.file_ids_);
  shuffled_shard_sample_count_.swap(tmp.shuffled_shard_sample_count_);
  partitioned_shard_sample_count_.swap(tmp.partitioned_shard_sample_count_);
  need_shuffle_ = tmp.need_shuffle_;
  shuffle_seed_ = tmp.shuffle_seed_;
  generator_ids_ = tmp.generator_ids_;
  load_mode_ = tmp.load_mode_;
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
  if (load_mode_ == LoadMode::kFast) {
    sample_meta_list_.pop_back();
  }
}

int64_t ShardTaskList::Size() const {
  if (load_mode_ != LoadMode::kSlow) {
    return static_cast<int64_t>(task_list_.size());
  }

  // slow load mode
  return shard_sample_count_[shard_sample_count_.size() - 1] + padded_sample_;
}

int64_t ShardTaskList::SizeAfterSampling() const {
  if (load_mode_ != LoadMode::kSlow) {
    return static_cast<int64_t>(sample_ids_.size());
  }

  // slow load mode
  int64_t count = 0;
  for (int32_t i = 0; i < partitioned_shard_sample_count_.size(); i++) {
    count += partitioned_shard_sample_count_[i].end - partitioned_shard_sample_count_[i].start;
  }
  return count;
}

int64_t ShardTaskList::SizeOfRows() const {
  int64_t size_of_rows = 0;
  if (task_list_.size() == 0) {
    return size_of_rows;
  }

  if (load_mode_ == LoadMode::kFast) {
    // 1 task is 1 page,blob index start from 2
    auto sum_num_rows = [](int64_t x, SampleMeta y) { return x + std::get<0>(y)[0]; };
    size_of_rows = std::accumulate(sample_meta_list_.begin(), sample_meta_list_.end(), 0, sum_num_rows);
  } else {
    MS_LOG(WARNING) << "In lazy load mode, size of rows will be " << size_of_rows << " which is not correctly.";
  }
  return size_of_rows;
}

ShardTask ShardTaskList::GetTaskByID(int64_t id) {
  if (load_mode_ == LoadMode::kFast) {
    return {std::get<0>(task_list_[id]), std::get<1>(task_list_[id]), std::get<0>(sample_meta_list_[id]),
            std::get<1>(sample_meta_list_[id])};
  } else if (load_mode_ == LoadMode::kLazy) {
    return {std::get<0>(task_list_[id]), std::get<1>(task_list_[id]), {}, json()};
  }

  TaskType task_type = TaskType::kCommonTask;
  // get the partitioned shard id
  int32_t shard_id = 0;
  int32_t row_id = 0;
  for (int32_t i = 0; i < partitioned_shard_sample_count_.size(); i++) {
    if (id >= partitioned_shard_sample_count_[i].start && id < partitioned_shard_sample_count_[i].end) {
      task_type = partitioned_shard_sample_count_[i].task_type;
      shard_id = partitioned_shard_sample_count_[i].shard_id;
      break;
    }
  }

  if (shard_id == -1) {
    return {TaskType::kPaddedTask, std::make_tuple(shard_id, row_id), {}, json()};
  }

  // get the original shard_id which is in order with mindrecord files
  shard_id = file_ids_[shard_id];

  // get the row id in the shard
  row_id = id;
  for (int32_t i = 0; i < shuffled_shard_sample_count_.size(); i++) {
    if (id < shuffled_shard_sample_count_[i]) {
      if (i > 0) {
        row_id = id - shuffled_shard_sample_count_[i - 1];
      }
      break;
    }
  }

  return {task_type, std::make_tuple(shard_id, row_id), {}, json()};
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
  if (load_mode_ == LoadMode::kFast) {
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
  if (!replacement) {
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

void ShardTaskList::SetShardSampleCount(const std::vector<int64_t> &shard_sample_count) {
  // original shard sample count like:
  // indicate shard_id : inc_count
  // 0 : 15  -  shard0 has 15 samples
  // 1 : 41  -  shard1 has 26 samples
  // 2 : 58  -  shard2 has 17 samples
  shard_sample_count_ = shard_sample_count;

  // generate new file_ids
  std::vector<int32_t> file_ids;
  for (int32_t i = 0; i < shard_sample_count_.size(); i++) {
    file_ids.push_back(i);
  }
  SetFileIds(file_ids);
}

void ShardTaskList::SetPaddedSample(const int32_t &padded_sample) { padded_sample_ = padded_sample; }

void ShardTaskList::SetFileIds(const std::vector<int32_t> &file_ids) {
  file_ids_ = file_ids;

  // original shard sample count like:
  // indicate shard_id : inc_count
  // 0 : 15  -  shard0 has 15 samples
  // 1 : 41  -  shard1 has 26 samples
  // 2 : 58  -  shard2 has 17 samples
  // create shuffled_shard_sample_count_
  // after shuffle
  // 0 : 17  -  shard0 has 17 samples - pre shard2
  // 1 : 32  -  shard1 has 15 samples - pre shard0
  // 2 : 58  -  shard2 has 26 samples - pre shard1
  std::vector<int64_t> shuffled_shard_sample_count;
  int64_t count = 0;
  int64_t start;
  for (int32_t i = 0; i < file_ids_.size(); i++) {
    if (file_ids[i] == 0) {
      start = 0;
    } else {
      start = shard_sample_count_[file_ids[i] - 1];
    }
    shuffled_shard_sample_count.push_back(shard_sample_count_[file_ids[i]] - start + count);
    count += shard_sample_count_[file_ids[i]] - start;
  }
  SetShuffledShardSampleCount(shuffled_shard_sample_count);
}

void ShardTaskList::SetShuffledShardSampleCount(const std::vector<int64_t> &shuffled_shard_sample_count) {
  shuffled_shard_sample_count_ = shuffled_shard_sample_count;

  // generate new partitioned_shard_sample_count
  std::vector<PartitionedShardSampleCount> vpssc;
  int64_t start = 0;
  for (int32_t shard_index = 0; shard_index < shuffled_shard_sample_count_.size(); shard_index++) {
    // add new range to vp
    PartitionedShardSampleCount pssc;
    pssc.task_type = TaskType::kCommonTask;
    pssc.shard_id = shard_index;
    pssc.start = start;
    pssc.end = shuffled_shard_sample_count_[shard_index];
    vpssc.push_back(pssc);
    start = shuffled_shard_sample_count_[shard_index];
  }

  // padded scenario
  if (padded_sample_ > 0) {
    PartitionedShardSampleCount pssc;
    pssc.task_type = TaskType::kPaddedTask;
    pssc.shard_id = -1;
    pssc.start = start;
    pssc.end = start + padded_sample_;
    vpssc.push_back(pssc);
  }

  SetPartitionedShardSampleCount(vpssc);
}

void ShardTaskList::SetPartitionedShardSampleCount(
  const std::vector<PartitionedShardSampleCount> &partitioned_shard_sample_count) {
  partitioned_shard_sample_count_ = partitioned_shard_sample_count;
  generator_ids_.SetShardSampleCount(partitioned_shard_sample_count_);
}

void ShardTaskList::UpdatePartitionedShardSampleCountByNumSamples(const int64_t &num_samples) {
  auto count = num_samples;
  std::vector<PartitionedShardSampleCount> new_partitioned_shard_sample_count = {};
  for (int32_t i = 0; i < partitioned_shard_sample_count_.size(); i++) {
    auto start = partitioned_shard_sample_count_[i].start;
    if (partitioned_shard_sample_count_[i].end - start <= count) {
      new_partitioned_shard_sample_count.push_back(partitioned_shard_sample_count_[i]);
      count = count - (partitioned_shard_sample_count_[i].end - start);
    } else {
      PartitionedShardSampleCount pssc;
      pssc.task_type = partitioned_shard_sample_count_[i].task_type;
      pssc.shard_id = partitioned_shard_sample_count_[i].shard_id;
      pssc.start = start;
      pssc.end = start + count;
      new_partitioned_shard_sample_count.push_back(pssc);
      break;
    }
  }

  SetPartitionedShardSampleCount(new_partitioned_shard_sample_count);
}

std::vector<int64_t> ShardTaskList::GetNextSampleIds() {
  return generator_ids_.GetNextSampleIds(need_shuffle_, shuffle_seed_);
}
}  // namespace mindrecord
}  // namespace mindspore
