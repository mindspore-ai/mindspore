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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/mindrecord_macro.h"

namespace mindspore {
namespace mindrecord {

// The data struct is as below:
// 1. TaskType: kCommonTask / kPaddedTask
// 2. std::tuple<int, int> : shard_id, group_id(fast load) / sample_id(lazy load)
// 3. std::vector<uint64_t> : [blob_start, blob_end]
// 4. json : scalar_variable_fields
using ShardTask = std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json>;

// The data struct is as below:
// 1. TaskType: kCommonTask / kPaddedTask
// 2. std::tuple<int, int> : shard_id, group_id(fast load) / sample_id(lazy load)
using TaskInfo = std::tuple<TaskType, std::tuple<int, int>>;

// The data struct is as below: contain the meta info
// 3. std::vector<uint64_t> : [blob_start, blob_end]
// 4. json : scalar_variable_fields
using SampleMeta = std::tuple<std::vector<uint64_t>, json>;

// The data struct is used to cache meta info when load mode is slow load
// task_type: kCommonTask / kPaddedTask
// shard_id: the index of mindrecord files
// start: the global index of all the samples
// end: the global index of all the samples
struct PartitionedShardSampleCount {
  TaskType task_type;
  int32_t shard_id;
  int64_t start;
  int64_t end;
};

class MINDRECORD_API GeneratorIds {
 public:
  GeneratorIds();
  void SetShardSampleCount(const std::vector<PartitionedShardSampleCount> &partitioned_shard_sample_count);
  void ResetShardIndexAndID();
  std::vector<int64_t> GetNextSampleIds(const bool &need_shuffle, const uint32_t &seed);

 private:
  // example:
  // kCommonTask, 4, 100, 250
  // kCommonTask, 5, 250, 700
  // kCommonTask, 0, 0, 15
  std::vector<PartitionedShardSampleCount> partitioned_shard_sample_count_;
  int32_t partition_index_;
  int64_t partition_sample_index_;
};

// There are three load mode
// fast mode: use ShardTask to cache meta data
// lazy mode: use TaskInfo to cache meta data
// slow mode: just cache shard_id:sample_count
class MINDRECORD_API ShardTaskList {
 public:
  ShardTaskList();

  ShardTaskList(const ShardTaskList &task);  // copy construction

  ShardTaskList &operator=(const ShardTaskList &task);  // assignment operator

  ~ShardTaskList() = default;

  void InitSampleIds();

  static void TaskListSwap(ShardTaskList &orig_tasks, ShardTaskList &new_tasks);

  // Assigns the task based on task id
  inline void AssignTask(ShardTaskList &sourceTasks, int64_t id);  // NOLINT

  inline void InsertTask(TaskType task_type, int shard_id, int group_id, const std::vector<uint64_t> &offset,
                         const json &label);

  inline void InsertTask(const int64_t &i, TaskType task_type, int shard_id, int group_id,
                         const std::vector<uint64_t> &offset, const json &label);

  inline void InsertTask(ShardTask task);

  inline void InsertTask(const int64_t &i, ShardTask task);

  void MakePerm();

  inline void InsertSampleId(int64_t id);

  void PopBack();

  int64_t Size() const;

  int64_t SizeAfterSampling() const;

  int64_t SizeOfRows() const;

  ShardTask GetTaskByID(int64_t id);

  ShardTask GetRandomTask();

  int64_t GetTaskSampleByID(int64_t id);

  int64_t GetRandomTaskID();

  static ShardTaskList Combine(std::vector<ShardTaskList> &category_tasks, bool replacement,  // NOLINT
                               int64_t num_elements, int64_t num_samples);

  inline void ResizeTask(const int64_t &size);

  // used for slow load mode
  void SetShardSampleCount(const std::vector<int64_t> &shard_sample_count);
  void SetPaddedSample(const int32_t &padded_sample);
  void SetFileIds(const std::vector<int32_t> &file_ids);
  void SetShuffledShardSampleCount(const std::vector<int64_t> &shuffled_shard_sample_count);
  void SetPartitionedShardSampleCount(const std::vector<PartitionedShardSampleCount> &partitioned_shard_sample_count);
  void UpdatePartitionedShardSampleCountByNumSamples(const int64_t &num_samples);
  std::vector<int64_t> GetNextSampleIds();

  uint32_t categories;

  // >>>> fast load meta data & lazy load meta data >>>>
  std::vector<int64_t> permutation_;  // A list of ints used for shuffling sample ids

  std::vector<int64_t> sample_ids_;  // The list of actual ids that were sampled

  // fast mode: [{TaskType, (shard_id, group_id(fast load))}, ...]
  // lazy mode: [{TaskType, (shard_id, sample_id(lazy load))}, ...]
  std::vector<TaskInfo> task_list_;

  // fast mode: [{[blob_start, blob_end], json}, ...]
  // lazy mode: none
  std::vector<SampleMeta> sample_meta_list_;

  // >>>> slow load meta data >>>>
  // indicate shard_id : inc_count
  // 0 : 15  -  shard 0 has 15 samples
  // 1 : 41  -  shard 1 has 26 samples
  // 2 : 58  -  shard 2 has 17 samples
  std::vector<int64_t> shard_sample_count_;
  int32_t padded_sample_;  // the padded sample
  // shuffle shard indexes from 0,1,2 to 2,0,1
  std::vector<int32_t> file_ids_;  // shuffle file names in each epoch
  // after shuffle
  // 0 : 17  -  shard 0 has 17 samples - pre shard 2
  // 1 : 32  -  shard 1 has 15 samples - pre shard 0
  // 2 : 58  -  shard 2 has 26 samples - pre shard 1
  std::vector<int64_t> shuffled_shard_sample_count_;
  // Assuming this is an 8-card training
  // card 0 : kCommonTask, 0, 0, 8
  // card 1 : kCommonTask, 0, 8, 16
  // card 2 : kCommonTask, 0, 16, 17
  // card 2 : kCommonTask, 1, 17, 24
  // card 3 : kCommonTask, 1, 24, 32
  // card 4 : kCommonTask, 2, 32, 40
  // card 5 : kCommonTask, 2, 40, 48
  // card 6 : kCommonTask, 2, 48, 56
  // card 7 : kCommonTask, 2, 56, 58
  // card 7 : kPaddedTask, -1, 58, 64
  std::vector<PartitionedShardSampleCount> partitioned_shard_sample_count_;
  // need shuffle the samples
  bool need_shuffle_;
  // the shuffle seed is from ShuffleOperator which is changed in every epoch
  uint32_t shuffle_seed_;
  // this can generator sample ids which are from partitioned_shard_sample_count_
  GeneratorIds generator_ids_;

  // load type: fast mode, lazy mode or slow mode
  LoadMode load_mode_;
};

inline void ShardTaskList::AssignTask(ShardTaskList &sourceTasks, int64_t id) {
  // Insert the sample id from the source into ourself by indexing at id position.
  // Important: The task list itself does not change.
  int64_t sample_id = sourceTasks.GetTaskSampleByID(id);
  MS_LOG(DEBUG) << "Insert sample id (" << sample_id << ") into task list from source task position: " << id;
  sample_ids_.push_back(sample_id);
}

inline void ShardTaskList::InsertTask(TaskType task_type, int shard_id, int group_id,
                                      const std::vector<uint64_t> &offset, const json &label) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << shard_id << ", group_id: " << group_id
                << ", label: " << label.dump() << ", size of task_list_: " << task_list_.size() << ".";
  (void)task_list_.emplace_back(task_type, std::make_tuple(shard_id, group_id));
  if (load_mode_ == LoadMode::kFast) {
    (void)sample_meta_list_.emplace_back(offset, label);
  }
}

inline void ShardTaskList::InsertTask(const int64_t &i, TaskType task_type, int shard_id, int group_id,
                                      const std::vector<uint64_t> &offset, const json &label) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << shard_id << ", group_id: " << group_id
                << ", label: " << label.dump() << ", size of task_list_: " << task_list_.size() << ".";
  task_list_[i] = {task_type, std::make_tuple(shard_id, group_id)};
  if (load_mode_ == LoadMode::kFast) {
    sample_meta_list_[i] = {offset, label};
  }
}

inline void ShardTaskList::InsertTask(ShardTask task) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << std::get<0>(std::get<1>(task))
                << ", group_id: " << std::get<1>(std::get<1>(task)) << ", label: " << std::get<3>(task).dump()
                << ", size of task_list_: " << task_list_.size() << ".";
  task_list_.push_back({std::get<0>(task), std::get<1>(task)});
  if (load_mode_ == LoadMode::kFast) {
    sample_meta_list_.push_back({std::get<2>(task), std::get<3>(task)});
  }
}

inline void ShardTaskList::InsertTask(const int64_t &i, ShardTask task) {
  task_list_[i] = {std::get<0>(task), std::get<1>(task)};
  if (load_mode_ == kFast) {
    sample_meta_list_[i] = {std::get<2>(task), std::get<3>(task)};
  }
}

inline void ShardTaskList::ResizeTask(const int64_t &size) {
  task_list_.resize(size);
  if (load_mode_ == kFast) {
    sample_meta_list_.resize(size);
  }
}
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_
