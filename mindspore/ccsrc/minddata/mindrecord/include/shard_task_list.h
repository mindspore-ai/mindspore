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

  int64_t SizeOfRows() const;

  ShardTask GetTaskByID(int64_t id);

  ShardTask GetRandomTask();

  int64_t GetTaskSampleByID(int64_t id);

  int64_t GetRandomTaskID();

  static ShardTaskList Combine(std::vector<ShardTaskList> &category_tasks, bool replacement,  // NOLINT
                               int64_t num_elements, int64_t num_samples);

  inline void ResizeTask(const int64_t &size);

  uint32_t categories;

  std::vector<int64_t> permutation_;  // A list of ints used for shuffling sample ids

  std::vector<int64_t> sample_ids_;  // The list of actual ids that were sampled

  // fast mode: [{TaskType, (shard_id, group_id(fast load))}, ...]
  // lazy mode: [{TaskType, (shard_id, sample_id(lazy load))}, ...]
  std::vector<TaskInfo> task_list_;

  // fast mode: [{[blob_start, blob_end], json}, ...]
  // lazy mode: none
  std::vector<SampleMeta> sample_meta_list_;

  // load type: fast mode or lazy mode
  bool lazy_load_;
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
  if (lazy_load_ == false) {
    sample_meta_list_.emplace_back(offset, label);
  }
}

inline void ShardTaskList::InsertTask(const int64_t &i, TaskType task_type, int shard_id, int group_id,
                                      const std::vector<uint64_t> &offset, const json &label) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << shard_id << ", group_id: " << group_id
                << ", label: " << label.dump() << ", size of task_list_: " << task_list_.size() << ".";
  task_list_[i] = {task_type, std::make_tuple(shard_id, group_id)};
  if (lazy_load_ == false) {
    sample_meta_list_[i] = {offset, label};
  }
}

inline void ShardTaskList::InsertTask(ShardTask task) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << std::get<0>(std::get<1>(task))
                << ", group_id: " << std::get<1>(std::get<1>(task)) << ", label: " << std::get<3>(task).dump()
                << ", size of task_list_: " << task_list_.size() << ".";
  task_list_.push_back({std::get<0>(task), std::get<1>(task)});
  if (lazy_load_ == false) {
    sample_meta_list_.push_back({std::get<2>(task), std::get<3>(task)});
  }
}

inline void ShardTaskList::InsertTask(const int64_t &i, ShardTask task) {
  task_list_[i] = {std::get<0>(task), std::get<1>(task)};
  if (lazy_load_ == false) {
    sample_meta_list_[i] = {std::get<2>(task), std::get<3>(task)};
  }
}

inline void ShardTaskList::ResizeTask(const int64_t &size) {
  task_list_.resize(size);
  if (lazy_load_ == false) {
    sample_meta_list_.resize(size);
  }
}
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_
