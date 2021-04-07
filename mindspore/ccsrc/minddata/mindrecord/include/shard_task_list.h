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

namespace mindspore {
namespace mindrecord {

// The data struct is as below:
// 1. TaskType: kCommonTask / kPaddedTask
// 2. std::tuple<int, int> : shard_id, group_id(fast load) / sample_id(lazy load)
// 3. std::vector<uint64_t>, json>> : [blob_start, blob_end], scalar_variable_fields
using ShardTask = std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json>;

class __attribute__((visibility("default"))) ShardTaskList {
 public:
  ShardTaskList();

  ShardTaskList(const ShardTaskList &task);  // copy construction

  ShardTaskList &operator=(const ShardTaskList &task);  // assignment operator

  ~ShardTaskList() = default;

  void InitSampleIds();

  static void TaskListSwap(ShardTaskList &orig_tasks, ShardTaskList &new_tasks);

  // Assigns the task based on task id
  inline void AssignTask(ShardTaskList &sourceTasks, size_t id);

  inline void InsertTask(TaskType task_type, int shard_id, int group_id, const std::vector<uint64_t> &offset,
                         const json &label);

  inline void InsertTask(const uint32_t &i, TaskType task_type, int shard_id, int group_id,
                         const std::vector<uint64_t> &offset, const json &label);

  inline void InsertTask(ShardTask task);

  inline void InsertTask(const uint32_t &i, ShardTask task);

  void MakePerm();

  inline void InsertSampleId(int id);

  void PopBack();

  uint32_t Size() const;

  uint32_t SizeOfRows() const;

  ShardTask &GetTaskByID(size_t id);

  ShardTask &GetRandomTask();

  int GetTaskSampleByID(size_t id);

  int GetRandomTaskID();

  static ShardTaskList Combine(std::vector<ShardTaskList> &category_tasks, bool replacement, int64_t num_elements,
                               int64_t num_samples);

  inline void ResizeTask(const uint32_t &size);

  uint32_t categories;

  std::vector<int> permutation_;  // A list of ints used for shuffling sample ids

  std::vector<int> sample_ids_;  // The list of actual ids that were sampled

  std::vector<ShardTask> task_list_;  // The full list of tasks
};

inline void ShardTaskList::AssignTask(ShardTaskList &sourceTasks, size_t id) {
  // Insert the sample id from the source into ourself by indexing at id position.
  // Important: The task list itself does not change.
  int sample_id = sourceTasks.GetTaskSampleByID(id);
  MS_LOG(DEBUG) << "Insert sample id (" << sample_id << ") into task list from source task position: " << id;
  sample_ids_.push_back(sample_id);
}

inline void ShardTaskList::InsertTask(TaskType task_type, int shard_id, int group_id,
                                      const std::vector<uint64_t> &offset, const json &label) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << shard_id << ", group_id: " << group_id
                << ", label: " << label.dump() << ", size of task_list_: " << task_list_.size() << ".";
  task_list_.emplace_back(task_type, std::make_tuple(shard_id, group_id), offset, label);
}

inline void ShardTaskList::InsertTask(const uint32_t &i, TaskType task_type, int shard_id, int group_id,
                                      const std::vector<uint64_t> &offset, const json &label) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << shard_id << ", group_id: " << group_id
                << ", label: " << label.dump() << ", size of task_list_: " << task_list_.size() << ".";
  task_list_[i] = {task_type, std::make_tuple(shard_id, group_id), offset, label};
}

inline void ShardTaskList::InsertTask(ShardTask task) {
  MS_LOG(DEBUG) << "Insert task into task list, shard_id: " << std::get<0>(std::get<1>(task))
                << ", group_id: " << std::get<1>(std::get<1>(task)) << ", label: " << std::get<3>(task).dump()
                << ", size of task_list_: " << task_list_.size() << ".";

  task_list_.push_back(std::move(task));
}

inline void ShardTaskList::InsertTask(const uint32_t &i, ShardTask task) { task_list_[i] = std::move(task); }

inline void ShardTaskList::ResizeTask(const uint32_t &size) { task_list_.resize(size); }
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_
