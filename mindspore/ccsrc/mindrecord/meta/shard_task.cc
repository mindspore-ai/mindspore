/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "mindrecord/include/shard_task.h"
#include "common/utils.h"
#include "mindrecord/include/common/shard_utils.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::DEBUG;

namespace mindspore {
namespace mindrecord {
void ShardTask::MakePerm() {
  permutation_ = std::vector<int>(task_list_.size());
  for (uint32_t i = 0; i < task_list_.size(); i++) {
    permutation_[i] = static_cast<int>(i);
  }
}

void ShardTask::InsertTask(TaskType task_type, int shard_id, int group_id, const std::vector<uint64_t> &offset,
                           const json &label) {
  MS_LOG(DEBUG) << "Into insert task, shard_id: " << shard_id << ", group_id: " << group_id
                << ", label: " << label.dump() << ", size of task_list_: " << task_list_.size() << ".";
  task_list_.emplace_back(task_type, std::make_tuple(shard_id, group_id), offset, label);
}

void ShardTask::InsertTask(std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> task) {
  MS_LOG(DEBUG) << "Into insert task, shard_id: " << std::get<0>(std::get<1>(task))
                << ", group_id: " << std::get<1>(std::get<1>(task)) << ", label: " << std::get<3>(task).dump()
                << ", size of task_list_: " << task_list_.size() << ".";

  task_list_.push_back(std::move(task));
}

void ShardTask::PopBack() { task_list_.pop_back(); }

uint32_t ShardTask::Size() const { return static_cast<uint32_t>(task_list_.size()); }

uint32_t ShardTask::SizeOfRows() const {
  if (task_list_.size() == 0) return static_cast<uint32_t>(0);

  // 1 task is 1 page
  auto sum_num_rows = [](int x, std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> y) {
    return x + std::get<2>(y)[0];
  };
  uint32_t nRows = std::accumulate(task_list_.begin(), task_list_.end(), 0, sum_num_rows);
  return nRows;
}

std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> &ShardTask::GetTaskByID(size_t id) {
  MS_ASSERT(id < task_list_.size());
  return task_list_[id];
}

std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> &ShardTask::GetRandomTask() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, task_list_.size() - 1);
  return task_list_[dis(gen)];
}

ShardTask ShardTask::Combine(std::vector<ShardTask> &category_tasks, bool replacement, int64_t num_elements) {
  ShardTask res;
  if (category_tasks.empty()) return res;
  auto total_categories = category_tasks.size();
  res.categories = static_cast<uint32_t>(total_categories);
  if (replacement == false) {
    auto minTasks = category_tasks[0].Size();
    for (uint32_t i = 1; i < total_categories; i++) {
      minTasks = std::min(minTasks, category_tasks[i].Size());
    }
    for (uint32_t task_no = 0; task_no < minTasks; task_no++) {
      for (uint32_t i = 0; i < total_categories; i++) {
        res.InsertTask(std::move(category_tasks[i].GetTaskByID(static_cast<int>(task_no))));
      }
    }
  } else {
    auto maxTasks = category_tasks[0].Size();
    for (uint32_t i = 1; i < total_categories; i++) {
      maxTasks = std::max(maxTasks, category_tasks[i].Size());
    }
    if (num_elements != std::numeric_limits<int64_t>::max()) {
      maxTasks = static_cast<decltype(maxTasks)>(num_elements);
    }
    for (uint32_t i = 0; i < total_categories; i++) {
      for (uint32_t j = 0; j < maxTasks; j++) {
        res.InsertTask(category_tasks[i].GetRandomTask());
      }
    }
  }
  return res;
}
}  // namespace mindrecord
}  // namespace mindspore
