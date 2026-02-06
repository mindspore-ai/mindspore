/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_TRACKER_GRAPH_H_
#define MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_TRACKER_GRAPH_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/runtime/memory/mem_pool/mem_tracker.h"

namespace mindspore {
namespace device {
namespace tracker {
namespace graph {
struct TrackerTensor {
  std::string ToString();
  std::string DtypeToString();
  std::string ShapeToString();
  std::string TensorInfoToString();
  int64_t start_time_stamp;
  TensorStorageInfoPtr tensor_info;
  ShapeVector shape;
  TypeId dtype;
  uintptr_t start_addr;
  uintptr_t end_addr;
};
using TrackerTensorPtr = std::shared_ptr<TrackerTensor>;

struct TrackerOperator {
  std::vector<TrackerTensorPtr> inputs;
  std::vector<TrackerTensorPtr> outputs;
  std::string ToString();
  std::string name();
  TaskInfoPtr task_info;
  size_t stream_id;
};
using TrackerOperatorPtr = std::shared_ptr<TrackerOperator>;

class TrackerGraph {
 public:
  static TrackerGraph &getInstance() {
    static TrackerGraph instance;
    return instance;
  }
  TrackerGraph(const TrackerGraph &) = delete;
  TrackerGraph &operator=(const TrackerGraph &) = delete;
  TrackerTensorPtr AddTensor(MemBlockInfoPtr mem_block, DeviceMemPtr device_ptr, TypeId dtype, const ShapeVector &shape,
                             TensorStorageInfoPtr tensor_info);
  void AddOperator(TaskInfoPtr task_info);
  void AddOperatorInput(TaskInfoPtr task_info, TrackerTensorPtr tensor);
  void AddOperatorOutput(TaskInfoPtr task_info, TrackerTensorPtr tensor);
  void Dump(const std::string &graph_path);
  bool NeedDump();
  // For some special tracker scenarios, need to cache the last task
  void CacheLastTask();
  void EmptyCache();

 private:
  TrackerGraph() = default;
  std::mutex mutex_;
  bool begin_race_check_ = false;
  std::vector<TrackerTensorPtr> tensors_;
  std::vector<TrackerOperatorPtr> operators_;
  std::unordered_map<TaskInfoPtr, TrackerOperatorPtr> task_operator_map_;
  TrackerOperatorPtr cache_ = nullptr;
  // for race checker
  void RaceCheck();
  std::vector<uintptr_t> GetAllAddresses();
  int32_t GetStreamSize();
};

bool NeedSkipRaceCheck(const TaskInfoPtr &task_info);

bool IsEvent(const TaskInfoPtr &task_info, const std::string &type);
}  // namespace graph
}  // namespace tracker
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_TRACKER_GRAPH_H_
