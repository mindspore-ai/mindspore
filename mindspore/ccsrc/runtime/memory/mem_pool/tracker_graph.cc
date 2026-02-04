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

#include "runtime/memory/mem_pool/tracker_graph.h"

#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>
#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>

#include "runtime/memory/mem_pool/race_checker.h"
#include "include/runtime/memory/mem_pool/dynamic_mem_pool.h"
#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/distributed_meta.h"
#include "include/utils/common.h"
#include "device_address/device_type.h"
#include "include/runtime/memory/mem_pool/mem_dynamic_allocator.h"
#include "include/utils/utils.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace device {
namespace tracker {
namespace graph {
namespace {
std::string SerializeMap(const std::unordered_map<std::string, std::string> &map,
                         const std::string &pair_separator = ";", const std::string &key_value_separator = "=") {
  std::ostringstream oss;
  bool first = true;
  for (const auto &pair : map) {
    if (!first) {
      oss << pair_separator;
    }
    oss << pair.first << key_value_separator << pair.second;
    first = false;
  }
  return oss.str();
}

size_t GetTaskInfoStreamId(const TaskInfoPtr &task_info) {
  auto iter = task_info->attrs.find(kStreamId);
  if (iter == task_info->attrs.end()) {
    MS_LOG(ERROR) << "Stream id is not found , task info: " << task_info->time_stamp;
    return 0;
  }
  return std::stoul(iter->second);
}

constexpr size_t kLogThreshold = 10;
constexpr size_t kLogPersentage = 100;
}  // namespace

std::string TrackerTensor::ToString() { return "%" + std::to_string(start_time_stamp); }

std::string TrackerTensor::DtypeToString() { return TypeIdToString(dtype); }

std::string TrackerTensor::ShapeToString() { return "[" + ShapeVectorToString(shape) + "]"; }

std::string TrackerTensor::TensorInfoToString() {
  std::ostringstream oss;
  oss << DtypeToString() << ":" << ShapeToString();
  if (tensor_info != nullptr) {
    oss << "{";
    oss << "strdes=" << VectorToString(tensor_info->strides) << ",";
    oss << "offset=" << tensor_info->storage_offset;
    oss << "}";
  }
  return oss.str();
}

std::string TrackerOperator::name() {
  MS_EXCEPTION_IF_NULL(task_info);
  return task_info->node_name;
}

std::string TrackerOperator::ToString() {
  MS_EXCEPTION_IF_NULL(task_info);
  std::ostringstream oss;

  oss << "(";
  for (size_t i = 0; i < outputs.size(); i++) {
    oss << outputs[i]->ToString();
    if (i != outputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << ") = " << task_info->node_name << "(";
  for (size_t i = 0; i < inputs.size(); i++) {
    oss << inputs[i]->ToString();
    if (i != inputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << "), task_info: " << task_info->time_stamp << ", ";
  oss << "attrs {" << SerializeMap(task_info->attrs) << "} \n";

  oss << "    (";
  for (size_t i = 0; i < outputs.size(); i++) {
    oss << outputs[i]->TensorInfoToString();
    if (i != outputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << ") <- (";
  for (size_t i = 0; i < inputs.size(); i++) {
    oss << inputs[i]->TensorInfoToString();
    if (i != inputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << ")\n";
  oss << "    # " << task_info->python_stack;
  std::string str = oss.str();
  return str;
}

void TrackerGraph::RaceCheck() {
  MS_LOG(WARNING) << "Begin race check";
  auto race_checker = RaceChecker(GetAllAddresses(), GetStreamSize());
  for (const auto &op : operators_) {
    MS_EXCEPTION_IF_NULL(op);
    MS_EXCEPTION_IF_NULL(op->task_info);
    auto stream_id = GetTaskInfoStreamId(op->task_info);
    if (IsEvent(op->task_info, "RecordEvent")) {
      auto iter = op->task_info->attrs.find(kEvent);
      if (iter == op->task_info->attrs.end()) {
        MS_LOG(ERROR) << "Event id is not found, task info: " << op->task_info->time_stamp;
        continue;
      }
      race_checker.RecordEvent(stream_id, iter->second);
    } else if (IsEvent(op->task_info, "WaitEvent")) {
      auto iter = op->task_info->attrs.find(kEvent);
      if (iter == op->task_info->attrs.end()) {
        MS_LOG(ERROR) << "Event id is not found, task info: " << op->task_info->time_stamp;
        continue;
      }
      race_checker.WaitEvent(stream_id, iter->second);
    } else {
      // check read and write
      if (NeedSkipRaceCheck(op->task_info)) {
        continue;
      }
      // not need to check side effect, because assign output and input are the same address.
      for (size_t i = 0; i < op->inputs.size(); i++) {
        auto input = op->inputs[i];
        MS_EXCEPTION_IF_NULL(input);
        bool read_error = race_checker.CheckRead(input->start_addr, input->end_addr, stream_id);
        if (read_error) {
          MS_LOG(ERROR) << "Read error, task info: " << op->task_info->time_stamp;
        }
      }
      for (size_t i = 0; i < op->outputs.size(); i++) {
        auto output = op->outputs[i];
        MS_EXCEPTION_IF_NULL(output);
        bool write_error = race_checker.CheckWrite(output->start_addr, output->end_addr, stream_id);
        if (write_error) {
          MS_LOG(ERROR) << "Write error, task info: " << op->task_info->time_stamp;
        }
      }
    }
  }
  MS_LOG(WARNING) << "End race check";
}

void TrackerGraph::Dump(const std::string &graph_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  RaceCheck();

  static bool is_simple_tracker = memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocSimpleTracker);
  if (is_simple_tracker) {
    MS_LOG(WARNING) << "Simple tracker, skip dump";
    return;
  }
  MS_LOG(WARNING) << "Dump graph to file: " << graph_path;
  std::ofstream graph_file(graph_path);
  // dump operators
  size_t log_threshold = operators_.size() / kLogThreshold;
  size_t i = 0;
  for (const auto &op : operators_) {
    i++;
    MS_EXCEPTION_IF_NULL(op);
    graph_file << op->ToString() << std::endl;
    // print log
    if (i > log_threshold) {
      MS_LOG(WARNING) << "TrackerGraph Dump progress: " << (i + 1) * kLogPersentage / operators_.size() << "%";
      log_threshold += operators_.size() / kLogThreshold;
    }
  }
  graph_file.close();
}

bool TrackerGraph::NeedDump() {
  std::lock_guard<std::mutex> lock(mutex_);
  return !operators_.empty();
}

TrackerTensorPtr TrackerGraph::AddTensor(MemBlockInfoPtr mem_block, DeviceMemPtr device_ptr, TypeId dtype,
                                         const ShapeVector &shape, TensorStorageInfoPtr tensor_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddTensor is not allowed when race check is enabled";
  }
  auto tensor = std::make_shared<TrackerTensor>();
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(mem_block);
  tensor->start_time_stamp = mem_block->start_time_stamp;
  // Strict validation, considering the entire memory block as unsafe, without considering offset and stride for now
  tensor->start_addr = reinterpret_cast<uintptr_t>(mem_block->device_addr);
  tensor->end_addr = tensor->start_addr + mem_block->size - 1;  // -1 for end_addr is exclusive
  tensor->shape = shape;
  tensor->dtype = dtype;
  tensor->tensor_info = tensor_info;
  tensors_.push_back(tensor);
  return tensor;
}

void TrackerGraph::AddOperator(TaskInfoPtr task_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddOperator is not allowed when race check is enabled";
  }
  auto op = std::make_shared<TrackerOperator>();
  MS_EXCEPTION_IF_NULL(op);
  op->task_info = task_info;
  op->stream_id = GetTaskInfoStreamId(task_info);
  operators_.push_back(op);
  task_operator_map_[task_info] = op;
}

void TrackerGraph::CacheLastTask() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (operators_.empty()) {
    cache_ = nullptr;
    return;
  }
  cache_ = operators_.back();
  operators_.pop_back();
}

void TrackerGraph::EmptyCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (cache_ == nullptr) {
    return;
  }
  operators_.push_back(cache_);
  cache_ = nullptr;
}

void TrackerGraph::AddOperatorInput(TaskInfoPtr task_info, TrackerTensorPtr tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddOperatorInput is not allowed when race check is enabled";
  }
  auto iter = task_operator_map_.find(task_info);
  if (iter == task_operator_map_.end()) {
    MS_LOG(EXCEPTION) << "Task info " << task_info->time_stamp << " not found";
  }
  auto op = iter->second;
  op->inputs.push_back(tensor);
}

void TrackerGraph::AddOperatorOutput(TaskInfoPtr task_info, TrackerTensorPtr tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddOperatorOutput is not allowed when race check is enabled";
  }
  auto iter = task_operator_map_.find(task_info);
  if (iter == task_operator_map_.end()) {
    MS_LOG(EXCEPTION) << "Task info " << task_info->time_stamp << " not found";
  }
  auto op = iter->second;
  op->outputs.push_back(tensor);
}

// for race checker
std::vector<uintptr_t> TrackerGraph::GetAllAddresses() {
  begin_race_check_ = true;
  std::vector<uintptr_t> addresses;
  for (const auto &tensor : tensors_) {
    addresses.push_back(tensor->start_addr);
    addresses.push_back(tensor->end_addr);
  }
  return addresses;
}

int32_t TrackerGraph::GetStreamSize() {
  begin_race_check_ = true;
  size_t stream_num = 0;
  for (const auto &op : operators_) {
    MS_EXCEPTION_IF_NULL(op);
    MS_EXCEPTION_IF_NULL(op->task_info);
    size_t stream_id = GetTaskInfoStreamId(op->task_info);
    stream_num = std::max(stream_num, stream_id);
  }
  return stream_num + 1;
}

namespace {
bool MatchOp(const std::string_view &src, const std::string_view &opname) {
  // Find the starting position of the operator name
  size_t op_start = src.rfind(opname);
  if (op_start == std::string_view::npos) {
    return false;
  }

  // Check if there is a delimiter before the operator name.
  if (op_start > 0 && src[op_start - 1] != '/') {
    return false;
  }

  // Check if there is a valid suffix after the operator name.
  size_t op_end = op_start + opname.length() - 1;
  if (op_end < src.length() - 1) {
    // Check if the suffix is "-op" followed by digits
    const size_t kSuffixLength = 4;
    if (op_end + kSuffixLength > src.length() - 1) {
      return false;
    }

    const std::string_view kExpectedSuffix = std::string_view{"-op"};
    if (src.substr(op_end + 1, kExpectedSuffix.length()).compare(0, kExpectedSuffix.length(), kExpectedSuffix) != 0) {
      return false;
    }

    // Check whether the remaining characters are digits.
    for (size_t i = op_end + kSuffixLength; i < src.length(); ++i) {
      if (!std::isdigit(src[i])) {
        return false;
      }
    }
  }

  return true;
}
}  // namespace

bool NeedSkipRaceCheck(const TaskInfoPtr &task_info) {
  return task_info == nullptr || MatchOp(task_info->node_name, "Reshape");
}

bool IsEvent(const TaskInfoPtr &task_info, const std::string &type) {
  const std::string exact_name = (type == "RecordEvent") ? "RecordEvent" : "WaitEvent";
  const std::string op_name = (type == "RecordEvent") ? "StreamSend" : "StreamRecv";

  return (task_info->node_name == exact_name) || MatchOp(task_info->node_name, op_name);
}
}  // namespace graph
}  // namespace tracker
}  // namespace device
}  // namespace mindspore
