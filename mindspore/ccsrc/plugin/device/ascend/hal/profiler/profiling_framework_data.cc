/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/profiler/profiling_framework_data.h"
#include <sys/syscall.h>
#include <utility>
#include <algorithm>
#include <mutex>
#include "ir/dtype.h"
#include "ops/structure_op_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "nlohmann/json.hpp"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace profiler {
namespace ascend {

uint64_t getClockSyscnt() {
  uint64_t cycles;
#if defined(__aarch64__)
  asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#elif defined(__x86_64__)
  constexpr uint32_t uint32Bits = 32U;
  uint32_t hi = 0;
  uint32_t lo = 0;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  cycles = (static_cast<uint64_t>(lo)) | ((static_cast<uint64_t>(hi)) << uint32Bits);
#elif defined(__arm__)
  const uint32_t uint32Bits = 32U;
  uint32_t hi = 0;
  uint32_t lo = 0;
  asm volatile("mrrc p15, 1, %0, %1, c14" : "=r"(lo), "=r"(hi));
  cycles = (static_cast<uint64_t>(lo)) | ((static_cast<uint64_t>(hi)) << uint32Bits);
#else
  cycles = 0;
#endif
  return cycles;
}

uint64_t GetClockMonotonicRawNs() {
  struct timespec ts = {0};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000 +
         static_cast<uint64_t>(ts.tv_nsec);  // 1000000000为秒转换为纳秒的倍数
}

inline void encodeStrData(uint16_t type, const std::string &data, const std::unique_ptr<std::vector<uint8_t>> &result) {
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    result->push_back((type >> (i * 8)) & 0xff);
  }
  uint32_t length = data.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    result->push_back((length >> (i * 8)) & 0xff);
  }
  result->insert(result->end(), data.begin(), data.end());
}

template <typename T>
void encodeFixedData(const std::vector<T> &data_list, const std::unique_ptr<std::vector<uint8_t>> &result) {
  for (auto data : data_list) {
    for (size_t i = 0; i < sizeof(T); ++i) {
      result->push_back((static_cast<size_t>(data) >> (i * 8)) & 0xff);
    }
  }
}

template <typename T>
void encode2DIntegerMatrixDatas(const uint16_t type, const std::vector<std::vector<T>> &data_list,
                                const std::unique_ptr<std::vector<uint8_t>> &result) {
  std::string rst;
  for (auto tensor : data_list) {
    std::stringstream ss;
    copy(tensor.begin(), tensor.end(), std::ostream_iterator<T>(ss, ","));
    std::string str = ss.str();
    if (!str.empty()) {
      str.pop_back();
    }
    rst += (str + ";");
  }
  if (!rst.empty()) {
    rst.pop_back();
  }
  encodeStrData(type, rst, result);
}

inline void encodeStrArrayData(const uint16_t type, const std::vector<std::string> &data_list,
                               const std::unique_ptr<std::vector<uint8_t>> &result) {
  std::string rst = std::accumulate(data_list.begin(), data_list.end(), std::string(""),
                                    [](const std::string &r, const auto d) { return r + ';' + d; });
  if (!rst.empty()) {
    rst.pop_back();
  }
  encodeStrData(type, rst, result);
}

std::vector<uint8_t> OpRangeData::encode() {
  std::unique_ptr<std::vector<uint8_t>> result = std::make_unique<std::vector<uint8_t>>();
  encodeFixedData<int64_t>({start_ns, end_ns, sequence_number}, result);
  encodeFixedData<uint64_t>({process_id, start_thread_id, end_thread_id, forward_thread_id}, result);
  result->push_back(is_async);
  encodeStrData(static_cast<uint16_t>(OpRangeDataType::NAME), name, result);
  if (!input_dtypes.empty()) {
    encodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::INPUT_DTYPES), input_dtypes, result);
  }
  if (!input_shapes.empty()) {
    encode2DIntegerMatrixDatas<int64_t>(static_cast<uint16_t>(OpRangeDataType::INPUT_SHAPE), input_shapes, result);
  }
  if (!stack.empty()) {
    encodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::STACK), stack, result);
  }
  if (!module_hierarchy.empty()) {
    encodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::MODULE_HIERARCHY), module_hierarchy, result);
  }
  std::vector<uint8_t> resultTLV;
  uint16_t dataType = static_cast<uint16_t>(OpRangeDataType::OP_RANGE_DATA);
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    resultTLV.push_back((dataType >> (i * 8)) & 0xff);
  }
  uint32_t length = result->size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    resultTLV.push_back((length >> (i * 8)) & 0xff);
  }
  resultTLV.insert(resultTLV.end(), result->cbegin(), result->cend());
  return std::move(resultTLV);
}

void ProfilingFrameworkData::RecordLaunchGETaskBegin(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto ascend_profiler = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  if (!ascend_profiler->GetEnableFlag()) {
    return;
  }

  int64_t start_ns = getClockSyscnt();
  auto tid = syscall(SYS_gettid);
  kernel_launch_begin_[std::to_string(tid) + "_" + node->fullname_with_scope()] = start_ns;
}

void ProfilingFrameworkData::RecordGETask(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto ascend_profiler = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  if (!ascend_profiler->GetEnableFlag()) {
    return;
  }

  auto tid = syscall(SYS_gettid);
  std::string full_scope_name = node->fullname_with_scope();
  auto iter = kernel_launch_begin_.find(std::to_string(tid) + "_" + full_scope_name);
  if (iter == kernel_launch_begin_.end()) {
    MS_LOG(WARNING) << "Do not find op info: " << full_scope_name;
    return;
  }
  int64_t start_ns = iter->second;
  int64_t end_ns = getClockSyscnt();
  int64_t sequence_number = 0;
  uint64_t process_id = getpid();
  uint64_t start_thread_id = static_cast<uint64_t>(tid);
  uint64_t end_thread_id = start_thread_id;
  uint64_t forward_thread_id = start_thread_id;
  bool is_async = false;

  OpRangeData report = OpRangeData(start_ns, end_ns, sequence_number, process_id, start_thread_id, end_thread_id,
                                   forward_thread_id, is_async, full_scope_name, ProfilingFrameworkData::Device_Id);
  size_t total_size = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t index = 0U; index < total_size; index++) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(node, index);
    auto input_node = input_node_with_index.first;
    auto input_index = input_node_with_index.second;
    ShapeVector shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    report.input_shapes.push_back(shape);

    TypeId type_id = AnfAlgo::GetOutputDeviceDataType(input_node, input_index);
    std::string type = TypeIdToString(type_id, true);
    report.input_dtypes.push_back(type);
  }

  ProfilingDataDumper::GetInstance()->Report(std::make_unique<OpRangeData>(report));
}

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
