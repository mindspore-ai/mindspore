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
#include "common/debug/profiler/profiling_framework_data.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <sys/syscall.h>
#endif
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

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
uint64_t GetClockMonotonicRawNs() {
  struct timespec ts = {0};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000 +
         static_cast<uint64_t>(ts.tv_nsec);  // 1000000000为秒转换为纳秒的倍数
}
#else
uint64_t GetClockMonotonicRawNs() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
#endif

inline void EncodeStrData(uint16_t type, const std::string &data, const std::unique_ptr<std::vector<uint8_t>> &result) {
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
void EncodeFixedData(const std::vector<T> &data_list, const std::unique_ptr<std::vector<uint8_t>> &result) {
  for (auto data : data_list) {
    for (size_t i = 0; i < sizeof(T); ++i) {
      result->push_back((static_cast<size_t>(data) >> (i * 8)) & 0xff);
    }
  }
}

template <typename T>
void Encode2DIntegerMatrixDatas(const uint16_t type, const std::vector<std::vector<T>> &data_list,
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
  EncodeStrData(type, rst, result);
}

inline void EncodeStrArrayData(const uint16_t type, const std::vector<std::string> &data_list,
                               const std::unique_ptr<std::vector<uint8_t>> &result) {
  std::string rst = std::accumulate(data_list.begin(), data_list.end(), std::string(""),
                                    [](const std::string &r, const auto d) { return r + ';' + d; });
  if (!rst.empty()) {
    rst.pop_back();
  }
  EncodeStrData(type, rst, result);
}

void OpRangeData::preprocess() {
  const std::string delim = "|";
  const std::string remove_ms = "site-packages/mindspore";
  if (stack.size() > 0 && !stack[0].empty()) {
    std::string all_stack = stack[0];
    stack.erase(stack.begin());
    size_t nPos = all_stack.find(delim.c_str());
    while (nPos != std::string::npos) {
      std::string temp = all_stack.substr(0, nPos + 1);
      if (temp.find(remove_ms.c_str()) == std::string::npos) {
        stack.push_back(temp);
      }
      all_stack = all_stack.substr(nPos + 1);
      nPos = all_stack.find(delim.c_str());
    }
  }
}

std::vector<uint8_t> OpRangeData::encode() {
  preprocess();
  std::unique_ptr<std::vector<uint8_t>> result = std::make_unique<std::vector<uint8_t>>();
  EncodeFixedData<int64_t>({start_ns, end_ns, sequence_number}, result);
  EncodeFixedData<uint64_t>({process_id, start_thread_id, end_thread_id, forward_thread_id}, result);
  EncodeFixedData<uint64_t>({flow_id}, result);
  EncodeFixedData<uint64_t>({step}, result);
  result->push_back(is_async);
  EncodeStrData(static_cast<uint16_t>(OpRangeDataType::NAME), name, result);
  if (!input_dtypes.empty()) {
    EncodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::INPUT_DTYPES), input_dtypes, result);
  }
  if (!input_shapes.empty()) {
    Encode2DIntegerMatrixDatas<int64_t>(static_cast<uint16_t>(OpRangeDataType::INPUT_SHAPE), input_shapes, result);
  }
  if (!stack.empty()) {
    EncodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::STACK), stack, result);
  }
  if (!module_hierarchy.empty()) {
    EncodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::MODULE_HIERARCHY), module_hierarchy, result);
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
  return resultTLV;
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
void ProfilingFrameworkData::RecordHostProfile(std::shared_ptr<ProfilerData> data, uint64_t step) {
  auto ascend_profiler = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  if (!ascend_profiler->EnableHostStack()) {
    return;
  }
  std::vector<std::string> stack_vec;
  stack_vec.push_back(data->py_stack_);
  std::string op_name = data->op_name_;
  if (data->is_stage_) {
    op_name = kProfilerStageString.at(data->stage_);
  } else if (data->op_name_ != "flow") {
    op_name = kProfilerModuleString.at(data->module_) + "::" + kProfilerEventString.at(data->event_) + "::" + op_name;
  }
  OpRangeData report =
    OpRangeData(data->start_time_, data->end_time_, 0, 0, data->tid_, data->tid_, data->tid_, false, op_name,
                std::move(stack_vec), data->flow_id_, ProfilingFrameworkData::Device_Id, step);
  ProfilingDataDumper::GetInstance().Report(std::make_unique<OpRangeData>(report));
}
#else
void ProfilingFrameworkData::RecordHostProfile(std::shared_ptr<ProfilerData> data, uint64_t step) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
#endif
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
