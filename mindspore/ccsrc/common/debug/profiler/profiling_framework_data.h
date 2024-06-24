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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_FRAMEWORK_DATA_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_FRAMEWORK_DATA_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <utility>
#include "kernel/kernel.h"
#include "common/debug/profiler/profiling_data_dumper.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace profiler {
namespace ascend {
using mindspore::runtime::kProfilerEventString;
using mindspore::runtime::kProfilerModuleString;
using mindspore::runtime::kProfilerStageString;
using mindspore::runtime::ProfilerData;

enum class COMMON_EXPORT OpRangeDataType {
  OP_RANGE_DATA = 1,
  IS_ASYNC = 2,
  NAME = 3,
  INPUT_DTYPES = 4,
  INPUT_SHAPE = 5,
  STACK = 6,
  MODULE_HIERARCHY = 7,
  EXTRA_ARGS = 8,
  RESERVED = 30,
};

struct COMMON_EXPORT OpRangeData : BaseReportData {
  int64_t start_ns{0};
  int64_t end_ns{0};
  int64_t sequence_number{0};
  uint64_t process_id{0};
  uint64_t start_thread_id{0};
  uint64_t end_thread_id{0};
  uint64_t forward_thread_id{0};
  bool is_async{false};
  std::string name;
  std::vector<std::string> input_dtypes;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::string> stack;
  std::vector<std::string> module_hierarchy;
  uint64_t flow_id{0};
  uint64_t step{0};
  // std::unordered_map<std::string, c10::IValue> extra_args;
  OpRangeData(int64_t start_ns, int64_t end_ns, int64_t sequence_number, uint64_t process_id, uint64_t start_thread_id,
              uint64_t end_thread_id, uint64_t forward_thread_id, bool is_async, std::string name,
              std::vector<std::string> stack, uint64_t flow_id, int32_t device_id, uint64_t step)
      : BaseReportData(device_id, "op_range_" + std::to_string(device_id)),
        start_ns(start_ns),
        end_ns(end_ns),
        sequence_number(sequence_number),
        process_id(process_id),
        start_thread_id(start_thread_id),
        end_thread_id(end_thread_id),
        forward_thread_id(forward_thread_id),
        is_async(is_async),
        name(std::move(name)),
        stack(std::move(stack)),
        flow_id(flow_id),
        step(step) {}
  std::vector<uint8_t> encode();
  void preprocess();
};

class COMMON_EXPORT ProfilingFrameworkData {
 public:
  static void RecordHostProfile(std::shared_ptr<ProfilerData> data, uint64_t step = 0);

  inline static std::map<std::string, uint64_t> kernel_launch_begin_;
  inline static int32_t Device_Id = 0;
  inline static bool added = false;
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_FRAMEWORK_DATA_H_
