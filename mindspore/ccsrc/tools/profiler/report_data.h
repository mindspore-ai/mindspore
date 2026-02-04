/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_REPORT_DATA_H
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_REPORT_DATA_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <numeric>
#include <utility>
#include "include/utils/visible.h"
namespace mindspore {
namespace profiler {
namespace ascend {

constexpr uint8_t kBitsPerByte = 8;
constexpr uint8_t kBytesMask = 0xff;

enum class PROFILER_EXPORT OpRangeDataType : uint8_t {
  THREAD_ID = 0,
  FLOW_ID = 1,
  STEP = 2,
  START_TIME_NS = 3,
  END_TIME_NS = 4,
  PROCESS_ID = 5,
  MODULE_INDEX = 6,
  EVENT_INDEX = 7,
  STAGE_INDEX = 8,
  LEVEL = 9,
  IS_GRAPH_DATA = 10,
  IS_STAGE = 11,
  IS_STACK = 12,
  NAME = 13,
  FULL_NAME = 14,
  MODULE_GRAPH = 15,
  EVENT_GRAPH = 16,
  CUSTOM_INFO = 17,
};

enum class PROFILER_EXPORT RecordShapesDataType : uint8_t {
  NAME = 0,
  INPUT_SHAPES = 1,
  INPUT_TYPE = 2,
};

enum class PROFILER_EXPORT ReportFileType : uint32_t {
  OP_RANGE = 0,
  PYTHON_STACK = 1,
  MEMORY_USAGE = 2,
  RECORD_SHAPES = 3,
};

static const std::map<ReportFileType, std::string> kReportFileTypeMap = {
  {ReportFileType::OP_RANGE, "mindspore.op_range"},
  {ReportFileType::PYTHON_STACK, "mindspore.py_stack"},
  {ReportFileType::MEMORY_USAGE, "mindspore.memory_usage"},
  {ReportFileType::RECORD_SHAPES, "mindspore.record_shapes"},
};

struct PROFILER_EXPORT BaseReportData {
  // 1 * 4 bytes = 4 bytes
  int32_t device_id{0};
  // 1 * 4 bytes = 4 bytes
  uint32_t tag;
  BaseReportData(int32_t device_id, uint32_t tag) : device_id(device_id), tag(tag) {}
  virtual ~BaseReportData() = default;
  virtual std::vector<uint8_t> encode() = 0;
};

struct PROFILER_EXPORT OpRangeData : BaseReportData {
  // 5 * 8 bytes = 40 bytes
  uint64_t thread_id{0};
  uint64_t flow_id{0};
  uint64_t step{0};
  uint64_t start_time_ns{0};
  uint64_t end_time_ns{0};

  // 1 * 4 bytes = 4 bytes
  int32_t process_id{0};

  // 3 * 2 bytes = 6 bytes
  uint16_t module_index{0};
  uint16_t event_index{0};
  uint16_t stage_index{0};

  // 1 * 1 bytes = 1 bytes
  int8_t level{-1};
  bool is_graph_data{false};
  bool is_stage{false};
  bool is_stack{false};

  // dynamic length
  std::string op_name;
  std::string op_full_name;
  std::string module_graph;
  std::string event_graph;
  std::map<std::string, std::string> custom_info{};

  OpRangeData(int32_t device_id, uint64_t thread_id, uint64_t flow_id, uint64_t step, uint64_t start_time_ns,
              uint64_t end_time_ns, int32_t process_id, uint16_t module_index, uint16_t event_index,
              uint16_t stage_index, int8_t level, bool is_graph_data, bool is_stage, const std::string &op_name,
              const std::string &op_full_name, const std::string &module_graph, const std::string &event_graph,
              const std::map<std::string, std::string> &custom_info)
      : BaseReportData(device_id, static_cast<uint32_t>(ReportFileType::OP_RANGE)),
        thread_id(thread_id),
        flow_id(flow_id),
        step(step),
        start_time_ns(start_time_ns),
        end_time_ns(end_time_ns),
        process_id(process_id),
        module_index(module_index),
        event_index(event_index),
        stage_index(stage_index),
        level(level),
        is_graph_data(is_graph_data),
        is_stage(is_stage),
        is_stack(false),
        op_name(std::move(op_name)),
        op_full_name(std::move(op_full_name)),
        module_graph(std::move(module_graph)),
        event_graph(std::move(event_graph)),
        custom_info(custom_info) {}

  // temporary constructor for python stack
  OpRangeData(uint64_t thread_id, uint64_t start_time_ns, uint64_t end_time_ns, const std::string &op_name,
              int32_t device_id)
      : BaseReportData(device_id, static_cast<uint32_t>(ReportFileType::OP_RANGE)),
        thread_id(thread_id),
        start_time_ns(start_time_ns),
        end_time_ns(end_time_ns),
        is_stack(true),
        op_name(std::move(op_name)) {}

  std::vector<uint8_t> encode() override;
};

struct PROFILER_EXPORT RecordShapesData : BaseReportData {
  // dynamic length
  std::string op_name;
  std::string input_shapes;
  std::string input_type;

  RecordShapesData(int32_t device_id, std::string op_name, std::string input_shapes, std::string input_type)
      : BaseReportData(device_id, static_cast<uint32_t>(ReportFileType::RECORD_SHAPES)),
        op_name(std::move(op_name)),
        input_shapes(std::move(input_shapes)),
        input_type(std::move(input_type)) {}

  std::vector<uint8_t> encode() override;
};

template <typename T>
inline void encodeFixedData(const T &data, std::vector<uint8_t> &result) {  // NOLINT(runtime/references)
  for (size_t i = 0; i < sizeof(T); ++i) {
    result.emplace_back((static_cast<size_t>(data) >> (i * kBitsPerByte)) & kBytesMask);
  }
}

inline void encodeStrData(uint16_t type, const std::string &data,
                          std::vector<uint8_t> &result) {  // NOLINT(runtime/references)
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    result.emplace_back((type >> (i * kBitsPerByte)) & kBytesMask);
  }
  uint32_t length = data.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    result.emplace_back((length >> (i * kBitsPerByte)) & kBytesMask);
  }
  for (const auto &c : data) {
    result.emplace_back(c);
  }
}

inline void encodeStrMapData(const uint16_t type, const std::map<std::string, std::string> &data_map,
                             std::vector<uint8_t> &result) {  // NOLINT(runtime/references)
  std::string rst = std::accumulate(data_map.begin(), data_map.end(), std::string(""),
                                    [](const std::string &r, const std::pair<const std::string, std::string> &element) {
                                      return r + element.first + ":" + element.second + ";";
                                    });
  if (!rst.empty()) {
    rst.pop_back();
  }
  encodeStrData(type, rst, result);
}

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_REPORT_DATA_H
