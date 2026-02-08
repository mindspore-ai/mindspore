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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_STATISTIC_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_STATISTIC_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "tools/data_dump/debug_services.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace datadump {
using device::DeviceAddressPtr;
using kernel::KernelTensor;
using kernel::KernelTensorPtr;
using mindspore::device::DeviceContext;
using TensorPtr = tensor::TensorPtr;

class TensorStat {
 public:
  TensorStat(const string &type, const string &name, size_t task_id, size_t stream_id, uint64_t timestamp,
             const string &io, size_t slot, size_t data_size, const string &data_type, const string &shape,
             const string &max_value, const string &min_value, const string &avg_value, const string &norm_value,
             size_t count)
      : type_(type),
        name_(name),
        task_id_(task_id),
        stream_id_(stream_id),
        timestamp_(timestamp),
        io_(io),
        slot_(slot),
        data_size_(data_size),
        data_type_(data_type),
        shape_(shape),
        max_value_(max_value),
        min_value_(min_value),
        avg_value_(avg_value),
        norm_value_(norm_value),
        count_(count) {}
  TensorStat() = default;
  std::map<std::string, std::string> header_item_map;
  void UpdateHeaderItemMap() {
    header_item_map = {{"max", max_value_}, {"min", min_value_}, {"avg", avg_value_}, {"l2norm", norm_value_}};
  }

 public:
  string type_;
  string name_;
  size_t task_id_;
  size_t stream_id_;
  uint64_t timestamp_;
  string io_;
  size_t slot_;
  size_t data_size_;
  string data_type_;
  string shape_;
  string max_value_;
  string min_value_;
  string avg_value_;
  string norm_value_;
  size_t count_;
};

class DumpTensorInfo {
 public:
  DumpTensorInfo(const DeviceContext *dc, KernelTensor *t, bool in, size_t s, const string &name, const string &type)
      : device_context(dc), tensor(t), is_input(in), slot(s), op_name(name), op_type(type) {}
  const DeviceContext *device_context;
  KernelTensor *tensor;
  bool is_input;
  size_t slot;
  string op_name;
  string op_type;
};

TensorStat GetKernelTensorStats(const DumpTensorInfo &, const std::set<std::string> &stat_name_list);
void DumpKernelTensorStats(const DeviceContext *device_context, std::vector<KernelTensor *> tensors, bool is_input,
                           const CNodePtr &node, uint32_t graph_id);

}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_STATISTIC_H_
