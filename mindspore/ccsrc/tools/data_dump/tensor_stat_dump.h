/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_STAT_DUMP_H_
#define MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_STAT_DUMP_H_

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "ir/dtype/type_id.h"
#include "tools/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
class Debugger;
class TensorData;

class TOOLS_EXPORT TensorStatDump {
 public:
  TensorStatDump(const std::string &op_type, const std::string &op_name, uint32_t task_id, uint32_t stream_id,
                 uint64_t timestamp, bool input, size_t slot, size_t tensor_loader_slot_);
  bool DumpTensorStatsToFile(const std::string &dump_path, const std::shared_ptr<TensorData> data);
  bool DumpTensorStatsToFile(const std::string &original_kernel_name, const std::string &dump_path,
                             const Debugger *debugger);

 private:
  const std::string op_type_;
  const std::string op_name_;
  const std::string task_id_;
  const std::string stream_id_;
  const std::string timestamp_;
  std::string io_;
  size_t slot_;
  size_t tensor_loader_slot_;
  mindspore::TypeId data_type_{mindspore::TypeId::kTypeUnknown};
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_STAT_DUMP_H_
