/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_PROFILING_DESC_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_PROFILING_DESC_H_

#include <string>
#include <utility>
#include <vector>

namespace mindspore {
namespace device {
namespace ascend {
class ProfDesc {
 public:
  explicit ProfDesc(std::string op_name) : op_name_(std::move(op_name)) {}
  virtual std::string ToString() = 0;

 protected:
  std::string op_name_;
};

class TaskDesc : public ProfDesc {
 public:
  TaskDesc(std::string op_name, uint32_t task_id, uint32_t block_dim, uint32_t stream_id)
      : ProfDesc(std::move(op_name)), task_id_(task_id), block_dim_(block_dim), stream_id_(stream_id) {}
  std::string ToString() override;

 private:
  uint32_t task_id_;
  uint32_t block_dim_;
  uint32_t stream_id_;
};

struct DataElement {
  size_t index_;
  std::string data_format_;
  int data_type_;
  std::vector<size_t> data_shape_;
};

class GraphDesc : public ProfDesc {
 public:
  GraphDesc(std::string op_name, std::string op_type, std::vector<DataElement> input_data_list,
            std::vector<DataElement> output_data_list)
      : ProfDesc(std::move(op_name)),
        op_type_(std::move(op_type)),
        input_data_list_(std::move(input_data_list)),
        output_data_list_(std::move(output_data_list)) {}
  std::string ToString() override;

 private:
  std::string op_type_;
  std::vector<DataElement> input_data_list_;
  std::vector<DataElement> output_data_list_;
  [[nodiscard]] static std::string DataShapeToString(const std::vector<size_t> &shape);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_PROFILING_DESC_H_
