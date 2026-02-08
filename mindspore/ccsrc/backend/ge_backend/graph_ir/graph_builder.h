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

#ifndef MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_GRAPH_BUILDER_H_
#define MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_GRAPH_BUILDER_H_

#include <string>
#include <memory>
#include <map>
#include <vector>
#include "backend/ge_backend/graph_ir/types.h"
#include "backend/ge_backend/graph_ir/convert.h"

namespace mindspore::backend::ge_backend {

class DatasetGraphParam {
 public:
  DatasetGraphParam(const std::string &name, int64_t size, int64_t batch_size, const std::vector<int64_t> &ge_types,
                    const std::vector<std::vector<int64_t>> &shapes, const std::vector<int64_t> &input_indexes)
      : queue_name_(name),
        loop_size_(size),
        batch_size_(batch_size),
        ge_types_(ge_types),
        shapes_(shapes),
        input_indexes_(input_indexes) {}

  ~DatasetGraphParam() = default;

  std::string ToString() const {
    std::ostringstream buffer;
    buffer << "DatasetGraphParam: queue_name=" << queue_name_ << " size=" << loop_size_ << " batch_size=" << batch_size_
           << " ge_types=" << ge_types_ << " shapes=" << shapes_ << " input_indexes=" << input_indexes_;
    return buffer.str();
  }
  std::string queue_name() const { return queue_name_; }
  int64_t loop_size() const { return loop_size_; }
  int64_t batch_size() const { return batch_size_; }
  std::vector<int64_t> ge_types() const { return ge_types_; }
  std::vector<std::vector<int64_t>> shapes() const { return shapes_; }
  std::vector<int64_t> input_indexes() const { return input_indexes_; }

 private:
  std::string queue_name_;
  int64_t loop_size_;
  int64_t batch_size_;
  std::vector<int64_t> ge_types_;
  std::vector<std::vector<int64_t>> shapes_;
  std::vector<int64_t> input_indexes_;
};

Status BuildDatasetGraph(const DatasetGraphParam &param, const std::string &phase = "dataset");
}  // namespace mindspore::backend::ge_backend

#endif  // MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_GRAPH_BUILDER_H_
