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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_PARSE_EXAMPLE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_PARSE_EXAMPLE_OP_H_

#include <unsupported/Eigen/CXX11/ThreadPool>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
constexpr int kThreadPoolSize = 32;

struct VarLenTensorBuffer {
  std::vector<std::shared_ptr<Tensor>> numeric_tensor;  // store the minibatch of numeric tensors
  std::vector<std::string> string_tensor;               // store the minibatch of strings
  size_t string_length;                                 // store the lengtn of string in minibatch
};

class ParseExampleOp : public TensorOp {
 public:
  ParseExampleOp(DataSchema data_schema, std::vector<std::string> column_list, bool parallel_parse)
      : data_schema_(std::move(data_schema)),
        column_list_(std::move(column_list)),
        parallel_parse_(parallel_parse),
        pool_(nullptr) {
    if (parallel_parse) {
      pool_ = std::make_unique<Eigen::ThreadPool>(kThreadPoolSize);
    }
  }

  ~ParseExampleOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kParseExampleOp; }

 private:
  Status ParseSingleExample(const TensorRow &raw_bytes, TensorRow *parsed_row);

  Status ParallelParseExample(const TensorRow &raw_bytes, TensorRow *parsed_row);

  Status ParseSerializedExample(const std::string &example_bytes, TensorRow *parsed_row,
                                std::unordered_map<int32_t, std::vector<std::string>> *string_column_map,
                                std::vector<VarLenTensorBuffer> *varlen_tensor_vector, size_t tensor_index);

  Status ConstructColumnMap(const std::string &example_bytes);

  DataSchema data_schema_;
  std::vector<std::string> column_list_;
  bool parallel_parse_;
  std::unique_ptr<Eigen::ThreadPool> pool_;
  std::unordered_map<std::string, int32_t> column_name_id_map_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_PARSE_EXAMPLE_OP_H_
