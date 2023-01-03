/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TRUNCATE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TRUNCATE_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class TruncateOp : public TensorOp {
 public:
  explicit TruncateOp(int32_t max_seq_len) : max_seq_len_(max_seq_len) {}

  ~TruncateOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kTruncateOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 private:
  int32_t max_seq_len_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_TRUNCATE_OP_H_
