/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_SLICE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_SLICE_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_helpers.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class SliceOp : public TensorOp {
 public:
  explicit SliceOp(std::vector<SliceOption> slice_input) : slice_options_(std::move(slice_input)) {}

  explicit SliceOp(const SliceOption &slice_option) { slice_options_.push_back(slice_option); }

  // short hand notation for slicing along fist dimension
  explicit SliceOp(Slice slice) { (void)slice_options_.emplace_back(slice); }

  explicit SliceOp(bool all) { (void)slice_options_.emplace_back(all); }

  explicit SliceOp(const std::vector<dsize_t> &indices) { (void)slice_options_.emplace_back(indices); }

  ~SliceOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSliceOp; }

 private:
  std::vector<SliceOption> slice_options_ = {};
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_SLICE_OP_H_
