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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TEXT_SLIDING_WINDOW_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TEXT_SLIDING_WINDOW_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/kernels/data_utils.h"

namespace mindspore {
namespace dataset {

class SlidingWindowOp : public TensorOp {
 public:
  /// \brief Constructor of SlidingWindowOp.
  /// \param[in] width - The axis along which sliding window is computed.
  /// \param[in] axis - The width of the window.
  /// \return Status return code
  explicit SlidingWindowOp(uint32_t width, int32_t axis = 0) : width_(width), axis_(axis) {}

  /// \brief Destructor of SlidingWindowOp.
  ~SlidingWindowOp() override = default;

  /// \brief Perform sliding window to tensor.
  /// \param[in] input - Input tensor of Op.
  /// \param[out] output - output tensor of Op.
  /// \return Status return code
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  /// \brief Calculate tensor shape for output tensor.
  /// \param[in] inputs - Input tensor shapes.
  /// \param[out] outputs - Output tensor shapes.
  /// \return Status return code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// \brief Print name of op.
  std::string Name() const override { return kSlidingWindowOp; }

 private:
  uint32_t width_;  // The width of the window. Must be an integer and greater than zero.
  int32_t axis_;    // The axis along which sliding window is computed, only support 0/-1 for now.
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TEXT_SLIDING_WINDOW_OP_H_
