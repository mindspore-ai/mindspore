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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_SWAP_RED_BLUE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_SWAP_RED_BLUE_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class SwapRedBlueOp : public TensorOp {
 public:
  // SwapRedBlues the image to the output specified size. If only one value is provided,
  // the it will crop the smaller size and maintains the aspect ratio.
  // @param size1: the first size of output. If only this parameter is provided
  // the smaller dimension will be cropd to this and then the other dimension changes
  // such that the aspect ratio is maintained.
  // @param size2: the second size of output. If this is also provided, the output size
  // will be (size1, size2)
  // @param InterpolationMode: the interpolation mode being used.
  SwapRedBlueOp() {}

  SwapRedBlueOp(const SwapRedBlueOp &rhs) = default;

  SwapRedBlueOp(SwapRedBlueOp &&rhs) = default;

  ~SwapRedBlueOp() override = default;

  void Print(std::ostream &out) const override { out << "SwapRedBlueOp x"; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSwapRedBlueOp; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_SWAP_RED_BLUE_OP_H_
