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
#ifndef DATASET_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
#define DATASET_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>
#include "dataset/core/tensor.h"
#include "dataset/kernels/image/resize_op.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class ResizeBilinearOp : public ResizeOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefWidth;

  // Name: constructor
  // Resizes the image to the output specified size using Bilinear interpolation.
  // If only one value is provided, the it will resize the smaller size and maintains
  // the aspect ratio.
  // @param size1: the first size of output. If only this parameter is provided
  // the smaller dimension will be resized to this and then the other dimension changes
  // such that the aspect ratio is maintained.
  // @param size2: the second size of output. If this is also provided, the output size
  // will be (size1, size2)
  explicit ResizeBilinearOp(int32_t size1, int32_t size2 = kDefWidth)
      : ResizeOp(size1, size2, ResizeOp::kDefInterpolation) {}

  // Name: Destructor
  // Description: Destructor
  ~ResizeBilinearOp() = default;

  // Name: Print()
  // Description: A function that prints info about the node
  void Print(std::ostream &out) const override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
