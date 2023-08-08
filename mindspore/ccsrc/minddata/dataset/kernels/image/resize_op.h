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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class ResizeOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefWidth;
  static const InterpolationMode kDefInterpolation;

  // Resizes the image to the output specified size. If only one value is provided,
  // the it will resize the smaller size and maintains the aspect ratio.
  // @param size1: the first size of output. If only this parameter is provided
  // the smaller dimension will be resized to this and then the other dimension changes
  // such that the aspect ratio is maintained.
  // @param size2: the second size of output. If this is also provided, the output size
  // will be (size1, size2)
  // @param InterpolationMode: the interpolation mode being used.
  explicit ResizeOp(int32_t size1, int32_t size2 = kDefWidth, InterpolationMode interpolation = kDefInterpolation)
      : size1_(size1), size2_(size2), interpolation_(interpolation) {}

  ResizeOp(const ResizeOp &rhs) = default;

  ResizeOp(ResizeOp &&rhs) = default;

  ~ResizeOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ": " << size1_ << " " << size2_; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  static TensorShape ComputeOutputShape(const TensorShape &input, int32_t output_h, int32_t output_w);

  std::string Name() const override { return kResizeOp; }

 protected:
  int32_t size1_;
  int32_t size2_;
  InterpolationMode interpolation_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_OP_H_
