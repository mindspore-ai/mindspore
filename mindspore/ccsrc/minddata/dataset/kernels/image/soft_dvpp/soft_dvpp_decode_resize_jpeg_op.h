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
#ifndef DATASET_KERNELS_IMAGE_SOFT_DVPP_DECODE_RESIZE_JPEG_OP_H_
#define DATASET_KERNELS_IMAGE_SOFT_DVPP_DECODE_RESIZE_JPEG_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class SoftDvppDecodeResizeJpegOp : public TensorOp {
 public:
  SoftDvppDecodeResizeJpegOp(int32_t target_height, int32_t target_width)
      : target_height_(target_height), target_width_(target_width) {}

  /// \brief Destructor
  ~SoftDvppDecodeResizeJpegOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kSoftDvppDecodeReiszeJpegOp; }

 private:
  int32_t target_height_;
  int32_t target_width_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_SOFT_DVPP_DECODE_RESIZE_JPEG_OP_H_
