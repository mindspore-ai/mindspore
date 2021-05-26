/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_GAUSSIAN_BLUR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_GAUSSIAN_BLUR_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class GaussianBlurOp : public TensorOp {
 public:
  /// \brief Constructor to GaussianBlur Op
  /// \param[in] kernel_x - Gaussian kernel size of width
  /// \param[in] kernel_y - Gaussian kernel size of height
  /// \param[in] sigma_x - Gaussian kernel standard deviation of width
  /// \param[in] sigma_y - Gaussian kernel standard deviation of height
  GaussianBlurOp(int32_t kernel_x, int32_t kernel_y, float sigma_x, float sigma_y)
      : kernel_x_(kernel_x), kernel_y_(kernel_y), sigma_x_(sigma_x), sigma_y_(sigma_y) {}

  ~GaussianBlurOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kGaussianBlurOp; }

  void Print(std::ostream &out) const override {
    out << Name() << " kernel_size: (" << kernel_x_ << ", " << kernel_y_ << "), sigma: (" << sigma_x_ << ", "
        << sigma_y_ << ")";
  }

 protected:
  int32_t kernel_x_;
  int32_t kernel_y_;
  float sigma_x_;
  float sigma_y_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_GAUSSIAN_BLUR_OP_H_
