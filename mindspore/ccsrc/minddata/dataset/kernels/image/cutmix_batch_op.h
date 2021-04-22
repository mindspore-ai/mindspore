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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CUTMIXBATCH_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CUTMIXBATCH_OP_H_

#include <memory>
#include <vector>
#include <random>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class CutMixBatchOp : public TensorOp {
 public:
  explicit CutMixBatchOp(ImageBatchFormat image_batch_format, float alpha, float prob);

  ~CutMixBatchOp() override = default;

  void Print(std::ostream &out) const override;

  void GetCropBox(int width, int height, float lam, int *x, int *y, int *crop_width, int *crop_height);

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kCutMixBatchOp; }

 private:
  /// \brief Helper function used in Compute to validate the input TensorRow.
  /// \param[in] input Input TensorRow of CutMixBatchOp
  /// \returns Status
  Status ValidateCutMixBatch(const TensorRow &input);

  /// \brief Helper function used in Compute to compute each image.
  /// \param[in] input Input TensorRow of CutMixBatchOp.
  /// \param[in] rand_indx_i The i-th generated random index as the start address of the input image.
  /// \param[in] lam A random variable follow Beta distribution, used in GetCropBox.
  /// \param[in] label_lam Lambda used for labels, will be updated after computing each image.
  /// \param[in] image_i The result of the i-th computed image.
  /// \returns Status
  Status ComputeImage(const TensorRow &input, const int64_t rand_indx_i, const float lam, float *label_lam,
                      std::shared_ptr<Tensor> *image_i);

  /// \brief Helper function used in Compute to compute each label corresponding to each image.
  /// \param[in] input Input TensorRow of CutMixBatchOp.
  /// \param[in] rand_indx_i The i-th generated random index as the start address of the input image.
  /// \param[in] index_i The i-th label to be generated, corresponding to the i-th computed image.
  /// \param[in] row_labels Number of rows of the label.
  /// \param[in] num_classes Number of class of the label.
  /// \param[in] label_shape_size The size of the label shape from input TensorRow.
  /// \param[in] label_lam Lambda used for setting the location.
  /// \param[in] out_labels The output of the i-th label, corresponding to the i-th computed image.
  /// \returns Status
  Status ComputeLabel(const TensorRow &input, const int64_t rand_indx_i, const int64_t index_i,
                      const int64_t row_labels, const int64_t num_classes, const std::size_t label_shape_size,
                      const float label_lam, std::shared_ptr<Tensor> *out_labels);
  float alpha_;
  float prob_;
  ImageBatchFormat image_batch_format_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CUTMIXBATCH_OP_H_
