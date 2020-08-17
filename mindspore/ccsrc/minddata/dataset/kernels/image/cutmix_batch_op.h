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
  float alpha_;
  float prob_;
  ImageBatchFormat image_batch_format_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CUTMIXBATCH_OP_H_
