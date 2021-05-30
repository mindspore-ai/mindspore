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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MIXUPBATCH_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MIXUPBATCH_OP_H_

#include <memory>
#include <vector>
#include <random>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class MixUpBatchOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc

  explicit MixUpBatchOp(float alpha);

  ~MixUpBatchOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kMixUpBatchOp; }

 private:
  // a helper function to shorten the main Compute function
  Status ComputeLabels(const TensorRow &input, std::shared_ptr<Tensor> *out_labels, std::vector<int64_t> *rand_indx,
                       const std::vector<int64_t> &label_shape, const float lam, const size_t images_size);
  float alpha_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MIXUPBATCH_OP_H_
