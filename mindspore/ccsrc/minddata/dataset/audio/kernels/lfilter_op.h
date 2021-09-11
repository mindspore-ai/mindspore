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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_LFILTER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_LFILTER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class LFilterOp : public TensorOp {
 public:
  LFilterOp(std::vector<float> a_coeffs, std::vector<float> b_coeffs, bool clamp)
      : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}

  ~LFilterOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": a_coeffs: ";
    for (int i = 0; i < a_coeffs_.size(); i++) {
      out << a_coeffs_[i] << " ";
    }
    out << "b_coeffs: ";
    for (int i = 0; i < b_coeffs_.size(); i++) {
      out << b_coeffs_[i] << " ";
    }
    out << "clamp: " << clamp_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kLFilterOp; }

 private:
  std::vector<float> a_coeffs_;
  std::vector<float> b_coeffs_;
  bool clamp_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_LFILTER_OP_H_
