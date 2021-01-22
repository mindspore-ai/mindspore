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
#include "minddata/dataset/kernels/image/normalize_pad_op.h"

#include <random>

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
NormalizePadOp::NormalizePadOp(float mean_r, float mean_g, float mean_b, float std_r, float std_g, float std_b,
                               std::string dtype) {
  Status s = Tensor::CreateFromVector<float>({mean_r, mean_g, mean_b}, &mean_);
  if (s.IsError()) {
    MS_LOG(ERROR) << "NormalizePad: invalid mean value.";
  }
  s = Tensor::CreateFromVector<float>({std_r, std_g, std_b}, &std_);
  if (s.IsError()) {
    MS_LOG(ERROR) << "NormalizePad: invalid std value.";
  }
  dtype_ = dtype;
}

Status NormalizePadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Doing the normalization + pad
  return NormalizePad(input, output, mean_, std_, dtype_);
}

void NormalizePadOp::Print(std::ostream &out) const {
  out << "NormalizeOp, mean: " << *(mean_.get()) << std::endl << "std: " << *(std_.get()) << std::endl;
}
}  // namespace dataset
}  // namespace mindspore
