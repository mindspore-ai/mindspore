/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/normalize_op.h"

#include <random>
#include <vector>

#include "minddata/dataset/kernels/data/data_utils.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
NormalizeOp::NormalizeOp(const std::vector<float> &mean, const std::vector<float> &std, bool is_hwc)
    : mean_(mean), std_(std), is_hwc_(is_hwc) {}

Status NormalizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Doing the Normalization
  auto input_shape = input->shape();
  dsize_t rank = input_shape.Rank();
  if (rank < kMinImageRank) {
    std::string err_msg = "Normalize: input tensor should have at least 2 dimensions, but got: " + std::to_string(rank);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else if (rank <= kDefaultImageRank) {
    // [H, W] or [H, W, C]
#ifndef ENABLE_ANDROID
    return Normalize(input, output, mean_, std_, is_hwc_);
#else
    return Normalize(input, output, mean_, std_);
#endif
  } else {
    // reshape [..., H, W, C] to [N, H, W, C]
    dsize_t num_batch = input->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]);
    TensorShape new_shape({num_batch, input_shape[-3], input_shape[-2], input_shape[-1]});
    RETURN_IF_NOT_OK(input->Reshape(new_shape));

    // split [N, H, W, C] to N [H, W, C], and normalize N [H, W, C]
    std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(input, &input_vector_hwc));
    for (auto input_hwc : input_vector_hwc) {
      std::shared_ptr<Tensor> normalize;
#ifndef ENABLE_ANDROID
      RETURN_IF_NOT_OK(Normalize(input_hwc, &normalize, mean_, std_, is_hwc_));
#else
      RETURN_IF_NOT_OK(Normalize(input_hwc, &normalize, mean_, std_));
#endif
      output_vector_hwc.push_back(normalize);
    }
    // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, &(*output)));
    RETURN_IF_NOT_OK((*output)->Reshape(input_shape));
    return Status::OK();
  }
}

void NormalizeOp::Print(std::ostream &out) const {
  out << "NormalizeOp, mean: ";
  for (const auto &m : mean_) {
    out << m << ", ";
  }
  out << "}" << std::endl << "std: ";
  for (const auto &s : std_) {
    out << s << ", ";
  }
  out << "}" << std::endl << "is_hwc: " << is_hwc_;
  out << "}" << std::endl;
}
}  // namespace dataset
}  // namespace mindspore
