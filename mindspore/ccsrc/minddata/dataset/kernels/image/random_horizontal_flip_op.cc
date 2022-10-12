/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/random_horizontal_flip_op.h"

#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const float RandomHorizontalFlipOp::kDefProbability = 0.5;

Status RandomHorizontalFlipOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  const auto output_count = input.size();
  output->resize(output_count);

  for (const auto &image : input) {
    RETURN_IF_NOT_OK(ValidateImage(image, "RandomHorizontalFlip", {1, 2, 3, 4, 5, 6, 10, 11, 12}));
  }

  if (distribution_(rnd_)) {
    for (size_t i = 0; i < output_count; ++i) {
      auto input_shape = input[i]->shape();
      dsize_t rank = input_shape.Rank();
      if (rank <= kDefaultImageRank) {
        // [H, W] or [H, W, C]
        RETURN_IF_NOT_OK(HorizontalFlip(input[i], &(*output)[i]));
      } else {
        // reshape [..., H, W, C] to [N, H, W, C]
        dsize_t num_batch = input[i]->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]);
        TensorShape new_shape({num_batch, input_shape[-3], input_shape[-2], input_shape[-1]});
        RETURN_IF_NOT_OK(input[i]->Reshape(new_shape));

        // split [N, H, W, C] to N [H, W, C], and flip N [H, W, C]
        std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
        RETURN_IF_NOT_OK(BatchTensorToTensorVector(input[i], &input_vector_hwc));
        for (auto input_hwc : input_vector_hwc) {
          std::shared_ptr<Tensor> flip;
          RETURN_IF_NOT_OK(HorizontalFlip(input_hwc, &flip));
          output_vector_hwc.push_back(flip);
        }
        // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
        RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, &(*output)[i]));
        RETURN_IF_NOT_OK((*output)[i]->Reshape(input_shape));
      }
    }
    return Status::OK();
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
