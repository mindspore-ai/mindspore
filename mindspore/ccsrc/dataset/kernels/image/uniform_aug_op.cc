/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include <utility>
#include "dataset/kernels/image/uniform_aug_op.h"
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
const int UniformAugOp::kDefNumOps = 2;

UniformAugOp::UniformAugOp(std::vector<std::shared_ptr<TensorOp>> op_list, int32_t num_ops)
    : tensor_op_list_(op_list), num_ops_(num_ops) {
  rnd_.seed(GetSeed());
}

// compute method to apply uniformly random selected augmentations from a list
Status UniformAugOp::Compute(const std::vector<std::shared_ptr<Tensor>> &input,
                             std::vector<std::shared_ptr<Tensor>> *output) {
  IO_CHECK_VECTOR(input, output);

  // randomly select ops to be applied
  std::vector<std::shared_ptr<TensorOp>> selected_tensor_ops;
  std::sample(tensor_op_list_.begin(), tensor_op_list_.end(), std::back_inserter(selected_tensor_ops), num_ops_, rnd_);

  bool first = true;
  for (const auto &tensor_op : selected_tensor_ops) {
    // Do NOT apply the op, if second random generator returned zero
    if (std::uniform_int_distribution<int>(0, 1)(rnd_)) {
      continue;
    }
    // apply C++ ops (note: python OPs are not accepted)
    if (first) {
      RETURN_IF_NOT_OK(tensor_op->Compute(input, output));
      first = false;
    } else {
      RETURN_IF_NOT_OK(tensor_op->Compute(std::move(*output), output));
    }
  }

  // The case where no tensor op is applied.
  if (output->empty()) {
    *output = input;
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
