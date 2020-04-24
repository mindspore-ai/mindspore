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
#include "dataset/kernels/image/uniform_aug_op.h"
#include "dataset/kernels/py_func_op.h"
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
const int UniformAugOp::kDefNumOps = 2;

UniformAugOp::UniformAugOp(py::list op_list, int32_t num_ops) : num_ops_(num_ops) {
  std::shared_ptr<TensorOp> tensor_op;
  // iterate over the op list, cast them to TensorOp and add them to tensor_op_list_
  for (auto op : op_list) {
    if (py::isinstance<py::function>(op)) {
      // python op
      tensor_op = std::make_shared<PyFuncOp>(op.cast<py::function>());
    } else if (py::isinstance<TensorOp>(op)) {
      // C++ op
      tensor_op = op.cast<std::shared_ptr<TensorOp>>();
    }
    tensor_op_list_.insert(tensor_op_list_.begin(), tensor_op);
  }

  rnd_.seed(GetSeed());
}
// compute method to apply uniformly random selected augmentations from a list
Status UniformAugOp::Compute(const std::vector<std::shared_ptr<Tensor>> &input,
                             std::vector<std::shared_ptr<Tensor>> *output) {
  IO_CHECK_VECTOR(input, output);

  // variables to copy the result to output if it is not already
  std::vector<std::shared_ptr<Tensor>> even_out;
  std::vector<std::shared_ptr<Tensor>> *even_out_ptr = &even_out;
  int count = 1;

  // randomly select ops to be applied
  std::vector<std::shared_ptr<TensorOp>> selected_tensor_ops;
  std::sample(tensor_op_list_.begin(), tensor_op_list_.end(), std::back_inserter(selected_tensor_ops), num_ops_, rnd_);

  for (auto tensor_op = selected_tensor_ops.begin(); tensor_op != selected_tensor_ops.end(); ++tensor_op) {
    // Do NOT apply the op, if second random generator returned zero
    if (std::uniform_int_distribution<int>(0, 1)(rnd_)) {
      continue;
    }

    // apply python/C++ op
    if (count == 1) {
      (**tensor_op).Compute(input, output);
    } else if (count % 2 == 0) {
      (**tensor_op).Compute(*output, even_out_ptr);
    } else {
      (**tensor_op).Compute(even_out, output);
    }
    count++;
  }

  // copy the result to output if it is not in output
  if (count == 1) {
    *output = input;
  } else if ((count % 2 == 1)) {
    (*output).swap(even_out);
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
