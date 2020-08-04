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

#ifndef DATASET_API_EXECUTE_H_
#define DATASET_API_EXECUTE_H_

#include <vector>
#include <memory>
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/include/de_tensor.h"
#include "minddata/dataset/include/transforms.h"

namespace mindspore {
namespace dataset {

class TensorOp;

namespace api {

// class to run tensor operations in eager mode
class Execute {
 public:
  /// \brief Constructor
  explicit Execute(std::shared_ptr<TensorOperation> op);

  /// \brief callable function to execute the TensorOperation in eager mode
  /// \param[inout] input - the tensor to be transformed
  /// \return - the output tensor, nullptr if Compute fails
  std::shared_ptr<tensor::MSTensor> operator()(std::shared_ptr<tensor::MSTensor> input);

 private:
  std::shared_ptr<TensorOperation> op_;
};

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_API_EXECUTE_H_
