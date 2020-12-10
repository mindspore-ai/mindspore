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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_MINDDATA_EAGER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_MINDDATA_EAGER_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "include/api/types.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"

namespace mindspore {
namespace api {

// class to run tensor operations in eager mode
class MindDataEager {
 public:
  /// \brief Constructor
  MindDataEager() = default;

  /// \brief Constructor
  /// \param[inout] ops Transforms to be applied
  explicit MindDataEager(std::vector<std::shared_ptr<dataset::TensorOperation>> ops);

  /// \brief Destructor
  ~MindDataEager() = default;

  /// \brief Function to read images from local directory
  /// \param[inout] image_dir Target directory which contains images
  /// \param[output] images Vector of image Tensor
  /// \return Status The status code returned
  static Status LoadImageFromDir(const std::string &image_dir, std::vector<std::shared_ptr<Tensor>> *images);

  /// \brief Callable function to execute the TensorOperation in eager mode
  /// \param[inout] input Tensor to be transformed
  /// \return Output tensor, nullptr if Compute fails
  std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input);

 private:
  std::vector<std::shared_ptr<dataset::TensorOperation>> ops_;
};

}  // namespace api
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_MINDDATA_EAGER_H_
