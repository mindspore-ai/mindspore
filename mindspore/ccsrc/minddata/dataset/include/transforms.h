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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TRANSFORMS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TRANSFORMS_H_

#include <vector>
#include <memory>
#include "minddata/dataset/core/constants.h"

namespace mindspore {
namespace dataset {

class TensorOp;

namespace api {
// Abstract class to represent a dataset in the data pipeline.
class TensorOperation : public std::enable_shared_from_this<TensorOperation> {
 public:
  /// \brief Constructor
  TensorOperation();

  /// \brief Destructor
  ~TensorOperation() = default;

  /// \brief Pure virtual function to convert a TensorOperation class into a runtime TensorOp object.
  /// \return shared pointer to the newly created TensorOp.
  virtual std::shared_ptr<TensorOp> Build() = 0;

  virtual bool ValidateParams() = 0;
};

// Transform operations for performing data transformation.
namespace transforms {

// Transform Op classes (in alphabetical order)
class OneHotOperation;

/// \brief Function to create a OneHot TensorOperation.
/// \notes Convert the labels into OneHot format.
/// \param[in] num_classes number of classes.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<OneHotOperation> OneHot(int32_t num_classes);

/* ####################################### Derived TensorOperation classes ################################# */

class OneHotOperation : public TensorOperation {
 public:
  explicit OneHotOperation(int32_t num_classes_);

  ~OneHotOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  bool ValidateParams() override;

 private:
  float num_classes_;
};
}  // namespace transforms
}  // namespace api
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TRANSFORMS_H_
