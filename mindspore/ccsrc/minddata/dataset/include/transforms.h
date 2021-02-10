/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "minddata/dataset/include/constants.h"

// FIXME - This internal IR header will be removed when external API classes are provided
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

namespace mindspore {
namespace dataset {
// Abstract class to represent a tensor transform operation in the data pipeline.
/// \class TensorTransform transforms.h
/// \brief A base class to represent a tensor transform operation in the data pipeline.
class TensorTransform : public std::enable_shared_from_this<TensorTransform> {
 public:
  /// \brief Constructor
  TensorTransform() {}

  /// \brief Destructor
  ~TensorTransform() = default;

  /// \brief Pure virtual function to convert a TensorTransform class into a IR TensorOperation object.
  /// \return shared pointer to the newly created TensorOperation.
  virtual std::shared_ptr<TensorOperation> Parse() = 0;
};

// Transform operations for performing data transformation.
namespace transforms {

// Transform Op classes (in alphabetical order)
class ComposeOperation;
class RandomApplyOperation;
class RandomChoiceOperation;

/// \brief Function to create a Compose TensorOperation.
/// \notes Compose a list of transforms into a single transform.
/// \param[in] transforms A vector of transformations to be applied.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<ComposeOperation> Compose(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

/// \brief Duplicate Op.
/// \notes Duplicate the input tensor to a new output tensor.
///     The input tensor is carried over to the output list.
class Duplicate : public TensorTransform {
 public:
  /// \brief Constructor.
  Duplicate();

  /// \brief Destructor
  ~Duplicate() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return return code
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief OneHot Op.
/// \notes Convert the labels into OneHot format.
class OneHot : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] num_classes number of classes.
  explicit OneHot(int32_t num_classes);

  /// \brief Destructor
  ~OneHot() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return return code
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  float num_classes_;
};

/// \brief Function to create a RandomApply TensorOperation.
/// \notes Randomly perform a series of transforms with a given probability.
/// \param[in] transforms A vector of transformations to be applied.
/// \param[in] prob The probability to apply the transformation list (default=0.5)
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomApplyOperation> RandomApply(const std::vector<std::shared_ptr<TensorOperation>> &transforms,
                                                  double prob = 0.5);

/// \brief Function to create a RandomChoice TensorOperation.
/// \notes Randomly selects one transform from a list of transforms to perform operation.
/// \param[in] transforms A vector of transformations to be chosen from to apply.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<RandomChoiceOperation> RandomChoice(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

/// \brief TypeCast Op.
/// \notes Tensor operation to cast to a given MindSpore data type.
class TypeCast : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] data_type mindspore.dtype to be cast to.
  explicit TypeCast(std::string data_type);

  /// \brief Destructor
  ~TypeCast() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return return code
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  std::string data_type_;
};

#ifndef ENABLE_ANDROID
/// \brief Unique Op.
/// \notes Return an output tensor containing all the unique elements of the input tensor in
///     the same order that they occur in the input tensor.
class Unique : public TensorTransform {
 public:
  /// \brief Constructor.
  Unique();

  /// \brief Destructor
  ~Unique() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return return code
  std::shared_ptr<TensorOperation> Parse() override;
};
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TRANSFORMS_H_
