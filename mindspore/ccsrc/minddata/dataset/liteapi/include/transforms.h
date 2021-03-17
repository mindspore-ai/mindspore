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

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/constants.h"

namespace mindspore {
namespace dataset {

class TensorOperation;

// We need the following two groups of forward declaration to friend the class in class TensorTransform.
namespace transforms {
class Compose;
class RandomApply;
class RandomChoice;
}  // namespace transforms

namespace vision {
class BoundingBoxAugment;
class RandomSelectSubpolicy;
class UniformAugment;
}  // namespace vision

// Abstract class to represent a tensor transform operation in the data pipeline.
/// \class TensorTransform transforms.h
/// \brief A base class to represent a tensor transform operation in the data pipeline.
class TensorTransform : public std::enable_shared_from_this<TensorTransform> {
  friend class Dataset;
  friend class Execute;
  friend class transforms::Compose;
  friend class transforms::RandomApply;
  friend class transforms::RandomChoice;
  friend class vision::BoundingBoxAugment;
  friend class vision::RandomSelectSubpolicy;
  friend class vision::UniformAugment;

 public:
  /// \brief Constructor
  TensorTransform() {}

  /// \brief Destructor
  ~TensorTransform() = default;

 protected:
  /// \brief Pure virtual function to convert a TensorTransform class into a IR TensorOperation object.
  /// \return shared pointer to the newly created TensorOperation.
  virtual std::shared_ptr<TensorOperation> Parse() = 0;

  /// \brief Virtual function to convert a TensorTransform class into a IR TensorOperation object.
  /// \param[in] env A string to determine the running environment
  /// \return shared pointer to the newly created TensorOperation.
  virtual std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) { return nullptr; }
};

// Transform operations for performing data transformation.
namespace transforms {

/// \brief Compose Op.
/// \notes Compose a list of transforms into a single transform.
class Compose final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms A vector of raw pointers to TensorTransform objects to be applied.
  explicit Compose(const std::vector<TensorTransform *> &transforms);
  /// \brief Constructor.
  /// \param[in] transforms A vector of shared pointers to TensorTransform objects to be applied.
  explicit Compose(const std::vector<std::shared_ptr<TensorTransform>> &transforms);
  /// \brief Constructor.
  /// \param[in] transforms A vector of TensorTransform objects to be applied.
  explicit Compose(const std::vector<std::reference_wrapper<TensorTransform>> &transforms);

  /// \brief Destructor
  ~Compose() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Duplicate Op.
/// \notes Duplicate the input tensor to a new output tensor.
///     The input tensor is carried over to the output list.
class Duplicate final : public TensorTransform {
 public:
  /// \brief Constructor.
  Duplicate();

  /// \brief Destructor
  ~Duplicate() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief OneHot Op.
/// \notes Convert the labels into OneHot format.
class OneHot final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] num_classes number of classes.
  explicit OneHot(int32_t num_classes);

  /// \brief Destructor
  ~OneHot() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief RandomApply Op.
/// \notes Randomly perform a series of transforms with a given probability.
class RandomApply final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms A vector of raw pointers to TensorTransform objects to be applied.
  /// \param[in] prob The probability to apply the transformation list (default=0.5)
  explicit RandomApply(const std::vector<TensorTransform *> &transforms, double prob = 0.5);
  /// \brief Constructor.
  /// \param[in] transforms A vector of shared pointers to TensorTransform objects to be applied.
  /// \param[in] prob The probability to apply the transformation list (default=0.5)
  explicit RandomApply(const std::vector<std::shared_ptr<TensorTransform>> &transforms, double prob = 0.5);
  /// \brief Constructor.
  /// \param[in] transforms A vector of TensorTransform objects to be applied.
  /// \param[in] prob The probability to apply the transformation list (default=0.5)
  explicit RandomApply(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, double prob = 0.5);

  /// \brief Destructor
  ~RandomApply() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief RandomChoice Op.
/// \notes Randomly selects one transform from a list of transforms to perform operation.
class RandomChoice final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms A vector of raw pointers to TensorTransform objects to be applied.
  explicit RandomChoice(const std::vector<TensorTransform *> &transforms);
  /// \brief Constructor.
  /// \param[in] transforms A vector of shared pointers to TensorTransform objects to be applied.
  explicit RandomChoice(const std::vector<std::shared_ptr<TensorTransform>> &transforms);
  /// \brief Constructor.
  /// \param[in] transforms A vector of TensorTransform objects to be applied.
  explicit RandomChoice(const std::vector<std::reference_wrapper<TensorTransform>> &transforms);

  /// \brief Destructor
  ~RandomChoice() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief TypeCast Op.
/// \notes Tensor operation to cast to a given MindSpore data type.
class TypeCast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] data_type mindspore.dtype to be cast to.
  explicit TypeCast(std::string data_type) : TypeCast(StringToChar(data_type)) {}

  explicit TypeCast(const std::vector<char> &data_type);

  /// \brief Destructor
  ~TypeCast() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Unique Op.
/// \notes Return an output tensor containing all the unique elements of the input tensor in
///     the same order that they occur in the input tensor.
class Unique final : public TensorTransform {
 public:
  /// \brief Constructor.
  Unique();

  /// \brief Destructor
  ~Unique() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TRANSFORMS_H_
