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

#include <memory>
#include <string>
#include <vector>
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/status.h"

namespace mindspore {
namespace dataset {

class TensorOp;

// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kComposeOperation[] = "Compose";
constexpr char kDuplicateOperation[] = "Duplicate";
constexpr char kOneHotOperation[] = "OneHot";
constexpr char kPreBuiltOperation[] = "PreBuilt";
constexpr char kRandomApplyOperation[] = "RandomApply";
constexpr char kRandomChoiceOperation[] = "RandomChoice";
constexpr char kRandomSelectSubpolicyOperation[] = "RandomSelectSubpolicy";
constexpr char kTypeCastOperation[] = "TypeCast";
constexpr char kUniqueOperation[] = "Unique";

// Abstract class to represent a dataset in the data pipeline.
class TensorOperation : public std::enable_shared_from_this<TensorOperation> {
 public:
  /// \brief Constructor
  TensorOperation() : random_op_(false) {}

  /// \brief Constructor
  explicit TensorOperation(bool random) : random_op_(random) {}

  /// \brief Destructor
  ~TensorOperation() = default;

  /// \brief Pure virtual function to convert a TensorOperation class into a runtime TensorOp object.
  /// \return shared pointer to the newly created TensorOp.
  virtual std::shared_ptr<TensorOp> Build() = 0;

  virtual Status ValidateParams() = 0;

  virtual std::string Name() const = 0;

  /// \brief Check whether the operation is deterministic.
  /// \return true if this op is a random op (returns non-deterministic result e.g. RandomCrop)
  bool IsRandomOp() const { return random_op_; }

 protected:
  bool random_op_;
};

// Helper function to validate fill value
Status ValidateVectorFillvalue(const std::string &transform_name, const std::vector<uint8_t> &fill_value);

// Helper function to validate probability
Status ValidateProbability(const std::string &transform_name, const float &probability);

// Helper function to validate padding
Status ValidateVectorPadding(const std::string &transform_name, const std::vector<int32_t> &padding);

// Helper function to validate size
Status ValidateVectorPositive(const std::string &transform_name, const std::vector<int32_t> &size);

// Helper function to validate transforms
Status ValidateVectorTransforms(const std::string &transform_name,
                                const std::vector<std::shared_ptr<TensorOperation>> &transforms);

// Helper function to compare float value
bool CmpFloat(const float &a, const float &b, float epsilon = 0.0000000001f);

// Transform operations for performing data transformation.
namespace transforms {

// Transform Op classes (in alphabetical order)
class ComposeOperation;
class DuplicateOperation;
class OneHotOperation;
class PreBuiltOperation;
class RandomApplyOperation;
class RandomChoiceOperation;
class TypeCastOperation;
#ifndef ENABLE_ANDROID
class UniqueOperation;
#endif

/// \brief Function to create a Compose TensorOperation.
/// \notes Compose a list of transforms into a single transform.
/// \param[in] transforms A vector of transformations to be applied.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<ComposeOperation> Compose(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

/// \brief Function to create a Duplicate TensorOperation.
/// \notes Duplicate the input tensor to a new output tensor.
///     The input tensor is carried over to the output list.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DuplicateOperation> Duplicate();

/// \brief Function to create a OneHot TensorOperation.
/// \notes Convert the labels into OneHot format.
/// \param[in] num_classes number of classes.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<OneHotOperation> OneHot(int32_t num_classes);

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

/// \brief Function to create a TypeCast TensorOperation.
/// \notes Tensor operation to cast to a given MindSpore data type.
/// \param[in] data_type mindspore.dtype to be cast to.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<TypeCastOperation> TypeCast(std::string data_type);

#ifndef ENABLE_ANDROID
/// \brief Function to create a Unique TensorOperation.
/// \notes Return an output tensor containing all the unique elements of the input tensor in
///     the same order that they occur in the input tensor.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<UniqueOperation> Unique();
#endif

/* ####################################### Derived TensorOperation classes ################################# */

class ComposeOperation : public TensorOperation {
 public:
  explicit ComposeOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

  ~ComposeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kComposeOperation; }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

class DuplicateOperation : public TensorOperation {
 public:
  DuplicateOperation() = default;

  ~DuplicateOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDuplicateOperation; }
};

class OneHotOperation : public TensorOperation {
 public:
  explicit OneHotOperation(int32_t num_classes_);

  ~OneHotOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kOneHotOperation; }

 private:
  float num_classes_;
};

class PreBuiltOperation : public TensorOperation {
 public:
  explicit PreBuiltOperation(std::shared_ptr<TensorOp> tensor_op);

  ~PreBuiltOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPreBuiltOperation; }

 private:
  std::shared_ptr<TensorOp> op_;
};

class RandomApplyOperation : public TensorOperation {
 public:
  explicit RandomApplyOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms, double prob);

  ~RandomApplyOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomApplyOperation; }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  double prob_;
};

class RandomChoiceOperation : public TensorOperation {
 public:
  explicit RandomChoiceOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

  ~RandomChoiceOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomChoiceOperation; }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};
class TypeCastOperation : public TensorOperation {
 public:
  explicit TypeCastOperation(std::string data_type);

  ~TypeCastOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kTypeCastOperation; }

 private:
  std::string data_type_;
};

#ifndef ENABLE_ANDROID
class UniqueOperation : public TensorOperation {
 public:
  UniqueOperation() = default;

  ~UniqueOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kUniqueOperation; }
};
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TRANSFORMS_H_
