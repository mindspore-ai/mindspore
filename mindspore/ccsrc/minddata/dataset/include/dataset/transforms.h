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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_TRANSFORMS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_TRANSFORMS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/dataset/constants.h"

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

/// \brief Slice object used in SliceOption.
class Slice {
 public:
  /// \brief Constructor, with start, stop and step default to 0.
  Slice() : start_(0), stop_(0), step_(0) {}
  /// \brief Constructor.
  /// \param[in] start Starting integer specifying where to start the slicing.
  /// \param[in] stop Ending integer specifying where to stop the slicing.
  /// \param[in] step An integer specifying the step of the slicing.
  Slice(dsize_t start, dsize_t stop, dsize_t step) : start_(start), stop_(stop), step_(step) {}
  /// \brief Constructor, with step=1
  /// \param[in] start Starting integer specifying where to start the slicing.
  /// \param[in] stop Ending integer specifying where to stop the slicing.
  Slice(dsize_t start, dsize_t stop) : start_(start), stop_(stop), step_(1) {}
  /// \brief Constructor, with start=0 and step=1
  /// \param[in] stop Ending integer specifying where to stop the slicing.
  explicit Slice(dsize_t stop) : start_(0), stop_(stop), step_(1) {}
  Slice(Slice const &slice) = default;

  ~Slice() = default;

  bool valid() const { return step_ != 0; }
  dsize_t start_;
  dsize_t stop_;
  dsize_t step_;
};

/// \brief SliceOption used in Slice TensorTransform.
class SliceOption {
 public:
  /// \param[in] all Slice the whole dimension
  explicit SliceOption(bool all) : all_(all) {}
  /// \param[in] indices Slice these indices along the dimension. Negative indices are supported.
  explicit SliceOption(std::vector<dsize_t> indices) : indices_(indices) {}
  /// \param[in] slice Slice the generated indices from the slice object along the dimension.
  explicit SliceOption(Slice slice) : slice_(slice) {}
  SliceOption(SliceOption const &slice) = default;

  ~SliceOption() = default;

  // only one of the following will be valid
  // given indices to slice the Tensor.
  std::vector<dsize_t> indices_ = {};
  // Slice object. All start, stop and step are 0 if invalid.
  Slice slice_;
  bool all_ = false;
};

// Transform operations for performing data transformation.
namespace transforms {

/// \brief Compose a list of transforms into a single transform.
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
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Concatenate all tensors into a single tensor.
class Concatenate final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] axis Concatenate the tensors along given axis, only support 0 or -1 so far (default=0).
  /// \param[in] prepend MSTensor to be prepended to the concatenated tensors (default={}).
  /// \param[in] append MSTensor to be appended to the concatenated tensors (default={}).
  explicit Concatenate(int8_t axis = 0, const MSTensor &prepend = {}, const MSTensor &append = {});

  /// \brief Destructor
  ~Concatenate() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Duplicate the input tensor to a new output tensor.
///     The input tensor is carried over to the output list.
class Duplicate final : public TensorTransform {
 public:
  /// \brief Constructor.
  Duplicate();

  /// \brief Destructor
  ~Duplicate() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Fill all elements in the tensor with the specified value.
///    The output tensor will have the same shape and type as the input tensor.
class Fill final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] fill_value Scalar value to fill the tensor with.
  ///               It can only be MSTensor of the following types from mindspore::DataType:
  ///               String, Bool, Int8/16/32/64, UInt8/16/32/64, Float16/32/64.
  explicit Fill(const MSTensor &fill_value);

  /// \brief Destructor
  ~Fill() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Mask content of the input tensor with the given predicate.
///     Any element of the tensor that matches the predicate will be evaluated to True, otherwise False.
class Mask final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] op One of the relational operators: EQ, NE LT, GT, LE or GE.
  /// \param[in] constant Constant to be compared to. It can only be MSTensor of the following types
  ///                from mindspore::DataType: String, Int, Float, Bool.
  /// \param[in] de_type Type of the generated mask. It can only be numeric or boolean datatype.
  ///               (default=mindspore::DataType::kNumberTypeBool)
  explicit Mask(RelationalOp op, const MSTensor &constant,
                mindspore::DataType ms_type = mindspore::DataType(mindspore::DataType::kNumberTypeBool));

  /// \brief Destructor
  ~Mask() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Convert the labels into OneHot format.
class OneHot final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] num_classes number of classes.
  explicit OneHot(int32_t num_classes);

  /// \brief Destructor
  ~OneHot() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Pad input tensor according to pad_shape
class PadEnd final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] pad_shape List of integers representing the shape needed, need to have same rank with input tensor.
  ///               Dimensions that set to `-1` will not be padded (i.e., original dim will be used).
  ///               Shorter dimensions will truncate the values.
  /// \param[in] pad_value Value used to pad (default={}).
  explicit PadEnd(const std::vector<dsize_t> &pad_shape, const MSTensor &pad_value = {});

  /// \brief Destructor
  ~PadEnd() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly perform a series of transforms with a given probability.
class RandomApply final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms A vector of raw pointers to TensorTransform objects to be applied.
  /// \param[in] prob The probability to apply the transformation list (default=0.5).
  explicit RandomApply(const std::vector<TensorTransform *> &transforms, double prob = 0.5);
  /// \brief Constructor.
  /// \param[in] transforms A vector of shared pointers to TensorTransform objects to be applied.
  /// \param[in] prob The probability to apply the transformation list (default=0.5).
  explicit RandomApply(const std::vector<std::shared_ptr<TensorTransform>> &transforms, double prob = 0.5);
  /// \brief Constructor.
  /// \param[in] transforms A vector of TensorTransform objects to be applied.
  /// \param[in] prob The probability to apply the transformation list (default=0.5).
  explicit RandomApply(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, double prob = 0.5);

  /// \brief Destructor
  ~RandomApply() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly select one transform from a list of transforms to perform on the input tensor.
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
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Extract a tensor out using the given n slices.
///     The functionality of Slice is similar to the feature of indexing of NumPy.
///     (Currently only rank-1 tensors are supported).
class Slice final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] slice_input Vector of SliceOption
  explicit Slice(const std::vector<SliceOption> &slice_input);

  /// \brief Destructor
  ~Slice() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Cast the MindSpore data type of a tensor to another.
class TypeCast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] data_type mindspore::DataType to be cast to.
  explicit TypeCast(mindspore::DataType data_type);

  /// \brief Destructor
  ~TypeCast() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Return an output tensor that contains all the unique elements of the input tensor in
///     the same order as they appear in the input tensor.
class Unique final : public TensorTransform {
 public:
  /// \brief Constructor.
  Unique();

  /// \brief Destructor
  ~Unique() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_TRANSFORMS_H_
