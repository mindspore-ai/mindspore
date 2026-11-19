/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <vector>

#include "utils/dual_abi_helper.h"
#include "include/dataset/constants.h"
#include "include/dataset/ms_tensor.h"
#include "minddata/dataset/util/visible.h"

namespace mindspore::dataset {
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
class DATASET_API TensorTransform : public std::enable_shared_from_this<TensorTransform> {
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
  TensorTransform() = default;

  /// \brief Destructor
  virtual ~TensorTransform() = default;

 protected:
  /// \brief Pure virtual function to convert a TensorTransform class into a IR TensorOperation object.
  /// \return shared pointer to the newly created TensorOperation.
  virtual std::shared_ptr<TensorOperation> Parse() = 0;

  /// \brief Virtual function to convert a TensorTransform class into a IR TensorOperation object.
  /// \param[in] env A string to determine the running environment
  /// \return shared pointer to the newly created TensorOperation.
  virtual std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) { return nullptr; }
};

/// \brief SliceOption used in Slice TensorTransform.
class DATASET_API Slice {
 public:
  /// \brief Constructor, with start, stop and step default to 0.
  Slice() : start_(0), stop_(0), step_(0) {}

  /// \brief Constructor.
  /// \param[in] start Starting integer specifying where to start the slicing.
  /// \param[in] stop Ending integer specifying where to stop the slicing.
  /// \param[in] step An integer specifying the step of the slicing.
  /// \par Example
  /// \code
  ///     /* Slice dimension from 2 to 10 with step 2. */
  ///     Slice(0, 10, 2);
  /// \endcode
  Slice(dsize_t start, dsize_t stop, dsize_t step) : start_(start), stop_(stop), step_(step) {}

  /// \brief Constructor, with step=1
  /// \param[in] start Starting integer specifying where to start the slicing.
  /// \param[in] stop Ending integer specifying where to stop the slicing.
  /// \par Example
  /// \code
  ///     /* Slice dimension from 5 to 10 with step 1. */
  ///     Slice(5, 10);
  /// \endcode
  Slice(dsize_t start, dsize_t stop) : start_(start), stop_(stop), step_(1) {}

  /// \brief Constructor, with start=0 and step=1
  /// \param[in] stop Ending integer specifying where to stop the slicing.
  /// \par Example
  /// \code
  ///     /* Slice dimension from 0 to 5 with step 1. */
  ///     Slice(5);
  /// \endcode
  explicit Slice(dsize_t stop) : start_(0), stop_(stop), step_(1) {}

  Slice(Slice const &slice) = default;

  Slice &operator=(const Slice &slice) = default;

  ~Slice() = default;

  bool valid() const { return step_ != 0; }
  dsize_t start_;
  dsize_t stop_;
  dsize_t step_;
};

class DATASET_API SliceOption {
 public:
  /// \param[in] all Slice the whole dimension
  /// \par Example
  /// \code
  ///     /* Slice all the data. */
  ///     SliceOption slice_option = SliceOption(True);
  /// \endcode
  explicit SliceOption(bool all) : all_(all) {}

  /// \param[in] indices Slice these indices along the dimension. Negative indices are supported.
  /// \par Example
  /// \code
  ///     /* Slice the given dimensions. */
  ///     std::vector<int64_t> indices = {0, 3, 6, 7};
  ///     SliceOption slice_option = SliceOption(indices);
  /// \endcode
  explicit SliceOption(const std::vector<dsize_t> &indices) : indices_(indices) {}

  /// \param[in] slice Slice the generated indices from the slice object along the dimension.
  /// \par Example
  /// \code
  ///     /* Slice dimension from 2 to 10 with step 2. */
  ///     SliceOption slice_option = SliceOption(Slice(0, 10, 2));
  ///     transforms::Slice slice_op = transforms::Slice({slice_option});
  /// \endcode
  explicit SliceOption(const Slice &slice) : slice_(slice) {}

  SliceOption(SliceOption const &slice) = default;

  SliceOption &operator=(const SliceOption &slice) = default;

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
class DATASET_API Compose final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms A vector of raw pointers to TensorTransform objects to be applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto resize_op(new vision::Resize({30, 30}));
  ///     auto center_crop_op(new vision::CenterCrop({16, 16}));
  ///     auto compose_op(new transforms::Compose({resize_op, center_crop_op}));
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({compose_op},  // operations
  ///                            {"image"});    // input columns
  /// \endcode
  explicit Compose(const std::vector<TensorTransform *> &transforms);

  /// \brief Constructor.
  /// \param[in] transforms A vector of shared pointers to TensorTransform objects to be applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  ///     std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  ///     std::shared_ptr<TensorTransform> compose_op(new transforms::Compose({resize_op, center_crop_op}));
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({compose_op},  // operations
  ///                            {"image"});    // input columns
  /// \endcode
  explicit Compose(const std::vector<std::shared_ptr<TensorTransform>> &transforms);

  /// \brief Constructor.
  /// \param[in] transforms A vector of TensorTransform objects to be applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     vision::Resize resize_op = vision::Resize({30, 30});
  ///     vision::CenterCrop center_crop_op = vision::CenterCrop({16, 16});
  ///     transforms::Compose compose_op = transforms::Compose({resize_op, center_crop_op});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({compose_op},  // operations
  ///                            {"image"});    // input columns
  /// \endcode
  explicit Compose(const std::vector<std::reference_wrapper<TensorTransform>> &transforms);

  /// \brief Destructor
  ~Compose() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Cast the MindSpore data type of a tensor to another.
class DATASET_API TypeCast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] data_type mindspore::DataType to be cast to.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto typecast_op = transforms::TypeCast(mindspore::DataType::kNumberTypeUInt8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({typecast_op},      // operations
  ///                            {"column"});        // input columns
  /// \endcode
  explicit TypeCast(mindspore::DataType data_type);

  /// \brief Destructor
  ~TypeCast() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

}  // namespace transforms
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_TRANSFORMS_H_
