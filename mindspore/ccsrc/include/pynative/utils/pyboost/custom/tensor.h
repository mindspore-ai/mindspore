/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_TENSOR_H_
#define MINDSPORE_CCSRC_EXTENSION_TENSOR_H_
#include <set>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <optional>
#include "pybind11/pybind11.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "pynative/utils/pyboost/custom/tensor_accessor.h"

namespace mindspore {
namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
}  // namespace tensor
class Value;
using ValuePtr = std::shared_ptr<Value>;
}  // namespace mindspore

namespace ms {
using TypeId = mindspore::TypeId;

/**
 * @class [API] Tensor
 * @brief Represents a tensor object in MindSpore, providing methods to manipulate and query its properties.
 */
class PYBOOST_API Tensor {
 public:
  /**
   * @brief [API] Constructs a placeholder Tensor.
   *
   * This default constructor creates an undefined Tensor, which acts as a placeholder.
   */
  Tensor() = default;

  /**
   * @brief [API] Constructs a Tensor with a specified data type and shape.
   *
   * This constructor initializes a Tensor object based on the given data type
   * and shape. The resulting Tensor will be allocated but uninitialized.
   *
   * @param type_id The data type of the Tensor.
   * @param shape The shape of the Tensor, represented as a vector of integers.
   */
  Tensor(TypeId type_id, const ShapeVector &shape);

  /**
   * @brief [API] Checks if the Tensor is defined.
   *        A defined Tensor has valid data and metadata, while an undefined Tensor does not.
   * @return True if the Tensor is defined, false otherwise.
   */
  bool is_defined() const { return _tensor_holder_ != nullptr; }

  /**
   * @brief [API] Retrieves the data type of the Tensor.
   * @return The data type of the Tensor.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  TypeId data_type() const;

  /**
   * @brief [API] Retrieves the format of the Tensor.
   * @return The format of the Tensor.
   * @throws If the Tensor is not defined, use "DefaultFormat".
   */
  std::string format() const;

  /**
   * @brief [API] Set the format for the Tensor.
   * @throws If the device address is nullptr, an exception is thrown.
   */
  void set_format(const std::string &format) const;

  /**
   * @brief [API] Retrieves the shape of the Tensor.
   * @return A reference to the shape vector of the Tensor.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  const ShapeVector &shape() const;

  /**
   * @brief [API] Returns the total number of elements in the tensor.
   * @return The total number of elements.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  size_t numel() const;

  /**
   * @brief [API] Calculates the stride of the Tensor.
   * @return A vector representing the strides of the Tensor along each dimension.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  std::vector<int64_t> stride() const;

  /**
   * @brief [API] Retrieves the storage offset of the Tensor.
   * @return The offset (in terms of elements) from the beginning of the storage.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  int64_t storage_offset() const;

  /**
   * @brief [API] Checks if the Tensor is stored contiguously in memory.
   * @return True if the Tensor is stored contiguously, false otherwise.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  bool is_contiguous() const;

  /**
   * @brief [API] Sets whether the Tensor needs contiguous memory.
   *        By default, `need_contiguous` is set to true. If non-contiguous storage is required,
   *        this method should be called with `false` before invoking `ms::pynative::PyboostRunner::Call`.
   * @param flag A boolean value indicating whether the Tensor should be stored contiguously.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  void SetNeedContiguous(bool flag) const;

  /**
   * @brief [API] Retrieves a pointer to the data of the Tensor.
   * @return A void pointer to the Tensor's raw data.
   * @throws If the Tensor is null, an exception is thrown.
   */
  void *GetDataPtr() const;

  template <typename T>
  T *data_ptr() const {
    return static_cast<T *>(GetDataPtr());
  }

  /// \brief Provides a TensorAccessor for efficient multidimensional tensor access.
  ///
  /// This method returns a TensorAccessor object that allows direct and efficient access
  /// to the tensor's underlying data, using the specified data type and dimensionality.
  ///
  /// \tparam T The data type of the tensor elements (e.g., float, int, etc.).
  /// \tparam N The number of dimensions (rank) of the tensor.
  ///
  /// \note The tensor's shape and stride sizes must match the specified dimension `N`.
  ///       If not, an exception will be thrown.
  ///
  /// \return A TensorAccessor object for accessing the tensor's data.
  ///
  /// \throws std::runtime_error If the tensor's shape or stride size does not match the specified dimension `N`.
  template <typename T, size_t N>
  TensorAccessor<T, N> accessor() const {
    // Check if the shape and stride sizes match the expected dimension N
    if (shape().size() != N || stride().size() != N) {
      throw std::runtime_error(
        "TensorAccessor error: The tensor's shape or stride does not match the specified dimension N. "
        "Expected dimension: " +
        std::to_string(N) + ", but got shape size: " + std::to_string(shape().size()) +
        " and stride size: " + std::to_string(stride().size()) + ".");
    }
    // Return a TensorAccessor with the data pointer, shape, and stride
    return TensorAccessor<T, N>(data_ptr<T>(), shape().data(), std::make_shared<ShapeVector>(std::move(stride())));
  }

  /* ====== Operators based on Tensor BEGIN ====== */
 public:
  /// \brief [API] Casts the tensor to the specified data type.
  ///
  /// Converts the current tensor to the specified type `dtype` and returns the result.
  ///
  /// \note
  /// Ensure the specified data type is compatible with the tensor's current data.
  /// Casting may result in loss of precision if converting to a lower precision type.
  ///
  /// \param[in] dtype The target data type to cast the tensor to.
  ///     Supported types include float, int, etc.
  ///
  /// \return A new tensor with the specified data type.
  Tensor cast(TypeId dtype) const;

  /// \brief [API] Splits the tensor into smaller chunks along a specified dimension.
  ///
  /// Divides the tensor into `chunks` number of smaller tensors along the specified dimension `dim`.
  /// Each chunk will have an approximately equal size, except for the last chunk which may be smaller if the dimension
  /// size is not divisible by `chunks`.
  ///
  /// \param[in] chunks The number of chunks to split the tensor into. Must be positive.
  /// \param[in] dim The dimension along which the tensor is split. Defaults to 0.
  ///
  /// \return A vector of tensors containing the chunks.
  std::vector<Tensor> chunk(int64_t chunks, int64_t dim = 0) const;

  /// \brief [API] Returns a contiguous tensor in memory order.
  ///
  /// Creates a contiguous version of the current tensor, ensuring that its data is stored in contiguous memory.
  ///
  /// \return A contiguous tensor.
  Tensor contiguous() const;

  /// \brief [API] Flattens the tensor into a single dimension or over a range of dimensions.
  ///
  /// Flattens the dimensions of the tensor starting from `start_dim` to `end_dim` into a single dimension.
  /// By default, it flattens the entire tensor.
  ///
  /// \param[in] start_dim The first dimension to flatten. Defaults to 0.
  /// \param[in] end_dim The last dimension to flatten. Defaults to -1 (last dimension).
  ///
  /// \return A flattened tensor.
  Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

  /// \brief [API] Selects elements along a specified dimension using indices.
  ///
  /// Gathers elements from the tensor along dimension `dim` based on the indices specified by `index`.
  ///
  /// \note
  /// The `index` tensor must have values within the range `[0, shape(dim)-1]`.
  ///
  /// \param[in] dim The dimension along which to select elements.
  /// \param[in] index A tensor containing the indices of elements to select.
  ///
  /// \return A new tensor containing the selected elements.
  Tensor index_select(int64_t dim, const Tensor &index) const;

  /// \brief [API] Reshapes the tensor to the specified shape.
  ///
  /// Returns a tensor with the same data but with a new shape defined by `shape`.
  ///
  /// \note
  /// The total number of elements in the new shape must match the original tensor's size.
  /// Use `-1` in `shape` to infer one dimension automatically.
  ///
  /// \param[in] shape A vector specifying the new shape. One dimension can be set to `-1` for automatic inference.
  ///
  /// \return A reshaped tensor.
  Tensor reshape(const std::vector<int64_t> &shape) const;

  /// \brief [API] Repeats the tensor along specified dimensions.
  ///
  /// Creates a new tensor by repeating the current tensor along each dimension as specified in `repeats`.
  ///
  /// \note
  /// The size of `repeats` must match the number of dimensions of the tensor.
  ///
  /// \param[in] repeats A vector specifying the number of repetitions for each dimension.
  ///
  /// \return A new tensor with repeated elements.
  Tensor repeat(const std::vector<int64_t> &repeats) const;

  /// \brief [API] Repeats elements of the tensor along a specified dimension.
  ///
  /// Repeats each element of the tensor the number of times specified by tensor `repeats`, it specifies the number of
  /// repetitions for each element in the dimension.
  ///
  /// \note
  /// The size of the `repeats` tensor must match the size of the tensor along `dim` if specified.
  ///
  /// \param[in] repeats A tensor or scalar specifying the number of repetitions for each element.
  /// \param[in] dim (Optional) The dimension along which to repeat elements.
  /// \param[in] output_size (Optional) The size of the output tensor along `dim`.
  ///
  /// \return A new tensor with repeated elements.
  Tensor repeat_interleave(const Tensor &repeats, const std::optional<int64_t> &dim = std::nullopt,
                           const std::optional<int64_t> &output_size = std::nullopt) const;

  /// \brief [API] Repeats elements of the tensor a fixed number of times along a specified dimension.
  ///
  /// Similar to the overloaded version, but `repeats` is a scalar that specifies the number of repetitions for all
  /// elements along the dimension `dim`.
  ///
  /// \param[in] repeats A scalar specifying the number of repetitions for each element.
  /// \param[in] dim (Optional) The dimension along which to repeat elements.
  /// \param[in] output_size (Optional) The size of the output tensor along `dim`.
  ///
  /// \return A new tensor with repeated elements.
  Tensor repeat_interleave(int64_t repeats, const std::optional<int64_t> &dim = std::nullopt,
                           const std::optional<int64_t> &output_size = std::nullopt) const;

  /* ====== Operators based on Tensor END ====== */
 public:
  /**
   * @brief Constructs a Tensor object from a given ValuePtr.
   * @param value A smart pointer to a MindSpore Value object. If the value is null, an undefined Tensor is constructed.
   *              Default nullptr.
   */
  explicit Tensor(const mindspore::ValuePtr &value);

  /**
   * @brief Deconstructor
   */
  ~Tensor() = default;

  /**
   * @brief Checks if the Tensor requires contiguous memory.
   * @return True if the Tensor needs to be stored contiguously, false otherwise.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  bool need_contiguous() const;

  /**
   * @brief Retrieves the stub node associated with the Tensor.
   * @return A smart pointer to the stub node (ValuePtr).
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  const mindspore::ValuePtr &stub_node() const;

  /**
   * @brief Retrieves the underlying tensor object.
   * @return A smart pointer to the TensorPtr object.
   * @throws If the Tensor is not defined, an exception is thrown.
   */
  const mindspore::tensor::TensorPtr &tensor() const;

  /**
   * @brief Converts the stub node to a Tensor object.
   *        This ensures that the Tensor is fully realized from its stub representation.
   *        After conversion, the stub node is released.
   */
  void ConvertStubNodeToTensor() const;

  /// \brief Assigns the value of another tensor to the current tensor.
  /// \param[in] src The source tensor whose value will be assigned to the current tensor.
  void AssignTensor(const Tensor &src) const;

 private:
  /**
   * @struct RealTensorHolder
   * @brief Holds the actual data and metadata of the Tensor object.
   */
  struct PYBOOST_API RealTensorHolder {
    explicit RealTensorHolder(const mindspore::ValuePtr &value);

    // Indicates if the Tensor data needs to be contiguous. Defaults to true.
    bool need_contiguous_{true};
    // The value associated with the Tensor.
    mindspore::ValuePtr value_{nullptr};
    // The underlying Tensor object.
    mindspore::tensor::TensorPtr tensor_{nullptr};
  };

  // Shared pointer to the Tensor's holder.
  std::shared_ptr<RealTensorHolder> _tensor_holder_{nullptr};
};
}  // namespace ms

namespace pybind11 {
namespace detail {
template <>
struct PYBOOST_API type_caster<ms::Tensor> {
  PYBIND11_TYPE_CASTER(ms::Tensor, _("Tensor"));
  bool load(handle src, bool);
  static handle cast(const ms::Tensor &src, return_value_policy, handle);
};
}  // namespace detail
}  // namespace pybind11
#endif  // MINDSPORE_CCSRC_EXTENSION_TENSOR_H_
