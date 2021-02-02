/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#undef HAVE_STDDEF_H
#undef HAVE_STDLIB_H
#endif

#include "include/constants.h"
#include "include/data_type.h"
#include "include/tensor_helpers.h"
#include "include/tensor_shape.h"
#include "include/status.h"

namespace mindspore {
namespace dataset {
class Tensor;
template <typename T>
class Allocator;

using CharAllocPtr = std::unique_ptr<Allocator<unsigned char>>;
using TensorAllocPtr = std::shared_ptr<Allocator<Tensor>>;  // An allocator shared_ptr for Tensors
using offset_t = uint32_t;                                  // type of offset values to store strings locations
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor {
 public:
  Tensor() = delete;
  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Create a tensor using shape and type. This constructor should not be used directly, use CreateFromTensor instead
  /// \note The shape and type information should be known and valid
  /// \note The constructor does not allocate data
  /// \param shape TensorShape
  /// \param type DataType
  Tensor(const TensorShape &shape, const DataType &type);

  /// Move constructor
  /// \param other Tensor to be moved
  Tensor(Tensor &&other) noexcept;

  /// Move assignment operator
  /// \param other Tensor to be moved
  Tensor &operator=(Tensor &&other) noexcept;

  /// Create a numeric tensor with type and shape. Items of the tensor would be uninitialized.
  /// \param[in] shape shape of the output tensor
  /// \param[in] type type of the output tensor
  /// \param[out] out Generated tensor
  /// \return Status code
  static Status CreateEmpty(const TensorShape &shape, const DataType &type, TensorPtr *out);

  /// Create a numeric tensor from a pointer in memory. Length of the source data is determined from the shape and type.
  /// Data will be copied into the new created tensor.
  /// \param[in] shape shape of the output tensor
  /// \param[in] type type of the output tensor
  /// \param[in] src pointer to the source data
  /// \param[out] out Generated tensor
  /// \return Status code
  static Status CreateFromMemory(const TensorShape &shape, const DataType &type, const uchar *src, TensorPtr *out);

  /// Create a tensor from a pointer in memory and length. Data will be copied into the new created tensor.
  /// \param[in] shape shape of the output tensor
  /// \param[in] type type of the output tensor
  /// \param[in] src pointer to the source data
  /// \param[in] length length of the src data
  /// \param[out] out Generated tensor
  /// \return Status code
  static Status CreateFromMemory(const TensorShape &shape, const DataType &type, const uchar *src,
                                 const dsize_t &length, TensorPtr *out);

  /// Create a copy of the input tensor
  /// \param[in] in original tensor to be copied
  /// \param[out] out output tensor to be generated
  /// \return Status
  static Status CreateFromTensor(const TensorPtr &in, TensorPtr *out) {
    return CreateFromMemory(in->shape(), in->type(), in->GetBuffer(), in->SizeInBytes(), out);
  }

  /// Create a Tensor from a given list of values.
  /// \tparam type of the values to be inserted.
  /// \param[in] items elements of the tensor
  /// \param[in] shape shape of the output tensor
  /// \param[out] out output argument to hold the created Tensor
  /// \return Status Code
  template <typename T>
  static Status CreateFromVector(const std::vector<T> &items, const TensorShape &shape, TensorPtr *out) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      items.size() == shape.NumOfElements(),
      "Number of elements in the vector does not match the number of elements of the shape required");
    // cppcheck-suppress shadowFunction
    DataType type = DataType::FromCType<T>();
    // if items is empty, items_ptr would be nullptr. CreateFromMemory will handle this case.
    auto items_ptr = reinterpret_cast<const uchar *>(&items[0]);
    return CreateFromMemory(shape, type, items_ptr, out);
  }

  /// Create a 1D Tensor from a given list of values.
  /// \tparam type of the values to be inserted.
  /// \param[in] items elements of the tensor
  /// \param[out] out output argument to hold the created Tensor
  /// \return Status Code
  template <typename T>
  static Status CreateFromVector(const std::vector<T> &items, TensorPtr *out) {
    return CreateFromVector(items, TensorShape({static_cast<dsize_t>(items.size())}), out);
  }

  /// Create a 1D boolean Tensor from a given list of boolean values.
  /// \param[in] items elements of the tensor
  /// \param[in] shape shape of the output tensor
  /// \param[out] out output argument to hold the created Tensor
  /// \return Status Code
  static Status CreateFromVector(const std::vector<bool> &items, const TensorShape &shape, TensorPtr *out) {
    std::vector<uint8_t> temp(items.begin(), items.end());
    RETURN_IF_NOT_OK(CreateFromVector(temp, shape, out));
    (*out)->type_ = DataType(DataType::DE_BOOL);
    return Status::OK();
  }

  /// Create a numeric scalar Tensor from the given value.
  /// \tparam T type of value
  /// \param[in] item value
  /// \param[out] out Created tensor
  /// \return Status code
  template <typename T>
  static Status CreateScalar(const T &item, TensorPtr *out) {
    // cppcheck-suppress shadowFunction
    DataType type = DataType::FromCType<T>();
    auto item_ptr = reinterpret_cast<const uchar *>(&item);
    return CreateFromMemory(TensorShape::CreateScalar(), type, item_ptr, out);
  }

  /// Create a tensor from a binary file on disk.
  /// \param[in] path file to be read
  /// \param[out] out Created Tensor
  /// \return Status code
  static Status CreateFromFile(const std::string &path, TensorPtr *out);

  /// Destruct the tensor and release the memory using the allocator
  virtual ~Tensor();

  /// Equality operator. compares tensor shape, type and data
  /// \param[in] rhs Tensor to be compared with
  /// \return bool
  bool operator==(const Tensor &rhs) const;

  bool operator!=(const Tensor &rhs) const { return !((*this) == rhs); }

  /// Get item located at `index`, caller needs to provide the type.
  /// \tparam T
  /// \param[in] index vector<dsize_t>
  /// \return return the item specified at index
  template <typename T>
  Status GetItemAt(T *o, const std::vector<dsize_t> &index) const;

  /// Get string located at `index`.
  /// \param[in] index vector<dsize_t>
  /// \return return std::string_view specified at index
  Status GetItemAt(std::string_view *o, const std::vector<dsize_t> &index) const;

  template <typename T>
  Status GetUnsignedIntAt(T *o, const std::vector<dsize_t> &index) const;

  template <typename T>
  Status GetSignedIntAt(T *o, const std::vector<dsize_t> &index) const;

  template <typename T>
  Status GetFloatAt(T *o, const std::vector<dsize_t> &index) const;

  /// set item at location specified by index
  /// \tparam `T`
  /// \param[in] index
  /// \param[in] value of type `T`
  template <typename T>
  Status SetItemAt(const std::vector<dsize_t> &index, const T &value) {
    T *ptr = nullptr;
    RETURN_IF_NOT_OK(GetItemPtr<T>(&ptr, index));
    *ptr = value;
    return Status::OK();
  }

  Status SetItemAt(const std::vector<dsize_t> &index, const std::string &value);

  /// fill tensor with Zeros. Does not support strings.
  Status Zero();

  /// Fill all elements in the Tensor with the given value of type `T`.  Does not support strings.
  /// \tparam T
  /// \param value[in]
  template <typename T>
  Status Fill(const T &value);

  /// Getter function for shape
  /// \return
  const TensorShape &shape() const { return shape_; }

  /// Check if tensor has data
  /// \return bool - true if tensor is not empty
  bool HasData() const { return data_ != nullptr; }

  /// Reshape the tensor. The given shape should have the same number of elements in the Tensor
  /// \param shape
  virtual Status Reshape(const TensorShape &shape);

  /// \return number of elements in this tensor
  dsize_t Size() const { return shape().NumOfElements(); }

  /// \return the number of bytes this tensor is needs
  dsize_t SizeInBytes() const {
    if (data_end_ == nullptr) return type_.SizeInBytes() * shape_.NumOfElements();
    return data_end_ - data_;
  }

  /// \return the rank of the tensor
  dsize_t Rank() const { return shape().Rank(); }

  /// Get the starting memory address as a constant for the data of the tensor.  This potentially
  /// drives an allocation if the data area.
  /// \return const unsigned char*
  const unsigned char *GetBuffer() const { return data_; }

  /// Getter of the type
  /// \return
  // cppcheck-suppress shadowFunction
  DataType type() const { return type_; }

  /// Provide stream operator for displaying it
  /// \param output stream
  /// \param so the Tensor object to be printed
  /// \return output stream
  friend std::ostream &operator<<(std::ostream &out, const Tensor &so) {
    so.Print(out);
    return out;
  }

  /// Invalidate this Tensor by setting the type and shape to unknown and MData to null.
  /// Calling this method will make the Tensor and its data inaccessible, use it with caution.
  void Invalidate();

  /// Copy input tensor into self at the location index.
  /// Index is a vector of axes which can be incomplete:
  /// Ex: shape <2,3>, inserting into index {0} will replace the first row. index {1,2} will replace the last cell.
  /// \param index
  /// \param input
  /// \param partial_insert: boolean to determine if insertion along the full axis is enforced
  /// \return Status code
  Status InsertTensor(const std::vector<dsize_t> &index, const std::shared_ptr<Tensor> &input,
                      const bool partial_insert = false);

  /// Find the address of the given index. Used in InsertTensor.
  /// Example:
  ///      Tensor t= [[1,2],[3,4]] , StartAddrOfIndex({0}) -> &1
  /// \param index  incomplete index
  /// \param output: startAddrofIndex
  /// \param output: remaining
  /// \return Status code
  Status StartAddrOfIndex(std::vector<dsize_t> ind, uchar **start_addr_of_index, TensorShape *remaining);

  /// Expand the shape of the Tensor with one extra dimension.
  /// For example, if the shape is <512,512,3>:
  ///     *- ExpandDim(0) gives: <1,512,512,3>
  ///     *- ExpandDim(1) gives: <512,1,512,3>
  ///     *- ExpandDim(3) gives: <512,512,3,1>
  /// \param axis location of the dim
  virtual Status ExpandDim(const dsize_t &axis);

  virtual void Squeeze();

  /// Calculates the strides of the Tensor
  /// Ex: Tensor of shape <4,2,2> and type DE_UINT8 (1 byte)
  /// The strides will be {6,2,1}.
  /// Ex: Tensor of shape <4,2,2> and type DE_UINT32 (4 byte)
  /// The strides will be {24,8,4}.
  /// \return vector of integers
  std::vector<dsize_t> Strides() const;

  std::string ToString() {
    std::stringstream ss;
    this->Print(ss);
    return ss.str();
  }

  /// Handle negative indices.
  /// \param[out] out modified index
  /// \param[in] index
  /// \param[in] length axis length used to modify index
  /// \return dsize_t modified index
  static inline dsize_t HandleNeg(dsize_t index, dsize_t length) { return (index < 0) ? (index + length) : index; }

  /// Handle negative indices for a vector of indices.
  /// \param[out] out modified vector of indices
  /// \param[in] index_vector vector of indices
  /// \return std::vector<dsize_t> modified vector of indices
  static inline std::vector<dsize_t> HandleNegIndices(std::vector<dsize_t> index_vector, std::vector<dsize_t> length) {
    std::vector<dsize_t> indices(index_vector.size(), 0);
    for (int i = 0; i < index_vector.size(); i++) {
      indices[i] = HandleNeg(index_vector[i], length[i]);
    }
    return indices;
  }

  /// Slice tensor bases on the given indices. Copy the sliced data into out tensor.
  /// Based on the type of tensor, SliceNumeric or SliceString will be called
  /// \param[out] out Tensor
  /// \param[in] slice_options vector of SliceOption objects
  /// \return Status error code
  // cppcheck-suppress passedByValue
  Status Slice(TensorPtr *out, const std::vector<mindspore::dataset::SliceOption> slice_options);

  /// TensorIterator is a linear iterator that can be used to iterate over the elements of the Tensor
  /// The order  elements  is as the memory layout (i.e., row-major) [[1,2,3],[4,5,6] --> 1,2,3,4,5,6
  /// \tparam T type of values in the Tensor Iterator
  template <typename T, bool = true>
  class TensorIterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    explicit TensorIterator(uchar *ptr = nullptr) { ptr_ = reinterpret_cast<T *>(ptr); }

    TensorIterator(const TensorIterator<T> &raw_iterator) { ptr_ = raw_iterator.ptr_; }

    ~TensorIterator() = default;

    // cppcheck-suppress operatorEqVarError
    TensorIterator<T> &operator=(const TensorIterator<T> &rhs) {
      ptr_ = rhs.ptr_;
      return *this;
    }

    TensorIterator<T> &operator=(T *rhs) {
      ptr_ = rhs;
      return *this;
    }

    bool operator==(const TensorIterator<T> &rhs) { return ptr_ == rhs.ptr_; }

    bool operator!=(const TensorIterator<T> &rhs) { return !(*this == rhs); }

    operator bool() const { return ptr_ != nullptr; }

    T &operator*() { return *ptr_; }

    const T &operator*() const { return *ptr_; }

    T *operator->() { return ptr_; }

    TensorIterator<T> &operator+=(const ptrdiff_t &inc) {
      ptr_ += inc;
      return *this;
    }

    TensorIterator<T> &operator-=(const ptrdiff_t &inc) {
      ptr_ -= inc;
      return *this;
    }

    TensorIterator<T> &operator++() {
      ++ptr_;
      return *this;
    }

    TensorIterator<T> &operator--() {
      --ptr_;
      return *this;
    }

    TensorIterator<T> operator++(int) {
      auto temp(*this);
      ++ptr_;
      return temp;
    }

    TensorIterator<T> operator--(int) {
      auto temp(*this);
      --ptr_;
      return temp;
    }

    TensorIterator<T> operator+(const ptrdiff_t &inc) {
      auto oldPtr = ptr_;
      ptr_ += inc;
      auto temp(*this);
      ptr_ = oldPtr;
      return temp;
    }

    TensorIterator<T> operator-(const ptrdiff_t &inc) {
      auto oldPtr = ptr_;
      ptr_ -= inc;
      auto temp(*this);
      ptr_ = oldPtr;
      return temp;
    }

   protected:
    T *ptr_;
  };

  // Specialization of TensorIterator for strings. It returns std::string_view for every item.
  // \tparam DUMMY, used to mbe able to specialize the inner class
  template <bool DUMMY>
  class TensorIterator<std::string_view, DUMMY> {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::string_view;
    using difference_type = ptrdiff_t;
    using pointer = std::string_view *;
    using reference = std::string_view &;

    explicit TensorIterator(uchar *data = nullptr, dsize_t index = 0) {
      data_ = reinterpret_cast<const char *>(data);
      // cppcheck-suppress useInitializationList
      index_ = index;
    }

    TensorIterator(const TensorIterator<std::string_view, DUMMY> &raw_iterator) {
      data_ = raw_iterator.data_;
      // cppcheck-suppress useInitializationList
      index_ = raw_iterator.index_;
    }

    ~TensorIterator() = default;

    bool operator==(const TensorIterator<std::string_view> &rhs) { return data_ == rhs.data_ && index_ == rhs.index_; }

    bool operator!=(const TensorIterator<std::string_view> &rhs) { return !(*this == rhs); }

    operator bool() const { return data_ != nullptr; }

    std::string_view operator*() const {
      auto offset_ = reinterpret_cast<const offset_t *>(data_);
      offset_t start = offset_[index_];
      return std::string_view{data_ + start};
    }

    TensorIterator<std::string_view> &operator+=(const dsize_t &inc) {
      index_ += inc;
      return *this;
    }

    TensorIterator<std::string_view> &operator-=(const dsize_t &inc) {
      index_ -= inc;
      return *this;
    }

    TensorIterator<std::string_view> &operator++() {
      ++index_;
      return *this;
    }

    TensorIterator<std::string_view> &operator--() {
      --index_;
      return *this;
    }

    TensorIterator<std::string_view> operator++(int) {
      auto temp(*this);
      ++index_;
      return temp;
    }

    TensorIterator<std::string_view> operator--(int) {
      auto temp(*this);
      --index_;
      return temp;
    }

    TensorIterator<std::string_view> operator+(const dsize_t &inc) {
      auto oldPtr = index_;
      index_ += inc;
      auto temp(*this);
      index_ = oldPtr;
      return temp;
    }

    TensorIterator<std::string_view> operator-(const dsize_t &inc) {
      auto oldPtr = index_;
      index_ -= inc;
      auto temp(*this);
      index_ = oldPtr;
      return temp;
    }

   protected:
    dsize_t index_;
    const char *data_;
  };

  /// Return a TensorIterator that points to the start of the Tensor.
  /// It's the user responsibility to use the correct type that matches the Tensor type
  /// \tparam T The type of values in the Tensor
  /// \return TensorIterator
  template <typename T>
  TensorIterator<T> begin() {
    return TensorIterator<T>(data_);
  }

  /// Return a linear iterator that points to the place after the last element of the Tensor.
  /// \tparam T The type of values in the Tensor
  /// \return TensorIterator
  template <typename T>
  TensorIterator<T> end() {
    return TensorIterator<T>(data_end_);
  }

  /// Copies the last dimension at `index` from Tensor `src` to this Tensor.
  /// \param[in] src Tensor
  /// \param[in] index vector to the start of the dimension. The last dim should be 0
  /// \return Status
  Status CopyLastDimAt(const std::shared_ptr<Tensor> &src, const std::vector<dsize_t> &index);

 protected:
  /// Allocate memory for the tensor using the data_allocator
  /// \param[in] length number of bytes to be allocated
  /// \return Error Status
  Status AllocateBuffer(const dsize_t &length);

  /// Get the starting memory address for the data of the tensor.  This potentially
  /// drives an allocation if the data is null.
  /// \return unsigned char*
  unsigned char *GetMutableBuffer() { return data_; }

  /// A function that prints Tensor recursively, first called by print
  /// \param[in] out
  /// \param[in] cur_dim
  /// \param[in] cur_index
  void PrintRecursive(std::ostream &out, int32_t cur_dim, const std::vector<dsize_t> &cur_index) const;

  /// A function that prints info about the tensor
  /// \param[out] out output stream
  void Print(std::ostream &out) const;

  /// A function that print the value as specified by its index
  /// \param[in] index vector representing the index
  /// \param[out] out
  void PrintItemAt(const std::vector<dsize_t> &index, std::ostream &out) const;

  /// Get pointer to item located at `index`, caller needs to provide the type.
  /// \tparam T
  /// \param[in] index vector<dsize_t>
  /// \return return a pointer to the item specified at index of type `T`
  template <typename T>
  Status GetItemPtr(T **, const std::vector<dsize_t> &index) const;

  /// Get pointer to string located at `index` and the length of string
  /// \param[in] index vector<dsize_t>
  /// \return return a pointer to the string specified at index and the length of the string
  Status GetItemPtr(uchar **, const std::vector<dsize_t> &index, offset_t *length = nullptr) const;

  /// Given a flat index of an item string, return the start and length of the item
  /// \param[in] index flat index of the item
  /// \param[out] start address of the ths string
  /// \param[out] length of the string
  Status GetStringAt(dsize_t index, uchar **string_start, offset_t *length) const;

  /// Skip the offsets and returns the start of the buffer where the real strings is stored. Caller needs to check if
  /// the tensor's type is a string, otherwise undefined address would be returned. \return address of the first string
  /// of the tensor.
  uchar *GetStringsBuffer() const { return data_ + kOffsetSize * shape_.NumOfElements() + kOffsetSize; }

  /// all access to shape_ should be via shape
  TensorShape shape_;
  /// data type of tensor
  DataType type_;
  /// pointer to the start of the physical data
  unsigned char *data_;
  /// An allocator for data_
  CharAllocPtr data_allocator_;
  /// pointer to the end of the physical data
  unsigned char *data_end_ = nullptr;

 private:
  /// Slice numeric tensors.
  Status SliceNumeric(TensorPtr *out, const std::vector<std::vector<dsize_t>> &indices, const TensorShape &shape);

  /// Slice string tensors
  Status SliceString(TensorPtr *out, const std::vector<std::vector<dsize_t>> &indices, const TensorShape &shape);

  /// Copy raw data of a array based on shape and strides to the destination pointer
  /// \param dst [out] Pointer to the destination array where the content is to be copied
  /// \param[in] src Pointer to the source of strided array to be copied
  /// \param[in] shape shape of the source array
  /// \param[in] strides strides of the source array
  /// \param[in] type_size number of bytes needed to store one array element's type
  /// \return Status Code
  static Status CopyStridedArray(unsigned char *dst, unsigned char *src, std::vector<dsize_t> shape,
                                 std::vector<dsize_t> strides, uint8_t type_size);

  /// const of the size of the offset variable
  static constexpr uint8_t kOffsetSize = sizeof(offset_t);
};
template <>
inline Tensor::TensorIterator<std::string_view> Tensor::end<std::string_view>() {
  return TensorIterator<std::string_view>(data_, shape_.NumOfElements());
}

/// Create a string scalar Tensor from the given value.
/// \param[in] item value
/// \param[out] out Created tensor
/// \return Status code
template <>
inline Status Tensor::CreateScalar<std::string>(const std::string &item, TensorPtr *out) {
  return CreateFromVector<std::string>({item}, TensorShape::CreateScalar(), out);
}
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_H_
