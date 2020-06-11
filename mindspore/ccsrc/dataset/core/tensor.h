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
#ifndef DATASET_CORE_TENSOR_H_
#define DATASET_CORE_TENSOR_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "./securec.h"
#include "utils/log_adapter.h"
#if defined(_WIN32) || defined(_WIN64)
#undef HAVE_STDDEF_H
#undef HAVE_STDLIB_H
#endif
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "dataset/core/constants.h"
#include "dataset/core/data_type.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/util/allocator.h"
#include "dataset/util/de_error.h"
#include "dataset/util/status.h"
#include "proto/example.pb.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {
class Tensor;

using CharAllocPtr = std::unique_ptr<Allocator<unsigned char>>;
using TensorAllocPtr = std::shared_ptr<Allocator<Tensor>>;  // An allocator shared_ptr for Tensors
using TensorRow = std::vector<std::shared_ptr<Tensor>>;     // A row is a set of Tensor pointers
using TensorTable = std::vector<TensorRow>;                 // The table of tensors is a vector of rows
using TensorQTable = std::deque<TensorRow>;  // A different flavour of tensor table, this one has queue functionality

class Tensor {
 public:
  Tensor() = delete;

  // Create a new tensor, does not internally allocate storage. This constructor is protected, use CreateTensor.
  // @note The shape and type information should be known and valid.
  // @param shape TensorShape
  // @param type DataType
  Tensor(const TensorShape &shape, const DataType &type);

  // Create a new tensor, allocates storage and copies in data. This constructor is protected, use CreateTensor.
  // @note The buffer should be valid and the shape and type information should be known and valid.
  // @param shape TensorShape
  // @param type DataType
  // @param data unsigned char*, pointer to the data.
  Tensor(const TensorShape &shape, const DataType &type, const unsigned char *data);

  Tensor(const TensorShape &shape, const DataType &type, const unsigned char *data, const dsize_t &length);

  Tensor(const Tensor &other) = delete;

  Tensor &operator=(const Tensor &other) = delete;

  Tensor(Tensor &&other) noexcept;

  Tensor &operator=(Tensor &&other) noexcept;

  Status AllocateBuffer(const dsize_t &length);

  // type of offest values to store strings information
  using offset_t = uint32_t;
  // const of the size of the offset variable
  static constexpr uint8_t kOffsetSize = sizeof(offset_t);
  // Tensor base class which holds the data in an unsigned char* buffer.

  // Construct  a scalar string Tensor
  explicit Tensor(const std::string &str) : Tensor(std::vector<std::string>{str}, TensorShape::CreateScalar()) {}

  // Construct a tensor from  a list of strings. Reshape the tensor with `shape` if given, otherwise assume the shape is
  // the size of the vector `strings`.
  // The memory layout of a Tensor of strings consists of the Offset_array followed by the strings.
  // Thr offset array will store one extra value to find the length of the last string.
  // OFFSET1, OFFSET2, ..., OFFSETn+1, STRING1, STRING2, ..., STRINGn
  // The value of each offset is the start index of the corresponding string
  // Offsets is of type offest_t
  // strings will ne null-terminated
  // example: Tensor(['abc', 'de'], shape={2}, type=DE_STRING)
  // |----------------------------------------------------------------|
  // |             OFFSET ARRAY           |            STRINGS        |
  // | bytes 0-3 | bytes 3-6 | bytes 7-10 | bytes 11-14 | bytes 15-17 |
  // |     11    |    15     |     18     |     abc\0   |      de\0   |
  // |----------------------------------------------------------------|
  explicit Tensor(const std::vector<std::string> &strings,
                  const TensorShape &shape = TensorShape::CreateUnknownRankShape());

  // Same as Tensor(vector<string>) but the input is protobuf bytelist
  explicit Tensor(const dataengine::BytesList &bytes_list,
                  const TensorShape &shape = TensorShape::CreateUnknownRankShape());

  // A static factory method to create the given flavour of derived Tensor
  // Returns the base class reference for the Tensor.
  // @param ptr output argument to hold the created Tensor of given tensor_impl
  // @param tensor_impl - which implementation of Tensor
  // @param shape - shape of the tensor
  // @param type - datatype of the tensor
  // @param data - data to be copied to Tensor new allocation
  // @return Status Code
  static Status CreateTensor(std::shared_ptr<Tensor> *, TensorImpl tensor_impl, const TensorShape &shape, DataType type,
                             const unsigned char *data = nullptr);

  // A static factory method to create a Tensor from a given py::array.
  // @param ptr output argument to hold the created Tensor
  // @param arr py::array
  // @return Status Code
  static Status CreateTensor(std::shared_ptr<Tensor> *ptr, py::array arr);

  // Helper function to create a tensor from Numpy of strings
  static Status CreateTensorFromNumpyString(std::shared_ptr<Tensor> *ptr, py::array arr);

  // A static factory method to create a Tensor from a given list of strings.
  // @param ptr output argument to hold the created Tensor
  // @param strings elements of the tensor
  // @param shape shape of the tensor
  // @return Status Code
  static Status CreateTensor(std::shared_ptr<Tensor> *ptr, const std::vector<std::string> &strings,
                             const TensorShape &shape = TensorShape::CreateUnknownRankShape());

  // create tensor from protobuf bytelist with strings
  static Status CreateTensor(std::shared_ptr<Tensor> *ptr, const dataengine::BytesList &bytes_list,
                             const TensorShape &shape);

  // A static factory method to create a Tensor from a given list of numbers.
  // @param ptr output argument to hold the created Tensor
  // @param items elements of the tensor
  // @param shape shape of the tensor
  // @return Status Code
  template <typename T>
  static Status CreateTensor(std::shared_ptr<Tensor> *ptr, const std::vector<T> &items,
                             const TensorShape &shape_req = TensorShape::CreateUnknownRankShape()) {
    DataType type = DataType::FromCType<T>();
    auto items_ptr = reinterpret_cast<const uchar *>(&items[0]);
    TensorShape shape = shape_req;
    if (!shape.known()) {
      shape = TensorShape({static_cast<dsize_t>(items.size())});
    }
    return CreateTensor(ptr, TensorImpl::kFlexible, shape, type, items_ptr);
  }

  // A static factory method to create a Tensor from a given number.
  // @param ptr output argument to hold the created Tensor
  // @param item value
  // @return Status Code
  template <typename T>
  static Status CreateTensor(std::shared_ptr<Tensor> *ptr, const T &item) {
    return CreateTensor<T>(ptr, {item}, TensorShape::CreateScalar());
  }
  // Create tensor from protobuf bytelist with uint8 or int8 types
  static Status CreateTensor(std::shared_ptr<Tensor> *ptr, const dataengine::BytesList &bytes_list,
                             const TensorShape &shape, const DataType &type, dsize_t pad_size);

  static Status CreateTensor(std::shared_ptr<Tensor> *ptr, const std::string &path);

  // Copy raw data of a array based on shape and strides to the destination pointer
  // @param dst Pointer to the destination array where the content is to be copied
  // @param src Pointer to the source of strided array to be copied
  // @param shape - shape of the source array
  // @param strides - strides of the source array
  // @param type_size - number of bytes needed to store one array element's type
  // @return Status Code
  static Status CopyStridedArray(unsigned char *dst, unsigned char *src, std::vector<dsize_t> shape,
                                 std::vector<dsize_t> strides, uint8_t type_size);

  // Release the memory using the allocator
  virtual ~Tensor();

  // compare the tensor shape and data
  bool operator==(const Tensor &rhs) const;

  bool operator!=(const Tensor &rhs) const { return !((*this) == rhs); }

  // Get item located at `index`, caller needs to provide the type.
  // @tparam T
  // @param index vector<dsize_t>
  // @return return the item specified at index
  template <typename T>
  Status GetItemAt(T *o, const std::vector<dsize_t> &index) const;

  // Get string located at `index`.
  // @param index vector<dsize_t>
  // @return return std::string_view specified at index
  Status GetItemAt(std::string_view *o, const std::vector<dsize_t> &index) const;

  template <typename T>
  Status GetUnsignedIntAt(T *o, const std::vector<dsize_t> &index) const;

  template <typename T>
  Status GetSignedIntAt(T *o, const std::vector<dsize_t> &index) const;

  template <typename T>
  Status GetFloatAt(T *o, const std::vector<dsize_t> &index) const;

  // set item at location specified by index
  // @tparam `T`
  // @param index
  // @param value of type `T`
  template <typename T>
  Status SetItemAt(const std::vector<dsize_t> &index, const T &value) {
    RETURN_IF_NOT_OK(AllocateBuffer(SizeInBytes()));
    T *ptr = nullptr;
    RETURN_IF_NOT_OK(GetItemPtr<T>(&ptr, index));
    *ptr = value;
    return Status::OK();
  }

  // set string item at location specified by index
  // @param index
  // @param value of type std::string
  Status SetItemAt(const std::vector<dsize_t> &index, const std::string &value) {
    RETURN_UNEXPECTED_IF_NULL(data_);
    uchar *ptr = nullptr;
    offset_t length = 0;
    RETURN_IF_NOT_OK(GetItemPtr(&ptr, index, &length));
    if (value.length() != length) {
      RETURN_STATUS_UNEXPECTED("Length of the new string does not match the item.");
    }
    memcpy_s(reinterpret_cast<char *>(ptr), length, value.c_str(), length);

    return Status::OK();
  }
  // fill tensor with Zeros. Does not support strings.
  Status Zero() {
    CHECK_FAIL_RETURN_UNEXPECTED(type_ != DataType::DE_STRING, "Cannot use Zero on tensor of strings..");
    dsize_t size = SizeInBytes();
    CHECK_FAIL_RETURN_UNEXPECTED(memset_sp(GetMutableBuffer(), size, 0, size) == 0,
                                 "Failed to fill tensor with zeroes.");
    return Status::OK();
  }

  // Fill all elements in the Tensor with the given value of type `T`.  Does not support strings.
  // @tparam T
  // @param value
  template <typename T>
  Status Fill(const T &value) {
    CHECK_FAIL_RETURN_UNEXPECTED(type_ != DataType::DE_STRING, "Cannot use fill on tensor of strings.");
    RETURN_IF_NOT_OK(AllocateBuffer(SizeInBytes()));
    int64_t cellSize = type_.SizeInBytes();
    if ((data_ != nullptr) && type_.IsCompatible<T>()) {
      for (dsize_t i = 0; i < Size(); i++) {
        CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s((data_ + i * cellSize), cellSize, &value, cellSize) == 0, "memcpy err");
      }
      return Status::OK();
    } else {
      std::string err;
      err += (data_ == nullptr) ? "data_ is nullptr \t" : "";
      err += type_.IsCompatible<T>() ? "data type not compatible\t" : "";
      return Status(StatusCode::kUnexpectedError, err);
    }
  }

  // Getter function for shape
  // @return
  const TensorShape &shape() const { return shape_; }

  // Reshape the tensor. The given shape should have the same number of elements in the Tensor
  // @param shape
  virtual Status Reshape(const TensorShape &shape);

  // @return number of elements in this tensor
  dsize_t Size() const { return shape().NumOfElements(); }

  // @return the number of bytes this tensor is needs
  dsize_t SizeInBytes() const {
    if (data_end_ == nullptr) return type_.SizeInBytes() * shape_.NumOfElements();
    return data_end_ - data_;
  }

  // @return the rank of the tensor
  dsize_t Rank() const { return shape().Rank(); }

  // Get the starting memory address as a constant for the data of the tensor.  This potentially
  // drives an allocation if the data area.
  // @return const unsigned char*
  const unsigned char *GetBuffer() const;

  // Getter of the type
  // @return
  DataType type() const { return type_; }

  // Provide stream operator for displaying it
  // @param output stream
  // @param so the Tensor object to be printed
  // @return output stream
  friend std::ostream &operator<<(std::ostream &out, const Tensor &so) {
    so.Print(out);
    return out;
  }

  // Invalidate this Tensor by setting the type and shape to unknown and MData to null.
  // Calling this method will make the Tensor and its data inaccessible, use it with caution.
  void Invalidate();

  // Copy input tensor into self at the location index.
  // Index is a vector of axises which can be incomplete:
  // Ex: shape <2,3>, inserting into index {0} will replace the first row. index {1,2} will replace the last cell.
  // @param index
  // @param input
  // @return Status code
  Status InsertTensor(const std::vector<dsize_t> &index, const std::shared_ptr<Tensor> &input);

  // Find the address of the given index. Used in InsertTensor.
  // Example:
  //      Tensor t= [[1,2],[3,4]] , StartAddrOfIndex({0}) -> &1
  // @param index  incomplete index
  // @param output: startAddrofIndex
  // @param output: remaining
  // @return Status code
  Status StartAddrOfIndex(std::vector<dsize_t> ind, uchar **start_addr_of_index, TensorShape *remaining);

  // Expand the shape of the Tensor with one extra dimension.
  // For example, if the shape is <512,512,3>:
  //     *- ExpandDim(0) gives: <1,512,512,3>
  //     *- ExpandDim(1) gives: <512,1,512,3>
  //     *- ExpandDim(3) gives: <512,512,3,1>
  // @param axis location of the dim
  virtual Status ExpandDim(const dsize_t &axis);

  virtual void Squeeze();

  /// Calculates the strides of the Tensor
  /// Ex: Tensor of shape <4,2,2> and type DE_UINT8 (1 byte)
  /// The strides will be {6,2,1}.
  /// Ex: Tensor of shape <4,2,2> and type DE_UINT32 (4 byte)
  /// The strides will be {24,8,4}.
  /// @return vector of integers
  std::vector<dsize_t> Strides();

  std::string ToString() {
    std::stringstream ss;
    this->Print(ss);
    return ss.str();
  }

  // Constructs numpy array from input tensor
  // @param data this data is the location of python data
  // @return Status code
  Status GetDataAsNumpy(py::array *data);

  Status GetDataAsNumpyStrings(py::array *data);

  static Status GetBufferInfo(Tensor &t, py::buffer_info *out);

  // TensorIterator is a linear iterator that can be used to iterate over the elements of the Tensor
  // The order  elements  is as the memory layout (i.e., row-major) [[1,2,3],[4,5,6] --> 1,2,3,4,5,6
  // @tparam T type of values in the Tensor Iterator
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
  // @tparam DUMMY, used to mbe able to specialize the inner class
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
      index_ = index;
    }

    TensorIterator(const TensorIterator<std::string_view, DUMMY> &raw_iterator) {
      data_ = raw_iterator.data_;
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

  // Return a TensorIterator that points to the start of the Tensor.
  // It's the user responsibility to use the correct type that matches the Tensor type
  // @tparam T The type of values in the Tensor
  // @return TensorIterator
  template <typename T>
  TensorIterator<T> begin() {
    AllocateBuffer(SizeInBytes());
    return TensorIterator<T>(data_);
  }

  // Return a linear iterator that points to the place after the last element of the Tensor.
  // @tparam T The type of values in the Tensor
  // @return TensorIterator
  template <typename T>
  TensorIterator<T> end() {
    return TensorIterator<T>(data_end_);
  }

  // Copies the last dimension at `index` from Tensor `src` to this Tensor.
  // @param src Tensor
  // @param index vector to the start of the dimension. The last dim should be 0
  // @return Status
  Status CopyLastDimAt(const std::shared_ptr<Tensor> &src, const std::vector<dsize_t> &index);

 protected:
  // Get the starting memory address for the data of the tensor.  This potentially
  // drives an allocation if the data is null.
  // @return unsigned char*
  unsigned char *GetMutableBuffer();

  // A function that prints Tensor recursively, first called by print
  // @param out
  // @param cur_dim
  // @param cur_index
  void PrintRecursive(std::ostream &out, int32_t cur_dim, const std::vector<dsize_t> &cur_index) const;

  // A function that prints info about the tensor
  // @param out output stream
  void Print(std::ostream &out) const;

  // A function that print the value as specified by its index
  // @param index vector representing the index
  // @param out
  void PrintItemAt(const std::vector<dsize_t> &index, std::ostream &out) const;

  // Get pointer to item located at `index`, caller needs to provide the type.
  // @tparam T
  // @param index vector<dsize_t>
  // @return return a pointer to the item specified at index of type `T`
  template <typename T>
  Status GetItemPtr(T **, const std::vector<dsize_t> &index) const;

  // Get pointer to string located at `index` and the length of string
  // @param index vector<dsize_t>
  // @return return a pointer to the string specified at index and the length of the string
  Status GetItemPtr(uchar **, const std::vector<dsize_t> &index, offset_t *length = nullptr) const;

  // Given a flat index of an item string, return the start and length of the item
  // @param index flat index of the item
  // @return start address of the ths string
  // @return length of the string
  Status GetStringAt(dsize_t index, uchar **string_start, offset_t *length) const;

  // Skip the offsets and returns the start of the buffer where the real strings is stored. Caller needs to check if the
  // tensor's type is a string, otherwise undefined address would be returned.
  // @return address of the first string of the tensor.
  uchar *GetStringsBuffer() const { return data_ + kOffsetSize * shape_.NumOfElements() + kOffsetSize; }

  // all access to shape_ should be via shape
  TensorShape shape_;
  // data type of tensor
  DataType type_;
  // pointer to the start of the physical data
  unsigned char *data_;
  // An allocator for data_
  CharAllocPtr data_allocator_;
  // pointer to the end of the physical data
  unsigned char *data_end_ = nullptr;
};
template <>
inline Tensor::TensorIterator<std::string_view> Tensor::end<std::string_view>() {
  return TensorIterator<std::string_view>(data_, shape_.NumOfElements());
}
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_CORE_TENSOR_H_
