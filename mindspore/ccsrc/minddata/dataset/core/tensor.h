/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#undef HAVE_STDDEF_H
#undef HAVE_STDLIB_H
#endif

#include "./securec.h"
#ifndef ENABLE_ANDROID
#include "proto/example.pb.h"
#endif
#ifdef ENABLE_PYTHON
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#endif

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/core/tensor_helpers.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/status.h"
#include "utils/ms_utils.h"

#ifdef ENABLE_PYTHON
namespace py = pybind11;
#endif

namespace mindspore {
namespace dataset {
class Tensor;
template <typename T>
class Allocator;

using CharAllocPtr = std::unique_ptr<Allocator<unsigned char>>;
using TensorAllocPtr = std::shared_ptr<Allocator<Tensor>>;  // An allocator shared_ptr for Tensors
using offset_t = uint32_t;                                  // type of offset values to store strings locations
using TensorPtr = std::shared_ptr<Tensor>;

class DATASET_API Tensor {
 public:
  Tensor() = delete;
  Tensor(const Tensor &other) = delete;
  Tensor &operator=(const Tensor &other) = delete;

  /// Create a tensor using shape and type. This constructor should not be used directly, use CreateFromTensor instead.
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

  /// Create a copy of the input tensor
  /// \param[in] MSTensor to create DETensorFrom
  /// \return Status
  static Status CreateFromMSTensor(const MSTensor &in, TensorPtr *out);

#ifdef ENABLE_PYTHON
  /// Create a Tensor from a given py::array
  /// \param[in] arr py::array
  /// \param[out] out Created tensor
  /// \return Status Code
  static Status CreateFromNpArray(const py::array &arr, TensorPtr *out);

  /// Helper function to create a tensor from a Python dictionary object
  /// \param[in] obj pybind11 wrapper for Python dictionary object
  /// \param[out] out Created Tensor
  /// \return Status
  static Status CreateFromPythonObject(py::object obj, TensorPtr *out);
#endif

#ifndef ENABLE_ANDROID
  /// Create a tensor of type DE_STRING from a BytesList.
  /// \param[in] bytes_list protobuf's Bytelist
  /// \param[in] shape shape of the output tensor
  /// \param[out] out created Tensor
  /// \return Status Code
  static Status CreateFromByteList(const dataengine::BytesList &bytes_list, const TensorShape &shape, TensorPtr *out);

  /// Create a tensor of type UINT8 or INT8 from a BytesList.
  /// The tensor will be padded with ' ' to reach the required pad_size.
  /// \param[in] bytes_list protobuf's Bytelist
  /// \param[in] shape shape of the output tensor
  /// \param[in] type type of created tensor. Should be DE_UINT8 or INT8
  /// \param[in] pad_size The size of the tensor after padding
  /// \param[out] out created Tensor
  /// \return Status Code
  static Status CreateFromByteList(const dataengine::BytesList &bytes_list, const TensorShape &shape,
                                   const DataType &type, dsize_t pad_size, TensorPtr *out);
#endif

  /// Create a Tensor from a given list of values.
  /// \tparam type of the values to be inserted.
  /// \param[in] items elements of the tensor
  /// \param[in] shape shape of the output tensor
  /// \param[out] out output argument to hold the created Tensor
  /// \return Status Code
  template <typename T>
  static Status CreateFromVector(const std::vector<T> &items, const TensorShape &shape, TensorPtr *out) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      static_cast<dsize_t>(items.size()) == shape.NumOfElements(),
      "Number of elements in the vector does not match the number of elements of the shape required");
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

  /// Create a Tensor from a given list of strings.
  /// @note: The memory layout of a Tensor of strings consists of the Offset_array followed by the strings.
  /// The offset array will store one extra value to find the length of the last string.
  /// OFFSET_1, OFFSET_2, ..., OFFSET_n+1, STRING_1, STRING_2, ..., STRING_n
  /// The value of each offset is the start index of the corresponding string
  /// Offsets is of type offset_t
  /// strings will ne null-terminated
  /// example: Tensor(['abc', 'de'], shape={2}, type=DE_STRING)
  /// |----------------------------------------------------------------|
  /// |             OFFSET ARRAY           |            STRINGS        |
  /// | bytes 0-3 | bytes 3-6 | bytes 7-10 | bytes 11-14 | bytes 15-17 |
  /// |     11    |    15     |     18     |     abc\0   |      de\0   |
  /// |----------------------------------------------------------------|
  /// \param[in] items elements of the tensor
  /// \param[in] shape shape of the output tensor
  /// \param[in] type data type of the output tensor, can only be DE_STRING or DE_BYTES
  /// \param[out] out output argument to hold the created Tensor
  /// \return Status Code
  static Status CreateFromVector(const std::vector<std::string> &items, const TensorShape &shape, const DataType &type,
                                 TensorPtr *out) {
    RETURN_UNEXPECTED_IF_NULL(out);
    CHECK_FAIL_RETURN_UNEXPECTED(static_cast<dsize_t>(items.size()) == shape.NumOfElements(),
                                 "The number of elements in the vector: " + std::to_string(items.size()) +
                                   " does not match the number of elements: " + std::to_string(shape.NumOfElements()) +
                                   " the shape required.");
    CHECK_FAIL_RETURN_UNEXPECTED(type.IsString(), "Can not create a numeric Tensor from a string vector.");
    const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
    *out = std::allocate_shared<Tensor>(*alloc, TensorShape({static_cast<dsize_t>(items.size())}), type);
    CHECK_FAIL_RETURN_UNEXPECTED(out != nullptr, "Allocate memory failed.");
    if (items.empty()) {
      if (shape.known()) {
        return (*out)->Reshape(shape);
      }
    }
    auto length_sum = [](size_t sum, const std::string &s) { return s.length() + sum; };
    dsize_t total_length = std::accumulate(items.begin(), items.end(), 0, length_sum);

    // total bytes needed = offset array + strings
    // offset array needs to store one offset var per element + 1 extra to get the length of the last string.
    // strings will be null-terminated --> need 1 extra byte per element
    size_t num_bytes = (kOffsetSize + 1) * (*out)->shape_.NumOfElements() + kOffsetSize + total_length;

    RETURN_IF_NOT_OK((*out)->AllocateBuffer(num_bytes));
    auto offset_arr = reinterpret_cast<offset_t *>((*out)->data_);
    uchar *buf = (*out)->GetStringsBuffer();

    offset_t offset = buf - (*out)->data_;  // the first string will start here
    uint32_t i = 0;
    for (const auto &str : items) {
      //  insert the start index of the string.
      offset_arr[i++] = offset;
      // insert actual string
      int ret_code = memcpy_s((*out)->data_ + offset, num_bytes - offset, common::SafeCStr(str), str.length() + 1);
      if (ret_code != 0) {
        MS_LOG(ERROR) << "Cannot copy string into Tensor";
      }
      //  next string will be stored right after the current one.
      offset = offset + str.length() + 1;
    }
    // store one more offset value so we can get the length of the last string
    offset_arr[i] = offset;

    (*out)->data_end_ = (*out)->data_ + offset_arr[i];

    MS_ASSERT(num_bytes - offset == 0);
    if (shape.known()) {
      RETURN_IF_NOT_OK((*out)->Reshape(shape));
    }
    return Status::OK();
  }

  // Create a string Tensor from a string vector by default.
  static Status CreateFromVector(const std::vector<std::string> &items, const TensorShape &shape, TensorPtr *out) {
    return CreateFromVector(items, shape, DataType(DataType::DE_STRING), out);
  }

  /// Create a numeric scalar Tensor from the given value.
  /// \tparam T type of value
  /// \param[in] item value
  /// \param[out] out Created tensor
  /// \return Status code
  template <typename T>
  static Status CreateScalar(const T &item, TensorPtr *out) {
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

  Status to_json(nlohmann::json *out_json);

  template <typename T>
  Status to_json_convert(nlohmann::json *args);

  static Status from_json(nlohmann::json op_params, std::shared_ptr<Tensor> *tensor);

  template <typename T>
  static Status from_json_convert(const nlohmann::json &json_data, const TensorShape &shape,
                                  std::shared_ptr<Tensor> *tensor);

  static Status from_json_convert(const nlohmann::json &json_data, const TensorShape &shape, const DataType &type,
                                  std::shared_ptr<Tensor> *tensor);

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

  /// set string item at location specified by index
  /// \param[in] index
  /// \param[in] value of type std::string
  Status SetItemAt(const std::vector<dsize_t> &index, const std::string &value) {
    RETURN_UNEXPECTED_IF_NULL(data_);
    uchar *ptr = nullptr;
    offset_t length = 0;
    RETURN_IF_NOT_OK(GetItemPtr(&ptr, index, &length));
    if (value.length() != length) {
      RETURN_STATUS_UNEXPECTED("Length of the new string does not match the item.");
    }
    int ret_code = memcpy_s(reinterpret_cast<char *>(ptr), length, value.c_str(), length);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == 0, "Failed to set data into tensor.");

    return Status::OK();
  }

  /// Fill tensor with zeros. Does not support string or bytes.
  Status Zero() {
    CHECK_FAIL_RETURN_UNEXPECTED(!type_.IsString(), "Can not fill zeros on tensor of type string or bytes.");
    dsize_t size = SizeInBytes();
    CHECK_FAIL_RETURN_UNEXPECTED(memset_sp(GetMutableBuffer(), size, 0, size) == 0,
                                 "Failed to fill tensor with zeroes.");
    return Status::OK();
  }

  /// Fill all elements in the Tensor with the given value of type `T`. Does not support string or bytes.
  /// \tparam T
  /// \param value[in]
  template <typename T>
  Status Fill(const T &value) {
    CHECK_FAIL_RETURN_UNEXPECTED(!type_.IsString(), "Can not fill on tensor of type string or bytes.");
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
      return Status(StatusCode::kMDUnexpectedError, err);
    }
  }

  /// Getter function for shape
  /// \return
  const TensorShape &shape() const { return shape_; }

  /// Check if tensor has data
  /// \return bool - true if tensor is not empty
  bool HasData() const { return data_ != nullptr; }

  /// Check if tensor is complex
  /// \return bool - true if tensor is complex
  bool IsComplex() const {
    if (shape_.empty()) {
      return false;
    }
    // check the last dim all be 2
    return shape_[-1] == 2;
  }

  /// Reshape the tensor. The given shape should have the same number of elements in the Tensor
  /// \param shape
  virtual Status Reshape(const TensorShape &shape);

  /// \return number of elements in this tensor
  dsize_t Size() const { return shape().NumOfElements(); }

  /// \return the number of bytes this tensor is needs
  dsize_t SizeInBytes() const {
    if (data_end_ == nullptr) {
      return type_.SizeInBytes() * shape_.NumOfElements();
    }
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
                      bool partial_insert = false);

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
    if (length.size() < index_vector.size()) {
      MS_LOG(ERROR) << "The size of length should be greater than the shape of index_vector";
      return {};
    }
    std::vector<dsize_t> indices(index_vector.size(), 0);
    for (size_t i = 0; i < index_vector.size(); i++) {
      indices[i] = HandleNeg(index_vector[i], length[i]);
    }
    return indices;
  }

  /// Slice tensor bases on the given indices. Copy the sliced data into out tensor.
  /// Based on the type of tensor, SliceNumeric or SliceString will be called
  /// \param[out] out Tensor
  /// \param[in] slice_options vector of SliceOption objects
  /// \return Status error code
  Status Slice(TensorPtr *out, const std::vector<mindspore::dataset::SliceOption> &slice_options);

  /// Get slice_option according to shape and index.
  /// \param[in] slice_option input SliceOption object
  /// \param[in] slice_index index of SliceOption object
  /// \param[out] output slice_option with shape info
  /// \return Status error code
  Status GetSliceOption(const SliceOption &slice_option, const int32_t &slice_index, SliceOption *slice_option_ptr);

#ifdef ENABLE_PYTHON
  /// Constructs numpy array from input tensor
  /// \param[out] data this data is the location of python data
  /// \return Status code
  Status GetDataAsNumpy(py::array *data);

  /// Constructs numpy array of string or bytes
  /// \param[out] data this data is the location of python data
  /// \return Status code
  Status GetDataAsNumpyStrings(py::array *data);

  template <typename T>
  Status GetDataAsNumpyStrings(py::array *data) {
    RETURN_UNEXPECTED_IF_NULL(data);
    if (Size() == 0) {
      // NumPy will create empty array in type of float64 by default. So we must define the data type.
      *data = py::array(type_.AsNumpyType(), shape_.AsVector(), nullptr);
    } else {
      std::vector<T> string_vector;
      string_vector.reserve(Size());
      // Iterate over tensor and create a vector of string_views of strings in the tensor.
      (void)std::transform(begin<std::string_view>(), end<std::string_view>(), std::back_inserter(string_vector),
                           [](const auto &element) { return static_cast<std::string>(element); });
      *data = py::array(py::cast(string_vector));
      data->resize(shape_.AsVector());
    }
    return Status::OK();
  }

  static Status GetBufferInfo(Tensor *t, py::buffer_info *out);

  /// Returns the Python dictionary stored in the tensor
  /// \param[out] data this data is the location of Python data (pybind11 wrapper)
  /// \return Status code
  Status GetDataAsPythonObject(py::dict *data);

#endif

  Status SetYuvShape(const uint32_t &width, const uint32_t &widthStride, const uint32_t &height,
                     const uint32_t &heightStride) {
    std::vector<uint32_t> tmp{width, widthStride, height, heightStride};
    yuv_shape_ = tmp;
    return Status::OK();
  }

  std::vector<uint32_t> GetYuvShape() { return yuv_shape_; }

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

    explicit TensorIterator(uchar *ptr = nullptr) : ptr_(reinterpret_cast<T *>(ptr)) {}

    TensorIterator(const TensorIterator<T> &raw_iterator) { ptr_ = raw_iterator.ptr_; }

    ~TensorIterator() = default;

    TensorIterator<T> &operator=(const TensorIterator<T> &rhs) {
      if (this == &rhs) {
        return *this;
      }
      ptr_ = rhs.ptr_;
      return *this;
    }

    TensorIterator<T> &operator=(T *rhs) {
      ptr_ = rhs;
      return *this;
    }

    bool operator==(const TensorIterator<T> &rhs) const { return ptr_ == rhs.ptr_; }

    bool operator!=(const TensorIterator<T> &rhs) const { return !(*this == rhs); }

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

    explicit TensorIterator(const uchar *data = nullptr, dsize_t index = 0) {
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
      offset_t end = offset_[index_ + 1];
      return std::string_view{data_ + start, end - start - 1};  // -1 to skip the \0 at the end
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

  /// A function that prints info about the tensor
  /// \param[out] out output stream
  void PrintData(std::ostream &out) const;

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
  /// the tensor's type is a string, otherwise undefined address would be returned.
  /// \return return the address of the first string of the tensor.
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

  /// shape for interpretation of YUV image
  std::vector<uint32_t> yuv_shape_;

#ifdef ENABLE_PYTHON
  /// Store python dictionary wrapper
  py::object python_dict_;
#endif

 private:
  friend class DETensor;

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

#ifdef ENABLE_PYTHON
  /// Helper function to create a tensor from Numpy array of strings
  /// \param[in] arr Numpy array
  /// \param[out] out Created Tensor
  /// \return Status
  static Status CreateFromNpString(py::array arr, TensorPtr *out);
#endif
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
  RETURN_UNEXPECTED_IF_NULL(out);
  return CreateFromVector({item}, TensorShape::CreateScalar(), DataType(DataType::DE_STRING), out);
}
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_TENSOR_H_
