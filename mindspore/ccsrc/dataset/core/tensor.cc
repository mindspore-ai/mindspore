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
#include "dataset/core/tensor.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include <functional>

#include "common/utils.h"
#include "dataset/core/constants.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/global_context.h"
#include "dataset/core/pybind_support.h"
#include "dataset/core/tensor_shape.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {
// Helper macros for printing tensor elements
#define CASE_PRINT(de_type, native_type)    \
  case de_type: {                           \
    native_type o;                          \
    rc = GetItemAt<native_type>(&o, index); \
    out << o;                               \
    break;                                  \
  }

#define CASE_PRINT_HEX(de_type, native_type)                                                    \
  case de_type: {                                                                               \
    native_type o;                                                                              \
    rc = GetItemAt<native_type>(&o, index);                                                     \
    out << std::hex << std::setw(2) << std::setfill('0') << o << std::dec << std::setfill(' '); \
    break;                                                                                      \
  }

Tensor::Tensor(const TensorShape &shape, const DataType &type) : shape_(shape), type_(type), data_(nullptr) {
  // grab the mem pool from global context and create the allocator for char data area
  std::shared_ptr<MemoryPool> global_pool = GlobalContext::Instance()->mem_pool();
  data_allocator_ = std::make_unique<Allocator<unsigned char>>(global_pool);
}

Tensor::Tensor(const TensorShape &shape, const DataType &type, const unsigned char *data) : Tensor(shape, type) {
  // If the data pointer was given, then we can also populate the tensor with data
  if (data != nullptr) {
    // Given the shape/type of this tensor, compute the data size and copy in the input bytes.
    int64_t byte_size = this->SizeInBytes();
    static_cast<void>(this->StartAddr());  // Allocates data_ inside itself
    if (data_ != nullptr) {
      int ret_code = memcpy_s(data_, byte_size, data, byte_size);
      if (ret_code != 0) {
        MS_LOG(ERROR) << "Failed to copy data into Tensor!";
      }
    } else {
      MS_LOG(ERROR) << "Failed to create memory for Tensor!";
    }
  }
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape_(other.shape()),
      type_(other.type()),
      data_(other.StartAddr()),
      data_allocator_(std::move(other.data_allocator_)) {
  other.Invalidate();
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (&other != this) {
    shape_ = other.shape();
    type_ = other.type();
    data_ = other.StartAddr();
    data_end_ = other.data_end_;
    data_allocator_ = std::move(other.data_allocator_);
    other.Invalidate();
  }
  return *this;
}

Status Tensor::CreateTensor(std::shared_ptr<Tensor> *ptr, TensorImpl tensor_impl, const TensorShape &shape,
                            DataType type, const unsigned char *data) {
  if (!shape.known()) {
    RETURN_STATUS_UNEXPECTED("Invalid shape.");
  }
  if (type == DataType::DE_UNKNOWN) {
    RETURN_STATUS_UNEXPECTED("Invalid data type.");
  }

  switch (tensor_impl) {
    case TensorImpl::kFlexible: {
      // The flex tensor is really just the base class tensor implementation
      const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
      *ptr = std::allocate_shared<Tensor>(*alloc, shape, type, data);
      break;
    }
    case TensorImpl::kCv: {
      const CVTensorAlloc *alloc = GlobalContext::Instance()->cv_tensor_allocator();
      *ptr = std::allocate_shared<CVTensor>(*alloc, shape, type, data);
      break;
    }
    default: {
      std::string err_msg("Invalid tensor implementation type.");
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  return Status::OK();  // returns base-class shared_ptr
}

Status Tensor::CreateTensor(std::shared_ptr<Tensor> *ptr, py::array arr) {
  const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
  *ptr = std::allocate_shared<Tensor>(*alloc, TensorShape({}), DataType(DataType::DE_UNKNOWN));

  std::vector<dsize_t> shape;
  for (dsize_t i = 0; i < arr.ndim(); i++) {
    shape.push_back(static_cast<dsize_t>(arr.shape()[i]));
  }

  (*ptr)->shape_ = TensorShape(shape);
  (*ptr)->type_ = DataType::FromNpArray(arr);
  if (!(*ptr)->shape_.known()) RETURN_STATUS_UNEXPECTED("Invalid shape.");

  if ((*ptr)->type_ == DataType::DE_UNKNOWN) RETURN_STATUS_UNEXPECTED("Invalid data type.");

  std::shared_ptr<MemoryPool> global_pool = GlobalContext::Instance()->mem_pool();
  (*ptr)->data_allocator_ = std::make_unique<Allocator<unsigned char>>(global_pool);
  static_cast<void>((*ptr)->StartAddr());
  int64_t byte_size = (*ptr)->SizeInBytes();
  unsigned char *data = static_cast<unsigned char *>(arr.request().ptr);
  if ((*ptr)->data_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Failed to create memory for Tensor.");
  }

  std::vector<dsize_t> strides;
  for (dsize_t i = 0; i < arr.ndim(); i++) {
    strides.push_back(static_cast<dsize_t>(arr.strides()[i]));
  }

  // check if strides are contiguous
  bool is_strided = false;
  dsize_t count = (*ptr)->shape_.NumOfElements();
  for (size_t i = 0; i < shape.size(); i++) {
    count /= shape[i];
    if (strides[i] != (*ptr)->type_.SizeInBytes() * count) {
      is_strided = true;
      break;
    }
  }

  if (is_strided) {
    RETURN_IF_NOT_OK(CopyStridedArray((*ptr)->data_, data, shape, strides, (*ptr)->type_.SizeInBytes()));
  } else {
    int ret_code = memcpy_s((*ptr)->data_, byte_size, data, byte_size);
    if (ret_code != 0) {
      RETURN_STATUS_UNEXPECTED("Failed to copy data into Tensor.");
    }
  }

  return Status::OK();  // returns base-class shared_ptr
}

// Memcpy the given strided array's used part to consecutive memory
// Consider a 3-d array
// A[(i * shape[1] + j) * shape[2] + k] = B[i][j][k] = C[i * strides[0] + j * strides[1] + k * strides[2]]
// Here we convert array C to array A, by memcpy index by index (Note that not all elements in C is copied)
Status Tensor::CopyStridedArray(unsigned char *dst, unsigned char *src, std::vector<dsize_t> shape,
                                std::vector<dsize_t> strides, uint8_t type_size) {
  dsize_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<dsize_t>());
  for (dsize_t i = 0; i < size; ++i) {
    dsize_t offset = 0;
    dsize_t count = i;
    for (size_t j = 0; j < shape.size(); ++j) {
      // convert 1d array's index to 3d array's index (A -> B)
      dsize_t idx = count % shape[shape.size() - 1 - j];
      count /= shape[shape.size() - 1 - j];
      // calculate the raw data offset based on strides (B -> C)
      offset += idx * strides[shape.size() - 1 - j];
      // once count = 0, the following idxes are all zero, skip them
      if (count == 0) break;
    }
    // strides already consider byte size of the data type, but dst doesn't.
    // dst[i] = dst + i * type_size = src + offset
    int ret_code = memcpy_s(dst + i * type_size, type_size, src + offset, type_size);
    if (ret_code != 0) {
      RETURN_STATUS_UNEXPECTED("Failed to copy data into Tensor.");
    }
  }
  return Status::OK();
}

// Name: Destructor
// Description: Destructor
Tensor::~Tensor() {
  if (data_ != nullptr) {
    if (data_allocator_ != nullptr) {
      data_allocator_->deallocate(data_);
      data_ = nullptr;
      data_end_ = nullptr;
    } else {
      // If we didn't have an allocator, but data_ is not null then it must
      // be a stand-alone tensor that used malloc directly.
      free(data_);
      data_ = nullptr;
      data_end_ = nullptr;
    }
  }
}

bool Tensor::operator==(const Tensor &rhs) const {
  // 1. different shape 2. different type 3. one data_ is nullptr and the other is not
  if (shape_ != rhs.shape() || type_ != rhs.type_ || (data_ == nullptr && rhs.data_ != nullptr) ||
      (data_ != nullptr && rhs.data_ == nullptr)) {
    return false;
  }
  if (data_ == nullptr && rhs.data_ == nullptr) {
    return true;
  }
  // use mem compare to compare the two data, size are already verified
  return memcmp(data_, rhs.data_, SizeInBytes()) == 0;
}

// Name: PrintItemAt()
// Description: A function that print the value as specified by its index
void Tensor::PrintItemAt(const std::vector<dsize_t> &index, std::ostream &out) const {
  Status rc;
  DS_ASSERT(data_);

  switch (type_.value()) {
    CASE_PRINT_HEX(DataType::DE_BOOL, uint8_t);

    CASE_PRINT_HEX(DataType::DE_INT8, int8_t);

    CASE_PRINT_HEX(DataType::DE_UINT8, uint8_t);

    CASE_PRINT(DataType::DE_INT16, int16_t);

    CASE_PRINT(DataType::DE_UINT16, uint16_t);

    CASE_PRINT(DataType::DE_INT32, int32_t);

    CASE_PRINT(DataType::DE_UINT32, uint32_t);

    CASE_PRINT(DataType::DE_INT64, int64_t);

    CASE_PRINT(DataType::DE_UINT64, uint64_t);

    CASE_PRINT(DataType::DE_FLOAT16, float16);

    CASE_PRINT(DataType::DE_FLOAT32, float);

    CASE_PRINT(DataType::DE_FLOAT64, double);

    default: {
      out << "?";
      break;
    }
  }
  if (rc.IsError()) {
    out << rc.ToString();
  }
}

// Name: PrintRecursive()
// Description: A function that prints Tensor recursively, first called by print
void Tensor::PrintRecursive(std::ostream &out, int32_t cur_dim, const std::vector<dsize_t> &cur_index) const {
  if (cur_index.size() == shape_.Rank()) {
    PrintItemAt(cur_index, out);
  } else {
    out << "[";
    for (dsize_t i = 0; i < shape_[cur_dim]; i++) {
      std::vector<dsize_t> new_index = cur_index;
      new_index.push_back(i);
      PrintRecursive(out, cur_dim + 1, new_index);
      if (i < shape_[cur_dim] - 1) {
        out << ",";
      }
    }
    out << "]";
  }
}

// Name: Print()
// Description: A function that prints info about the tensor
void Tensor::Print(std::ostream &out) const {
  out << "Tensor (shape: ";
  out << shape_;
  out << ", Type: " << type_ << ")\n";
  if (data_) {
    PrintRecursive(out, 0, std::vector<dsize_t>{});
  } else {
    out << "[Data area is null]";
  }
}

// Name: ToFlatIndex()
// Description: convert a vector style index to number, used to access memory internal use only
Status Tensor::ToFlatIndex(const std::vector<dsize_t> &index, dsize_t *flat_index) const {
  if (!shape_.IsValidIndex(index)) {
    std::string err = "Not a valid index";
    RETURN_STATUS_UNEXPECTED(err);
  }
  *flat_index = 0;
  for (size_t k = 0; k < index.size(); k++) {
    dsize_t product = 1;
    for (size_t l = k + 1; l < index.size(); l++) {
      product *= shape_[l];
    }
    *flat_index += index[k] * product;
  }
  return Status::OK();
}

const unsigned char *Tensor::StartAddr() const {
  // This version cannot modify anything.  data_ could possibly be null.
  return data_;
}

unsigned char *Tensor::StartAddr() {
  if (!shape_.known() || type_ == DataType::DE_UNKNOWN) {
    return nullptr;
  }
  // If the data area is already created, return the pointer to it
  if (data_ != nullptr) {
    return data_;
  } else {
    // If the data area is not created, then identify the memory size based
    // on the shape and type and allocate it.
    if (data_allocator_ != nullptr) {
      data_ = data_allocator_->allocate(this->SizeInBytes());
      data_end_ = data_ + SizeInBytes();
    } else {
      data_ = static_cast<unsigned char *>(malloc(this->SizeInBytes()));
      data_end_ = data_ + SizeInBytes();
      if (data_ == nullptr) {
        return nullptr;
      }
    }
    return data_;
  }
}

Status Tensor::Reshape(const TensorShape &shape) {
  if (shape.NumOfElements() == shape_.NumOfElements()) {
    shape_ = shape;
    return Status::OK();
  } else {
    std::string err = "Cannot reshape, Number of elements do not match";
    RETURN_STATUS_UNEXPECTED(err);
  }
}

void Tensor::Invalidate() {
  shape_ = TensorShape::CreateUnknownRankShape();
  type_ = DataType(DataType::DE_UNKNOWN);
  data_ = nullptr;
  data_end_ = nullptr;
  data_allocator_ = nullptr;
}

template <typename T>
Status Tensor::GetItemPtr(T **ptr, const std::vector<dsize_t> &index) const {
  if (type_.IsCompatible<T>()) {
    if (data_ == nullptr) {
      std::string err = "Data is not allocated yet";
      RETURN_STATUS_UNEXPECTED(err);
    }
    dsize_t flat_idx;
    RETURN_IF_NOT_OK(ToFlatIndex(index, &flat_idx));
    *ptr = reinterpret_cast<T *>(data_ + flat_idx * type_.SizeInBytes());
    return Status::OK();
  } else {
    std::string err = "data type not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
}

Status Tensor::StartAddrOfIndex(std::vector<dsize_t> ind, uchar **start_addr_of_index, TensorShape *remaining) {
  dsize_t flat_ind;
  std::vector<dsize_t> t_shape = shape().AsVector();
  std::vector<dsize_t> r(t_shape.begin() + ind.size(), t_shape.end());
  *remaining = TensorShape(r);
  ind.resize(this->Rank(), 0);  //  same as -> while (ind.size() < this->Rank()) ind.push_back(0);
  RETURN_IF_NOT_OK(ToFlatIndex(ind, &flat_ind));
  // check if StartAddr() returns null, we should flag this as an error, this sanity check will only
  // be true is the tensor failed to allocate memory.
  if (StartAddr() == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid StartAddr in Tensor, got nullptr");
  }
  *start_addr_of_index = StartAddr() + flat_ind * this->type().SizeInBytes();
  return Status::OK();
}

Status Tensor::InsertTensor(const std::vector<dsize_t> &ind, const std::shared_ptr<Tensor> &tensor) {
  std::string err_msg;
  err_msg += (!this->shape().known() || !tensor->shape().known()) ? "[Tensor] unknown shape\n" : "";
  err_msg += (ind.size() + tensor->Rank() != this->Rank()) ? "[Tensor] incorrect index\n" : "";
  err_msg += tensor->type().SizeInBytes() != this->type().SizeInBytes() ? "[Tensor] incorrect datatype\n" : "";
  uchar *start_addr_of_ind = nullptr;
  TensorShape remaining_shape({-1});
  err_msg += (!StartAddrOfIndex(ind, &start_addr_of_ind, &remaining_shape).IsOk()) ? "[Tensor] incorrect index\n" : "";
  err_msg += !(remaining_shape == tensor->shape()) ? "[Tensor] memory error\n" : "";
  if (!err_msg.empty()) {
    MS_LOG(INFO) << "Insert tensor message: " << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    if (start_addr_of_ind != nullptr) {
      int ret_code = memcpy_s(start_addr_of_ind, tensor->SizeInBytes(), tensor->StartAddr(), tensor->SizeInBytes());
      if (ret_code == 0) {
        return Status::OK();
      } else {
        err_msg += "[Tensor] error in memcpy_s when inserting tensor\n";
        MS_LOG(INFO) << "Tensor message: " << err_msg;
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
    } else {
      RETURN_STATUS_UNEXPECTED("Failed to create memory for Tensor.");
    }
  }
}

Status Tensor::ExpandDim(const dsize_t &axis) {
  if (axis > Rank()) {
    std::string err = "Axis is out of bound";
    RETURN_STATUS_UNEXPECTED(err);
  }
  if (axis == Rank()) {
    shape_ = shape_.AppendDim(1);
  } else {
    shape_ = shape_.InsertDim(axis, 1);
  }
  return Status::OK();
}

std::vector<dsize_t> Tensor::Strides() {
  std::vector<dsize_t> strides(Rank());
  dsize_t count = shape_.NumOfElements();
  for (dsize_t i = 0; i < Rank(); i++) {
    count /= shape_[i];
    strides[i] = type_.SizeInBytes() * count;
  }
  return strides;
}

Status Tensor::GetBufferInfo(Tensor &t, py::buffer_info *out) {
  std::string format_desc = t.type().GetPybindFormat();
  if (format_desc.empty()) {
    RETURN_STATUS_UNEXPECTED("Cannot convert DE type tp pybind format");
  }
  *out = py::buffer_info(t.StartAddr(),          /* Pointer to buffer */
                         t.type().SizeInBytes(), /* Size of one scalar */
                         format_desc,            /* Python struct-style format descriptor */
                         t.Rank(),               /* Number of dimensions */
                         t.shape().AsVector(),   /* Buffer dimensions */
                         t.Strides());
  return Status::OK();
}

template <typename T>
Status Tensor::GetItemAt(T *o, const std::vector<dsize_t> &index) const {
  if (data_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Data is not allocated yet");
  }
  if (!type_.IsLooselyCompatible<T>()) {
    std::string err = "Template type and Tensor type are not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
  if (type_.IsUnsignedInt()) {
    RETURN_IF_NOT_OK(GetUnsignedIntAt<T>(o, index));
  } else if (type_.IsSignedInt()) {
    RETURN_IF_NOT_OK(GetSignedIntAt<T>(o, index));
  } else if (type_.IsFloat()) {
    RETURN_IF_NOT_OK(GetFloatAt<T>(o, index));
  } else if (type_.IsBool()) {
    bool *ptr = nullptr;
    RETURN_IF_NOT_OK(GetItemPtr<bool>(&ptr, index));
    *o = static_cast<T>(*ptr);
  } else {
    std::string err = "Tensor Type is unknown";
    RETURN_STATUS_UNEXPECTED(err);
  }
  return Status::OK();
}

// return data as numpy, should return status
Status Tensor::GetDataAsNumpy(py::array *data) {
  RETURN_UNEXPECTED_IF_NULL(data_);
  RETURN_UNEXPECTED_IF_NULL(data);
  if (type_ == DataType::DE_BOOL) {
    *data = py::array_t<bool>(shape_.AsVector(), reinterpret_cast<bool *>(data_));
  } else if (type_ == DataType::DE_INT8) {
    *data = py::array_t<int8_t>(shape_.AsVector(), reinterpret_cast<int8_t *>(data_));
  } else if (type_ == DataType::DE_INT16) {
    *data = py::array_t<int16_t>(shape_.AsVector(), reinterpret_cast<int16_t *>(data_));
  } else if (type_ == DataType::DE_INT32) {
    *data = py::array_t<int32_t>(shape_.AsVector(), reinterpret_cast<int32_t *>(data_));
  } else if (type_ == DataType::DE_INT64) {
    *data = py::array_t<int64_t>(shape_.AsVector(), reinterpret_cast<int64_t *>(data_));
  } else if (type_ == DataType::DE_UINT8) {
    *data = py::array_t<uint8_t>(shape_.AsVector(), reinterpret_cast<uint8_t *>(data_));
  } else if (type_ == DataType::DE_UINT16) {
    *data = py::array_t<uint16_t>(shape_.AsVector(), reinterpret_cast<uint16_t *>(data_));
  } else if (type_ == DataType::DE_UINT32) {
    *data = py::array_t<uint32_t>(shape_.AsVector(), reinterpret_cast<uint32_t *>(data_));
  } else if (type_ == DataType::DE_UINT64) {
    *data = py::array_t<uint64_t>(shape_.AsVector(), reinterpret_cast<uint64_t *>(data_));
  } else if (type_ == DataType::DE_FLOAT16) {
    *data = py::array_t<float16>(shape_.AsVector(), reinterpret_cast<float16 *>(data_));
  } else if (type_ == DataType::DE_FLOAT32) {
    *data = py::array_t<float>(shape_.AsVector(), reinterpret_cast<float *>(data_));
  } else if (type_ == DataType::DE_FLOAT64) {
    *data = py::array_t<double>(shape_.AsVector(), reinterpret_cast<double *>(data_));
  } else {
    RETURN_STATUS_UNEXPECTED("Got unexpected type when returning numpy");
  }
  return Status::OK();
}

void Tensor::Squeeze() { shape_ = shape_.Squeeze(); }

template <typename T>
Status Tensor::GetUnsignedIntAt(T *o, const std::vector<dsize_t> &index) const {
  if (data_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Data is not allocated yet");
  }
  if (!type_.IsLooselyCompatible<T>()) {
    std::string err = "Template type and Tensor type are not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
  switch (type_.value()) {
    case DataType::DE_UINT8: {
      uint8_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<uint8_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_UINT16: {
      uint16_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<uint16_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_UINT32: {
      uint32_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<uint32_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_UINT64: {
      uint64_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<uint64_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    default:
      std::string err = "Tensor Type is not an unsigned Integer";
      RETURN_STATUS_UNEXPECTED(err);
  }
  return Status::OK();
}

template <typename T>
Status Tensor::GetSignedIntAt(T *o, const std::vector<dsize_t> &index) const {
  if (data_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Data is not allocated yet");
  }
  if (!type_.IsLooselyCompatible<T>()) {
    std::string err = "Template type and Tensor type are not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
  switch (type_.value()) {
    case DataType::DE_INT8: {
      int8_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<int8_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_INT16: {
      int16_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<int16_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_INT32: {
      int32_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<int32_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_INT64: {
      int64_t *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<int64_t>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    default:
      std::string err = "Tensor Type is not a signed Integer";
      RETURN_STATUS_UNEXPECTED(err);
  }
  return Status::OK();
}

template <typename T>
Status Tensor::GetFloatAt(T *o, const std::vector<dsize_t> &index) const {
  if (data_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Data is not allocated yet");
  }
  if (!type_.IsLooselyCompatible<T>()) {
    std::string err = "Template type and Tensor type are not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
  switch (type_.value()) {
    case DataType::DE_FLOAT16: {
      float16 *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<float16>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_FLOAT32: {
      float *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<float>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    case DataType::DE_FLOAT64: {
      double *ptr = nullptr;
      RETURN_IF_NOT_OK(GetItemPtr<double>(&ptr, index));
      *o = static_cast<T>(*ptr);
      break;
    }
    default:
      std::string err = "Tensor Type is not a float/double";
      RETURN_STATUS_UNEXPECTED(err);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
