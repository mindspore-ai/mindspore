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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
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
  if (type.IsNumeric()) {
    // If the data pointer was given, then we can also populate the tensor with data
    if (data != nullptr) {
      // Given the shape/type of this tensor, compute the data size and copy in the input bytes.
      int64_t byte_size = this->SizeInBytes();
      Status s = this->AllocateBuffer(byte_size);  // Allocates data_ inside itself
      if (s.IsOk() && data_ != nullptr) {
        int ret_code = memcpy_s(data_, byte_size, data, byte_size);
        if (ret_code != 0) {
          MS_LOG(ERROR) << "Failed to copy data into Tensor!";
        }
      } else {
        MS_LOG(ERROR) << "Failed to create memory for Tensor!";
      }
    }
  } else {
    MS_LOG(ERROR) << "Type should be numeric to use this constructor.";
  }
}

Tensor::Tensor(const TensorShape &shape, const DataType &type, const unsigned char *data, const dsize_t &length)
    : Tensor(shape, type) {
  // If the data pointer was given, then we can also populate the tensor with data
  if (data != nullptr) {
    // Allocates data_ inside itself
    Status s = AllocateBuffer(length);
    if (s.IsError()) {
      MS_LOG(ERROR) << "Failed to create memory for Tensor!";
    }
    if (data_ != nullptr) {
      int ret_code = memcpy_s(data_, length, data, length);
      if (ret_code != 0) {
        MS_LOG(ERROR) << "Failed to copy data into Tensor!";
      }
    }
  }
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape_(other.shape()),
      type_(other.type()),
      data_(other.GetMutableBuffer()),
      data_allocator_(std::move(other.data_allocator_)) {
  other.Invalidate();
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (&other != this) {
    shape_ = other.shape();
    type_ = other.type();
    data_ = other.GetMutableBuffer();
    data_end_ = other.data_end_;
    data_allocator_ = std::move(other.data_allocator_);
    other.Invalidate();
  }
  return *this;
}

Tensor::Tensor(const std::vector<std::string> &strings, const TensorShape &shape)
    : Tensor(TensorShape({static_cast<dsize_t>(strings.size())}), DataType(DataType::DE_STRING)) {
  auto length_sum = [](dsize_t sum, const std::string &s) { return s.length() + sum; };
  dsize_t total_length = std::accumulate(strings.begin(), strings.end(), 0, length_sum);

  // total bytes needed = offset array + strings
  // offset array needs to store one offset var per element + 1 extra to get the length of the last string.
  // strings will be null-terminated --> need 1 extra byte per element
  dsize_t num_bytes = (kOffsetSize + 1) * shape_.NumOfElements() + kOffsetSize + total_length;

  data_ = data_allocator_->allocate(num_bytes);

  auto offset_arr = reinterpret_cast<offset_t *>(data_);
  uchar *buf = GetStringsBuffer();

  offset_t offset = buf - data_;  // the first string will start here
  uint32_t i = 0;
  for (const auto &str : strings) {
    //  insert the start index of the string.
    offset_arr[i++] = offset;
    // total bytes are reduced by kOffsetSize
    num_bytes -= kOffsetSize;
    // insert actual string
    int ret_code = memcpy_s(data_ + offset, num_bytes, common::SafeCStr(str), str.length() + 1);
    if (ret_code != 0) MS_LOG(ERROR) << "Cannot copy string into Tensor";
    //  next string will be stored right after the current one.
    offset = offset + str.length() + 1;
    // total bytes are reduced by the length of the string
    num_bytes -= str.length() + 1;
  }
  // store one more offset value so we can get the length of the last string
  // length[last_element] = offset_arr[last_element + 1] - offset_arr[last_element]
  offset_arr[i] = offset;

  this->data_end_ = data_ + offset_arr[i];

  DS_ASSERT(num_bytes == 0);
  if (shape.known()) Tensor::Reshape(shape);
}
Tensor::Tensor(const dataengine::BytesList &bytes_list, const TensorShape &shape)
    : Tensor(TensorShape({static_cast<dsize_t>(bytes_list.value_size())}), DataType(DataType::DE_STRING)) {
  // total bytes needed = offset array + strings
  // offset array needs to store one offset var per element + 1 extra to get the length of the last string.
  // strings will be null-terminated --> need 1 extra byte per element
  dsize_t num_bytes = (kOffsetSize)*shape_.NumOfElements() + kOffsetSize + bytes_list.ByteSizeLong();

  data_ = data_allocator_->allocate(num_bytes);

  auto offset_arr = reinterpret_cast<offset_t *>(data_);
  uchar *buf = GetStringsBuffer();

  offset_t offset = buf - data_;  // the first string will start here
  uint32_t i = 0;
  for (; i < bytes_list.value_size(); i++) {
    const std::string &str = bytes_list.value(i);
    //  insert the start index of the string.
    offset_arr[i] = offset;
    // total bytes are reduced by kOffsetSize
    num_bytes -= kOffsetSize;
    // insert actual string
    int ret_code = memcpy_s(data_ + offset, num_bytes, common::SafeCStr(str), str.length() + 1);
    if (ret_code != 0) {
      MS_LOG(ERROR) << "Cannot copy string into Tensor";
    }
    //  next string will be stored right after the current one.
    offset = offset + str.length() + 1;
    // total bytes are reduced by the length of the string
    num_bytes -= str.length() + 1;
  }
  // store one more offset value so we can get the length of the last string
  // length[last_element] = offset_arr[last_element + 1] - offset_arr[last_element]
  offset_arr[i] = offset;

  data_end_ = data_ + offset_arr[i];

  DS_ASSERT(num_bytes == 0);
  if (shape.known()) Tensor::Reshape(shape);
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

Status Tensor::CreateTensorFromNumpyString(std::shared_ptr<Tensor> *ptr, py::array arr) {
  std::vector<dsize_t> shape;
  for (dsize_t i = 0; i < arr.ndim(); i++) {
    shape.push_back(static_cast<dsize_t>(arr.shape()[i]));
  }
  arr.resize({arr.size()});  // flatten the py::array so we can iterate once
  std::vector<std::string> strings;

  if (arr.dtype().kind() == 'U') {
    std::for_each(arr.begin(), arr.end(), [&strings](const auto &s) { strings.emplace_back(py::cast<py::str>(s)); });
  } else {
    std::for_each(arr.begin(), arr.end(), [&strings](const auto &s) { strings.emplace_back(py::cast<py::bytes>(s)); });
  }

  arr.resize(shape);  // resize arr back to the original shape

  return CreateTensor(ptr, strings, TensorShape{shape});
}

Status Tensor::CreateTensor(std::shared_ptr<Tensor> *ptr, py::array arr) {
  if (DataType::FromNpArray(arr) == DataType::DE_STRING) {
    return CreateTensorFromNumpyString(ptr, arr);
  }
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
  int64_t byte_size = (*ptr)->SizeInBytes();
  RETURN_IF_NOT_OK((*ptr)->AllocateBuffer(byte_size));

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

Status Tensor::CreateTensor(std::shared_ptr<Tensor> *ptr, const std::vector<std::string> &strings,
                            const TensorShape &shape) {
  const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
  *ptr = std::allocate_shared<Tensor>(*alloc, strings, shape);
  return Status::OK();
}

Status Tensor::CreateTensor(std::shared_ptr<Tensor> *ptr, const dataengine::BytesList &bytes_list,
                            const TensorShape &shape) {
  const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
  *ptr = std::allocate_shared<Tensor>(*alloc, bytes_list, shape);
  return Status::OK();
}

Status Tensor::CreateTensor(std::shared_ptr<Tensor> *ptr, const std::string &file_path) {
  std::ifstream fs;
  fs.open(file_path, std::ios::binary | std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(!fs.fail(), "Fail to open file: " + file_path);
  int64_t num_bytes = fs.seekg(0, std::ios::end).tellg();
  CHECK_FAIL_RETURN_UNEXPECTED(fs.seekg(0, std::ios::beg).good(), "Fail to find size of file");
  RETURN_IF_NOT_OK(
    Tensor::CreateTensor(ptr, TensorImpl::kFlexible, TensorShape{num_bytes}, DataType(DataType::DE_UINT8)));
  int64_t written_bytes = fs.read(reinterpret_cast<char *>((*ptr)->GetMutableBuffer()), num_bytes).gcount();
  CHECK_FAIL_RETURN_UNEXPECTED(written_bytes == num_bytes && fs.good(), "Error in writing to tensor");
  fs.close();
  return Status::OK();
}

Status Tensor::CreateTensor(std::shared_ptr<Tensor> *ptr, const dataengine::BytesList &bytes_list,
                            const TensorShape &shape, const DataType &type, dsize_t pad_size) {
  RETURN_IF_NOT_OK(Tensor::CreateTensor(ptr, TensorImpl::kFlexible, shape, type));

  unsigned char *current_tensor_addr = (*ptr)->GetMutableBuffer();
  int64_t tensor_bytes_remaining = bytes_list.value_size() * pad_size;

  for (int i = 0; i < bytes_list.value_size(); i++) {
    // read string data into tensor
    const std::string &current_element = bytes_list.value(i);
    int return_code =
      memcpy_s(current_tensor_addr, tensor_bytes_remaining, common::SafeCStr(current_element), current_element.size());

    CHECK_FAIL_RETURN_UNEXPECTED(return_code == 0, "memcpy_s failed when reading bytesList element into Tensor");

    current_tensor_addr += current_element.size();
    tensor_bytes_remaining -= current_element.size();

    // pad
    int64_t chars_to_pad = pad_size - current_element.size();
    return_code = memset_s(current_tensor_addr, tensor_bytes_remaining, static_cast<int>(' '), chars_to_pad);
    CHECK_FAIL_RETURN_UNEXPECTED(return_code == 0, "memcpy_s failed when padding Tensor");

    current_tensor_addr += chars_to_pad;
    tensor_bytes_remaining -= chars_to_pad;
  }

  return Status::OK();
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
    CASE_PRINT_HEX(DataType::DE_BOOL, bool);

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

    case DataType::DE_STRING: {
      std::string_view o{""};
      GetItemAt(&o, index);
      out << "\"" << o << "\"";
      break;
    }
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
Status Tensor::AllocateBuffer(const dsize_t &length) {
  if (data_ == nullptr) {
    if (data_allocator_ != nullptr) {
      data_ = data_allocator_->allocate(length);
      RETURN_UNEXPECTED_IF_NULL(data_);
      data_end_ = data_ + length;
    } else {
      data_ = static_cast<unsigned char *>(malloc(length));
      data_end_ = data_ + length;
      RETURN_UNEXPECTED_IF_NULL(data_);
    }
  }
  return Status::OK();
}
const unsigned char *Tensor::GetBuffer() const {
  // This version cannot modify anything.  data_ could possibly be null.
  return data_;
}

unsigned char *Tensor::GetMutableBuffer() {
  if (!shape_.known() || type_ == DataType::DE_UNKNOWN) {
    return nullptr;
  }
  // If the data area is already created, return the pointer to it
  if (data_ != nullptr) {
    return data_;
  } else {
    // If the data area is not created, then identify the memory size based
    // on the shape and type and allocate it.
    if (this->AllocateBuffer(this->SizeInBytes()).IsOk()) {
      return data_;
    } else {
      return nullptr;
    }
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
    RETURN_IF_NOT_OK(shape_.ToFlatIndex(index, &flat_idx));
    *ptr = reinterpret_cast<T *>(data_ + flat_idx * type_.SizeInBytes());

    return Status::OK();
  } else {
    std::string err = "data type not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
}

Status Tensor::GetItemPtr(uchar **ptr, const std::vector<dsize_t> &index, offset_t *length) const {
  if (type_ == DataType::DE_STRING) {
    if (data_ == nullptr) {
      std::string err = "Data is not allocated yet";
      RETURN_STATUS_UNEXPECTED(err);
    }
    dsize_t flat_idx;
    RETURN_IF_NOT_OK(shape_.ToFlatIndex(index, &flat_idx));
    offset_t length_temp = 0;
    RETURN_IF_NOT_OK(GetStringAt(flat_idx, ptr, &length_temp));
    if (length != nullptr) *length = length_temp;
    return Status::OK();
  } else {
    std::string err = "data type not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
}

Status Tensor::StartAddrOfIndex(std::vector<dsize_t> ind, uchar **start_addr_of_index, TensorShape *remaining) {
  if (type() == DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("StartAddrOfIndex does not support string tensors yet.");
  }
  dsize_t flat_ind;
  std::vector<dsize_t> t_shape = shape().AsVector();
  std::vector<dsize_t> r(t_shape.begin() + ind.size(), t_shape.end());
  *remaining = TensorShape(r);
  ind.resize(this->Rank(), 0);  //  same as -> while (ind.size() < this->Rank()) ind.push_back(0);
  RETURN_IF_NOT_OK(shape_.ToFlatIndex(ind, &flat_ind));
  // check if GetBuffer() returns null, we should flag this as an error, this sanity check will only
  // be true is the tensor failed to allocate memory.
  if (GetMutableBuffer() == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid GetBuffer in Tensor, got nullptr");
  }
  *start_addr_of_index = GetMutableBuffer() + flat_ind * this->type().SizeInBytes();
  return Status::OK();
}

Status Tensor::InsertTensor(const std::vector<dsize_t> &ind, const std::shared_ptr<Tensor> &tensor) {
  std::string err_msg;
  err_msg += (this->type() == DataType::DE_STRING) ? "[Tensor] Cannot batch tensors of type string\n" : "";
  err_msg += (!this->shape().known() || !tensor->shape().known()) ? "[Tensor] unknown shape\n" : "";
  err_msg += (ind.size() + tensor->Rank() != this->Rank()) ? "[Tensor] incorrect index\n" : "";
  err_msg += tensor->type().SizeInBytes() != this->type().SizeInBytes() ? "[Tensor] incorrect datatype\n" : "";
  uchar *start_addr_of_ind = nullptr;
  TensorShape remaining_shape({-1});
  err_msg += (!StartAddrOfIndex(ind, &start_addr_of_ind, &remaining_shape).IsOk()) ? "[Tensor] incorrect index\n" : "";
  err_msg += !(remaining_shape == tensor->shape()) ? "[Tensor] memory error\n" : "";
  if (!err_msg.empty()) {
    MS_LOG(DEBUG) << "Insert tensor message: " << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    if (start_addr_of_ind != nullptr) {
      int ret_code =
        memcpy_s(start_addr_of_ind, tensor->SizeInBytes(), tensor->GetMutableBuffer(), tensor->SizeInBytes());
      if (ret_code == 0) {
        return Status::OK();
      } else {
        err_msg += "[Tensor] error in memcpy_s when inserting tensor\n";
        MS_LOG(DEBUG) << "Tensor message: " << err_msg;
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
  std::vector<dsize_t> strides = shape_.Strides();
  uint8_t size = type_.SizeInBytes();
  std::transform(strides.begin(), strides.end(), strides.begin(), [&size](const auto &c) { return c * size; });
  return strides;
}

Status Tensor::GetBufferInfo(Tensor &t, py::buffer_info *out) {
  CHECK_FAIL_RETURN_UNEXPECTED(t.type().IsNumeric(), "Cannot use GetBufferInfo on tensor of strings.");

  std::string format_desc = t.type().GetPybindFormat();
  if (format_desc.empty()) {
    RETURN_STATUS_UNEXPECTED("Cannot convert DE type tp pybind format");
  }
  *out = py::buffer_info(t.GetMutableBuffer(),   /* Pointer to buffer */
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

Status Tensor::GetItemAt(std::string_view *o, const std::vector<dsize_t> &index) const {
  RETURN_UNEXPECTED_IF_NULL(data_);
  RETURN_UNEXPECTED_IF_NULL(o);
  CHECK_FAIL_RETURN_UNEXPECTED(type_ == DataType::DE_STRING, "Tensor type is not a string");

  uchar *start = nullptr;
  offset_t length = 0;
  RETURN_IF_NOT_OK(GetItemPtr(&start, index, &length));
  std::string_view sv{reinterpret_cast<const char *>(start)};
  o->swap(sv);
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
  } else if (type_ == DataType::DE_STRING) {
    GetDataAsNumpyStrings(data);
  } else {
    RETURN_STATUS_UNEXPECTED("Got unexpected type when returning numpy");
  }
  return Status::OK();
}
Status Tensor::GetDataAsNumpyStrings(py::array *data) {
  auto itr = begin<std::string_view>();
  uint64_t max = 0;
  for (; itr != end<std::string_view>(); itr++) {
    max = std::max((*itr).length(), max);
  }
  // if all strings are empty, numpy stores a byte for each string |S1
  max = (max == 0 ? 1 : max);
  uint64_t total_size = shape_.NumOfElements() * max;
  char *tmp_data = reinterpret_cast<char *>(data_allocator_->allocate(total_size));
  if (tmp_data == nullptr) RETURN_STATUS_UNEXPECTED("Cannot create temp array.");
  int ret_code = memset_s(tmp_data, total_size, 0, total_size);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == 0, "Failed to initialize temp memory");

  itr = begin<std::string_view>();
  uint64_t i = 0;
  for (; itr != end<std::string_view>(); itr++, i++) {
    if (!(*itr).empty()) {
      ret_code = memcpy_s(tmp_data + i * max, total_size, (*itr).data(), (*itr).length());
      CHECK_FAIL_RETURN_UNEXPECTED(ret_code == 0, "Failed to copy string data.");
    }
  }
  auto strides = shape_.Strides();
  std::transform(strides.begin(), strides.end(), strides.begin(), [&max](const auto &s) { return s * max; });
  *data = py::array(py::dtype("S" + std::to_string(max)), shape_.AsVector(), strides, tmp_data);
  data_allocator_->deallocate(reinterpret_cast<uchar *>(tmp_data));
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
Status Tensor::GetStringAt(dsize_t index, uchar **string_start, offset_t *length) const {
  CHECK_FAIL_RETURN_UNEXPECTED(type_ == DataType::DE_STRING, "Type is not string");
  RETURN_UNEXPECTED_IF_NULL(data_);
  RETURN_UNEXPECTED_IF_NULL(string_start);
  RETURN_UNEXPECTED_IF_NULL(length);
  auto *offset_ptr = reinterpret_cast<offset_t *>(data_);  // offsets starts here
  offset_t start = offset_ptr[index];
  *string_start = data_ + start;
  *length = offset_ptr[index + 1] - start - 1;  // -1 to skip the \0 from the string length
  return Status::OK();
}
Status Tensor::CopyLastDimAt(const std::shared_ptr<Tensor> &src, const std::vector<dsize_t> &index) {
  CHECK_FAIL_RETURN_UNEXPECTED(src->type() == type_, "Source Tensor has a different type");
  CHECK_FAIL_RETURN_UNEXPECTED(index.back() == 0, "Last dim in index should be 0");

  uint8_t type_size = type_.SizeInBytes();
  size_t len = std::min(src->shape()[-1], shape_[-1]) * type_size;
  dsize_t src_flat_ind = 0, dst_flat_ind = 0;
  RETURN_IF_NOT_OK(src->shape().ToFlatIndex(index, &src_flat_ind));
  RETURN_IF_NOT_OK(shape_.ToFlatIndex(index, &dst_flat_ind));

  const unsigned char *src_addr = src->GetBuffer() + src_flat_ind * type_size;
  unsigned char *dst_addr = GetMutableBuffer() + dst_flat_ind * type_size;
  CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(dst_addr, len, src_addr, len) == 0, "memcpy error");
  return Status::OK();
}
Status Tensor::Slice(std::shared_ptr<Tensor> *out, const std::vector<dsize_t> &indices) {
  CHECK_FAIL_RETURN_UNEXPECTED(shape_.Rank() == 1, "Currently Slice work with rank 1 tensors only.");
  CHECK_FAIL_RETURN_UNEXPECTED(!indices.empty(), "Indices are empty, generated tensor would be empty.");
  if (type_.IsNumeric()) {
    return SliceNumeric(out, indices);
  } else {
    return SliceString(out, indices);
  }
}
Status Tensor::SliceNumeric(std::shared_ptr<Tensor> *out, const std::vector<dsize_t> &indices) {
  RETURN_IF_NOT_OK(
    CreateTensor(out, TensorImpl::kFlexible, TensorShape({static_cast<dsize_t>(indices.size())}), type_));
  (*out)->GetMutableBuffer();
  dsize_t out_index = 0;
  dsize_t dim_length = shape_[0];
  dsize_t type_size = type_.SizeInBytes();
  dsize_t src_start = HandleNeg(indices[0], dim_length);
  uchar *dst_addr = (*out)->data_;
  dsize_t count = 1;

  for (dsize_t i = 0; i < indices.size(); i++) {
    dsize_t cur_index = HandleNeg(indices[i], dim_length);
    CHECK_FAIL_RETURN_UNEXPECTED(
      cur_index >= 0 && cur_index < dim_length,
      "Index " + std::to_string(indices[i]) + " is out of bounds [0," + std::to_string(dim_length) + ")");
    if (i < indices.size() - 1) {
      dsize_t next_index = HandleNeg(indices[i + 1], dim_length);
      if (next_index == cur_index + 1) {
        count++;
        continue;
      }
    }
    memcpy_s(dst_addr + out_index * type_size, (*out)->SizeInBytes(), data_ + src_start * type_size, count * type_size);
    out_index += count;
    if (i < indices.size() - 1) {
      src_start = HandleNeg(indices[i + 1], dim_length);  // next index
    }
    count = 1;
  }
  return Status::OK();
}
Status Tensor::SliceString(std::shared_ptr<Tensor> *out, const std::vector<dsize_t> &indices) {
  dsize_t dim_length = shape_[0];
  std::vector<std::string> strings;
  for (dsize_t index : indices) {
    dsize_t cur_index = HandleNeg(index, dim_length);
    CHECK_FAIL_RETURN_UNEXPECTED(
      cur_index >= 0 && cur_index < dim_length,
      "Index " + std::to_string(index) + " is out of bounds [0," + std::to_string(dim_length) + ")");
    std::string_view sv;
    GetItemAt(&sv, {cur_index});
    strings.emplace_back(sv);
  }
  return CreateTensor(out, strings);
}

}  // namespace dataset
}  // namespace mindspore
