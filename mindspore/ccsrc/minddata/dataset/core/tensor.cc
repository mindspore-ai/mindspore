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
#include "minddata/dataset/core/tensor.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <vector>
#include <utility>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/core/cv_tensor.h"
#endif
#include "minddata/dataset/core/global_context.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/core/pybind_support.h"
#endif
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/util/validators.h"
#include "utils/ms_utils.h"

#ifdef ENABLE_PYTHON
namespace py = pybind11;
#endif

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

Tensor::Tensor(Tensor &&other) noexcept
    : shape_(other.shape()),
      type_(other.type()),
      data_(other.GetMutableBuffer()),
      data_end_(other.data_end_),
      data_allocator_(std::move(other.data_allocator_)) {
#ifdef ENABLE_PYTHON
  if (type_.value() == DataType::DE_PYTHON) {
    py::gil_scoped_acquire gil_acquire;
    python_dict_ = (other.python_dict_);
  }
#endif
  other.Invalidate();
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (&other != this) {
    shape_ = other.shape();
    type_ = other.type();
    data_ = other.GetMutableBuffer();
    data_end_ = other.data_end_;
    data_allocator_ = std::move(other.data_allocator_);
    yuv_shape_ = other.yuv_shape_;
#ifdef ENABLE_PYTHON
    if (type_.value() == DataType::DE_PYTHON) {
      py::gil_scoped_acquire gil_acquire;
      python_dict_ = (other.python_dict_);
    }
#endif
    other.Invalidate();
  }
  return *this;
}

Status Tensor::CreateEmpty(const TensorShape &shape, const DataType &type, TensorPtr *out) {
  CHECK_FAIL_RETURN_UNEXPECTED(shape.known(), "Failed to create empty tensor, tensor shape is unknown.");
  CHECK_FAIL_RETURN_UNEXPECTED(type != DataType::DE_UNKNOWN, "Failed to create empty tensor, data type is unknown.");
  RETURN_UNEXPECTED_IF_NULL(out);
  const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
  *out = std::allocate_shared<Tensor>(*alloc, shape, type);
  CHECK_FAIL_RETURN_UNEXPECTED(out != nullptr, "Failed to create empty tensor, allocate memory failed.");
  // if it's a string tensor and it has no elements, Just initialize the shape and type.
  if (!type.IsNumeric()) {
    if (shape.NumOfElements() == 0) {
      return Status::OK();
    } else {
      RETURN_STATUS_UNEXPECTED(
        "Failed to create empty tensor, number of elements should be 0 when data type is string.");
    }
  }

  int64_t byte_size = (*out)->SizeInBytes();

  // Don't allocate if we have a tensor with no elements.
  if (byte_size != 0) {
    RETURN_IF_NOT_OK((*out)->AllocateBuffer(byte_size));
  }
  return Status::OK();
}

Status Tensor::CreateFromMemory(const TensorShape &shape, const DataType &type, const uchar *src, TensorPtr *out) {
  RETURN_IF_NOT_OK(CreateEmpty(shape, type, out));
  if (src != nullptr && out != nullptr) {
    // Given the shape/type of this tensor, compute the data size and copy in the input bytes.
    int64_t byte_size = (*out)->SizeInBytes();
    if (byte_size == 0) {
      return Status::OK();
    }
    std::string err_msg =
      "Failed to copy data into tensor. If GeneratorDataset(source=Pyfunc, ...) or map(operations=Pyfunc, ...) is "
      "used, please check whether the memory of the "
      "Numpy object returned by Pyfunc has been unexpectedly freed. Adding copy.deepcopy(numpy_object) before "
      "numpy_object returned by Pyfunc maybe solve the issue. For more details, please refer to the FAQ at "
      "https://www.mindspore.cn/docs/en/master/faq/data_processing.html.";
    if (byte_size < SECUREC_MEM_MAX_LEN) {
      int ret_code = memcpy_s((*out)->data_, byte_size, src, byte_size);
      CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK, err_msg);
    } else {
      auto ret_code = std::memcpy((*out)->data_, src, byte_size);
      CHECK_FAIL_RETURN_UNEXPECTED(ret_code == (*out)->data_, err_msg);
    }
  }
  return Status::OK();
}

Status Tensor::CreateFromMemory(const TensorShape &shape, const DataType &type, const uchar *src, const dsize_t &length,
                                TensorPtr *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
  *out = std::allocate_shared<Tensor>(*alloc, shape, type);
  CHECK_FAIL_RETURN_UNEXPECTED(out != nullptr, "Allocate memory failed.");
  if (type.IsNumeric()) {
    dsize_t calculated_length = (*out)->SizeInBytes();
    CHECK_FAIL_RETURN_UNEXPECTED(calculated_length == length, "Length of source data does not match the shape.");
  } else {
    // min_length is the length of a tensor with empty strings
    // min_length = the number of bytes needed to store the offsets + 1 byte for each element
    dsize_t min_length = (shape.NumOfElements() + 1) * kOffsetSize + shape.NumOfElements();
    CHECK_FAIL_RETURN_UNEXPECTED(min_length <= length, "Length of source data does not match the shape.");
  }

  RETURN_IF_NOT_OK((*out)->AllocateBuffer(length));
  if (length == 0) {
    return Status::OK();
  }
  RETURN_UNEXPECTED_IF_NULL(src);  // needs to be here as we may return early without using src content (empty tensors)
  if (length < SECUREC_MEM_MAX_LEN) {
    int ret_code = memcpy_s((*out)->data_, length, src, length);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK, "Failed to copy data into tensor.");
  } else {
    auto ret_code = std::memcpy((*out)->data_, src, length);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == (*out)->data_, "Failed to copy data into tensor.");
  }

  return Status::OK();
}

#ifdef ENABLE_PYTHON
Status Tensor::CreateFromNpString(py::array arr, std::shared_ptr<Tensor> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  std::vector<dsize_t> shape;
  for (size_t i = 0; i < arr.ndim(); i++) {
    shape.push_back(static_cast<dsize_t>(arr.shape()[i]));
  }
  arr.resize({arr.size()});  // flatten the py::array so we can iterate once
  std::vector<std::string> strings;
  strings.reserve(arr.size());
  (void)std::for_each(arr.begin(), arr.end(),
                      [&strings](const auto &s) { strings.emplace_back(py::cast<py::str>(s)); });
  arr.resize(shape);  // resize arr back to the original shape

  if (arr.dtype().kind() == 'U') {  // numpy dtype type is "U"
    RETURN_IF_NOT_OK(CreateFromVector(strings, TensorShape{shape}, DataType(DataType::DE_STRING), out));
  } else {  // numpy dtype type is "S"
    RETURN_IF_NOT_OK(CreateFromVector(strings, TensorShape{shape}, DataType(DataType::DE_BYTES), out));
  }

  return Status::OK();
}

Status Tensor::CreateFromNpArray(const py::array &arr, std::shared_ptr<Tensor> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  DataType type = DataType::FromNpArray(arr);
  CHECK_FAIL_RETURN_UNEXPECTED(type != DataType::DE_UNKNOWN,
                               "Failed to create tensor from numpy array, data type is unknown.");

  if (type.IsString()) {
    return CreateFromNpString(arr, out);
  }

  std::vector<dsize_t> shape;
  std::vector<dsize_t> strides;
  // check if strides are contiguous
  bool is_strided = false;
  dsize_t count = arr.size();
  for (dsize_t i = 0; i < arr.ndim(); i++) {
    shape.push_back(static_cast<dsize_t>(arr.shape()[i]));
    strides.push_back(static_cast<dsize_t>(arr.strides()[i]));
    // in case of empty array num_items=0
    if (count != 0 && shape.size() > i && shape[i] != 0) {
      count /= shape[i];
      if (strides[i] != arr.itemsize() * count) {
        is_strided = true;
      }
    }
  }

  unsigned char *data = static_cast<unsigned char *>(arr.request().ptr);

  if (is_strided) {
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape(shape), type, out));
    RETURN_IF_NOT_OK(CopyStridedArray((*out)->data_, data, shape, strides, (*out)->type_.SizeInBytes()));
  } else {
    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(TensorShape(shape), type, data, out));
  }
  return Status::OK();
}

Status Tensor::CreateFromPythonObject(py::object obj, std::shared_ptr<Tensor> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  std::vector<dsize_t> shape{};
  DataType type = DataType(DataType::DE_PYTHON);
  const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
  *out = std::allocate_shared<Tensor>(*alloc, TensorShape({0}), type);
  {
    py::gil_scoped_acquire gil_acquire;
    (*out)->python_dict_ = obj;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(out != nullptr, "Failed to create a tensor for python object.");
  return Status::OK();
}

#endif

#ifndef ENABLE_ANDROID
Status Tensor::CreateFromByteList(const dataengine::BytesList &bytes_list, const TensorShape &shape, TensorPtr *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  const TensorAlloc *alloc = GlobalContext::Instance()->tensor_allocator();
  *out = std::allocate_shared<Tensor>(*alloc, TensorShape({static_cast<dsize_t>(bytes_list.value_size())}),
                                      DataType(DataType::DE_STRING));
  CHECK_FAIL_RETURN_UNEXPECTED(out != nullptr, "Allocate memory failed.");
  // total bytes needed = offset array + strings
  // offset array needs to store one offset var per element + 1 extra to get the length of the last string.
  // strings will be null-terminated --> need 1 extra byte per element
  dsize_t num_bytes = (kOffsetSize) * (*out)->shape_.NumOfElements() + kOffsetSize + bytes_list.ByteSizeLong();

  (*out)->data_ = (*out)->data_allocator_->allocate(num_bytes);

  auto offset_arr = reinterpret_cast<offset_t *>((*out)->data_);
  uchar *buf = (*out)->GetStringsBuffer();

  offset_t offset = buf - (*out)->data_;  // the first string will start here
  int32_t i = 0;
  for (; i < bytes_list.value_size(); i++) {
    const std::string &str = bytes_list.value(i);
    //  insert the start index of the string.
    offset_arr[i] = offset;
    // total bytes are reduced by kOffsetSize
    num_bytes -= kOffsetSize;
    // insert actual string
    int ret_code = memcpy_s((*out)->data_ + offset, num_bytes, common::SafeCStr(str), str.length() + 1);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK, "Cannot copy string into Tensor");
    //  next string will be stored right after the current one.
    offset = offset + str.length() + 1;
    // total bytes are reduced by the length of the string
    num_bytes -= str.length() + 1;
  }
  // store one more offset value so we can get the length of the last string
  // length[last_element] = offset_arr[last_element + 1] - offset_arr[last_element]
  offset_arr[i] = offset;

  (*out)->data_end_ = (*out)->data_ + offset_arr[i];

  MS_ASSERT(num_bytes == 0);
  RETURN_IF_NOT_OK((*out)->Reshape(shape));
  return Status::OK();
}
#endif

Status Tensor::CreateFromFile(const std::string &path, std::shared_ptr<Tensor> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  Path file(path);
  if (file.IsDirectory()) {
    RETURN_STATUS_UNEXPECTED("Invalid file found: " + path + ", should be file, but got directory.");
  }
  std::ifstream fs;
  fs.open(path, std::ios::binary | std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(!fs.fail(), "Failed to open file: " + path);
  int64_t num_bytes = fs.seekg(0, std::ios::end).tellg();
  CHECK_FAIL_RETURN_UNEXPECTED(num_bytes < kDeMaxDim, "Invalid file to allocate tensor memory, check path: " + path);
  CHECK_FAIL_RETURN_UNEXPECTED(fs.seekg(0, std::ios::beg).good(), "Failed to find size of file, check path: " + path);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape{num_bytes}, DataType(DataType::DE_UINT8), out));
  int64_t written_bytes = fs.read(reinterpret_cast<char *>((*out)->GetMutableBuffer()), num_bytes).gcount();
  if (!(written_bytes == num_bytes && fs.good())) {
    fs.close();
    RETURN_STATUS_UNEXPECTED("Error in writing to tensor, check path: " + path);
  }
  fs.close();
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status Tensor::CreateFromByteList(const dataengine::BytesList &bytes_list, const TensorShape &shape,
                                  const DataType &type, dsize_t pad_size, TensorPtr *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, type, out));

  RETURN_UNEXPECTED_IF_NULL(out);
  unsigned char *current_tensor_addr = (*out)->GetMutableBuffer();
  int64_t tensor_bytes_remaining = bytes_list.value_size() * pad_size;

  for (int i = 0; i < bytes_list.value_size(); i++) {
    // read string data into tensor
    const std::string &current_element = bytes_list.value(i);
    int return_code =
      memcpy_s(current_tensor_addr, tensor_bytes_remaining, common::SafeCStr(current_element), current_element.size());

    CHECK_FAIL_RETURN_UNEXPECTED(return_code == EOK, "memcpy_s failed when reading bytesList element into Tensor");

    current_tensor_addr += current_element.size();
    tensor_bytes_remaining -= current_element.size();

    // pad
    int64_t chars_to_pad = pad_size - current_element.size();
    return_code = memset_s(current_tensor_addr, tensor_bytes_remaining, static_cast<int>(' '), chars_to_pad);
    CHECK_FAIL_RETURN_UNEXPECTED(return_code == EOK, "memcpy_s failed when padding Tensor");

    current_tensor_addr += chars_to_pad;
    tensor_bytes_remaining -= chars_to_pad;
  }

  return Status::OK();
}
#endif

// Memcpy the given strided array's used part to consecutive memory
// Consider a 3-d array
// A[(i * shape[1] + j) * shape[2] + k] = B[i][j][k] = C[i * strides[0] + j * strides[1] + k * strides[2]]
// Here we convert array C to array A, by memcpy index by index (Note that not all elements in C is copied)
Status Tensor::CopyStridedArray(unsigned char *dst, unsigned char *src, std::vector<dsize_t> shape,
                                std::vector<dsize_t> strides, uint8_t type_size) {
  RETURN_UNEXPECTED_IF_NULL(dst);
  RETURN_UNEXPECTED_IF_NULL(src);
  dsize_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  for (dsize_t i = 0; i < size; ++i) {
    dsize_t offset = 0;
    dsize_t count = i;
    for (size_t j = 0; j < shape.size(); ++j) {
      // convert 1d array's index to 3d array's index (A -> B)
      CHECK_FAIL_RETURN_UNEXPECTED(shape[shape.size() - 1 - j] != 0, "Invalid data, shape can't be zero.");
      dsize_t idx = count % shape[shape.size() - 1 - j];
      count /= shape[shape.size() - 1 - j];
      // calculate the raw data offset based on strides (B -> C)
      offset += idx * strides[shape.size() - 1 - j];
      // once count = 0, the following idxes are all zero, skip them
      if (count == 0) {
        break;
      }
    }
    // strides already consider byte size of the data type, but dst doesn't.
    // dst[i] = dst + i * type_size = src + offset
    int ret_code = memcpy_s(dst + i * type_size, type_size, src + offset, type_size);
    if (ret_code != EOK) {
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
#ifdef ENABLE_PYTHON
  try {
    if (Py_IsInitialized()) {
      if (static_cast<bool>(python_dict_)) {  // if it contains data
        // Acquire Python GIL
        py::gil_scoped_acquire gil_acquire;
        if (python_dict_.ref_count() == 1) {  // if we aren't referencing it anywhere else
          python_dict_.dec_ref();             // manually set the ref count to zero (to be garbage collected by Python)
          python_dict_ = py::none();          // wrapper now pointing to a meaningful thing (added to avoid a segfault)
        }
      }
    }
  } catch (py::error_already_set &e) {
    // ignore exceptions as everything could be shutting down at this point
  }
#endif
}  // namespace dataset

bool Tensor::operator==(const Tensor &rhs) const {
#ifdef ENABLE_PYTHON
  if (type_.value() == DataType::DE_PYTHON) {  // we are holding a python object
    if (static_cast<bool>(python_dict_) && static_cast<bool>(rhs.python_dict_) && python_dict_ == rhs.python_dict_) {
      return true;
    }
    return false;
  }
#endif
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
  MS_ASSERT(data_);

  switch (type_.value()) {
    CASE_PRINT_HEX(DataType::DE_BOOL, bool)

    CASE_PRINT_HEX(DataType::DE_INT8, int8_t)

    CASE_PRINT_HEX(DataType::DE_UINT8, uint8_t)

    CASE_PRINT(DataType::DE_INT16, int16_t)

    CASE_PRINT(DataType::DE_UINT16, uint16_t)

    CASE_PRINT(DataType::DE_INT32, int32_t)

    CASE_PRINT(DataType::DE_UINT32, uint32_t)

    CASE_PRINT(DataType::DE_INT64, int64_t)

    CASE_PRINT(DataType::DE_UINT64, uint64_t)

    CASE_PRINT(DataType::DE_FLOAT16, float16)

    CASE_PRINT(DataType::DE_FLOAT32, float)

    CASE_PRINT(DataType::DE_FLOAT64, double)

    case DataType::DE_STRING: {
      std::string_view o{""};
      rc = GetItemAt(&o, index);
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
#ifdef ENABLE_PYTHON
  } else if (static_cast<bool>(python_dict_)) {
    std::string s;
    {
      py::gil_scoped_acquire gil_acquire;
      s = py::str(python_dict_);
    }
    out << s;
#endif
  } else {
    out << "[Data area is null]";
  }
}

void Tensor::PrintData(std::ostream &out) const {
  if (data_) {
    PrintRecursive(out, 0, std::vector<dsize_t>{});
  }
}

Status Tensor::AllocateBuffer(const dsize_t &length) {
  RETURN_UNEXPECTED_IF_NULL(data_allocator_);
  if (data_ == nullptr) {
    data_ = data_allocator_->allocate(length);
    CHECK_FAIL_RETURN_UNEXPECTED(data_ != nullptr, "Failed to allocate memory for tensor.");
    data_end_ = data_ + length;
  }
  return Status::OK();
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
#ifdef ENABLE_PYTHON
  if (type_.value() == DataType::DE_PYTHON) {
    py::gil_scoped_acquire gil_acquire;
    python_dict_ = py::none();
  }
#endif
}

template <typename T>
Status Tensor::GetItemPtr(T **ptr, const std::vector<dsize_t> &index) const {
  RETURN_UNEXPECTED_IF_NULL(ptr);
  if (type_.IsCompatible<T>()) {
    if (data_ == nullptr) {
      std::string err = "Data is not allocated yet";
      RETURN_STATUS_UNEXPECTED(err);
    }
    dsize_t flat_idx;
    RETURN_IF_NOT_OK(shape_.ToFlatIndex(index, &flat_idx));
    *ptr = reinterpret_cast<T *>(data_ + flat_idx * type_.SizeInBytes());
    RETURN_UNEXPECTED_IF_NULL(*ptr);

    return Status::OK();
  } else {
    std::string err = "data type not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
}

Status Tensor::GetItemPtr(uchar **ptr, const std::vector<dsize_t> &index, offset_t *length) const {
  RETURN_UNEXPECTED_IF_NULL(ptr);
  RETURN_UNEXPECTED_IF_NULL(length);
  if (type_.IsString()) {
    if (data_ == nullptr) {
      std::string err = "Data is not allocated yet";
      RETURN_STATUS_UNEXPECTED(err);
    }
    dsize_t flat_idx;
    RETURN_IF_NOT_OK(shape_.ToFlatIndex(index, &flat_idx));
    offset_t length_temp = 0;
    RETURN_IF_NOT_OK(GetStringAt(flat_idx, ptr, &length_temp));
    *length = length_temp;
    return Status::OK();
  } else {
    std::string err = "data type not compatible";
    RETURN_STATUS_UNEXPECTED(err);
  }
}

Status Tensor::StartAddrOfIndex(std::vector<dsize_t> ind, uchar **start_addr_of_index, TensorShape *remaining) {
  RETURN_UNEXPECTED_IF_NULL(start_addr_of_index);
  RETURN_UNEXPECTED_IF_NULL(remaining);
  if (type().IsString()) {
    RETURN_STATUS_UNEXPECTED("StartAddrOfIndex does not support string and bytes tensors yet.");
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

Status Tensor::InsertTensor(const std::vector<dsize_t> &ind, const std::shared_ptr<Tensor> &tensor,
                            const bool partial_insert) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  std::string err_msg;
  if (partial_insert) {
    err_msg += (ind.size() != 1)
                 ? "[Tensor] only supports 1D insertion of elements not along the full length of the axis\n"
                 : "";
    err_msg +=
      (ind.at(0) + tensor->shape().NumOfElements() > shape().NumOfElements()) ? "[Tensor] incorrect index\n" : "";
  } else {
    err_msg += (ind.size() + tensor->Rank() != Rank()) ? "[Tensor] incorrect index\n" : "";
  }
  err_msg += (type().IsString()) ? "[Tensor] Cannot insert into a tensor of type string or bytes\n" : "";
  err_msg += (!shape().known() || !tensor->shape().known()) ? "[Tensor] unknown shape\n" : "";

  err_msg += tensor->type().SizeInBytes() != type().SizeInBytes() ? "[Tensor] incorrect datatype\n" : "";
  uchar *start_addr_of_ind = nullptr;
  if (partial_insert) {
    TensorShape remaining_shape = tensor->shape();
    err_msg +=
      (!StartAddrOfIndex(ind, &start_addr_of_ind, &remaining_shape).IsOk()) ? "[Tensor] incorrect index\n" : "";
  } else {
    TensorShape remaining_shape = TensorShape::CreateUnknownRankShape();
    err_msg +=
      (!StartAddrOfIndex(ind, &start_addr_of_ind, &remaining_shape).IsOk()) ? "[Tensor] incorrect index\n" : "";
    err_msg += !(remaining_shape == tensor->shape()) ? "[Tensor] memory error\n" : "";
  }

  if (!err_msg.empty()) {
    MS_LOG(DEBUG) << "Insert tensor message: " << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    if (start_addr_of_ind != nullptr) {
      int ret_code =
        memcpy_s(start_addr_of_ind, tensor->SizeInBytes(), tensor->GetMutableBuffer(), tensor->SizeInBytes());
      if (ret_code == EOK) {
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

std::vector<dsize_t> Tensor::Strides() const {
  std::vector<dsize_t> strides = shape_.Strides();
  uint8_t size = type_.SizeInBytes();
  (void)std::transform(strides.begin(), strides.end(), strides.begin(), [&size](const auto &c) { return c * size; });
  return strides;
}

#ifdef ENABLE_PYTHON
Status Tensor::GetBufferInfo(Tensor *t, py::buffer_info *out) {
  RETURN_UNEXPECTED_IF_NULL(t);
  RETURN_UNEXPECTED_IF_NULL(out);
  CHECK_FAIL_RETURN_UNEXPECTED(t->type().IsNumeric(), "Cannot use GetBufferInfo on tensor of strings or bytes.");

  std::string format_desc = t->type().GetPybindFormat();
  if (format_desc.empty()) {
    RETURN_STATUS_UNEXPECTED("Cannot convert DE type to pybind format");
  }
  *out = py::buffer_info(t->GetMutableBuffer(),   /* Pointer to buffer */
                         t->type().SizeInBytes(), /* Size of one scalar */
                         format_desc,             /* Python struct-style format descriptor */
                         t->Rank(),               /* Number of dimensions */
                         t->shape().AsVector(),   /* Buffer dimensions */
                         t->Strides());
  return Status::OK();
}
#endif

Status Tensor::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["shape"] = shape_.AsVector();
  args["type"] = type_.ToString();
  if (type_ == DataType::DE_BOOL) {
    RETURN_IF_NOT_OK(to_json_convert<bool>(&args));
  } else if (type_ == DataType::DE_INT8) {
    RETURN_IF_NOT_OK(to_json_convert<int8_t>(&args));
  } else if (type_ == DataType::DE_INT16) {
    RETURN_IF_NOT_OK(to_json_convert<int16_t>(&args));
  } else if (type_ == DataType::DE_INT32) {
    RETURN_IF_NOT_OK(to_json_convert<int32_t>(&args));
  } else if (type_ == DataType::DE_INT64) {
    RETURN_IF_NOT_OK(to_json_convert<int64_t>(&args));
  } else if (type_ == DataType::DE_UINT8) {
    RETURN_IF_NOT_OK(to_json_convert<uint8_t>(&args));
  } else if (type_ == DataType::DE_UINT16) {
    RETURN_IF_NOT_OK(to_json_convert<uint16_t>(&args));
  } else if (type_ == DataType::DE_UINT32) {
    RETURN_IF_NOT_OK(to_json_convert<uint32_t>(&args));
  } else if (type_ == DataType::DE_UINT64) {
    RETURN_IF_NOT_OK(to_json_convert<uint64_t>(&args));
  } else if (type_ == DataType::DE_FLOAT32) {
    RETURN_IF_NOT_OK(to_json_convert<float>(&args));
  } else if (type_ == DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(to_json_convert<double>(&args));
  } else if (type_.IsString()) {
    std::vector<std::string> data_out;
    for (auto it = this->begin<std::string_view>(); it != this->end<std::string_view>(); ++it) {
      data_out.emplace_back(*it);
    }
    args["data"] = data_out;
  } else {
    return Status(StatusCode::kMDUnexpectedError, "Type is not supported for tensor");
  }
  *out_json = args;
  return Status::OK();
}

template <typename T>
Status Tensor::to_json_convert(nlohmann::json *args) {
  std::vector<T> data_out;
  for (auto it = this->begin<T>(); it != this->end<T>(); it++) {
    data_out.emplace_back(*it);
  }
  (*args)["data"] = data_out;
  return Status::OK();
}

Status Tensor::from_json(nlohmann::json op_params, std::shared_ptr<Tensor> *tensor) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "shape", "Tensor"));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "type", "Tensor"));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "data", "Tensor"));
  std::string type = op_params["type"];
  std::vector<dsize_t> list = op_params["shape"];
  TensorShape shape = TensorShape(list);
  if (type == "bool") {
    RETURN_IF_NOT_OK(from_json_convert<bool>(op_params["data"], shape, tensor));
  } else if (type == "int8") {
    RETURN_IF_NOT_OK(from_json_convert<int8_t>(op_params["data"], shape, tensor));
  } else if (type == "int16") {
    RETURN_IF_NOT_OK(from_json_convert<int16_t>(op_params["data"], shape, tensor));
  } else if (type == "int32") {
    RETURN_IF_NOT_OK(from_json_convert<int32_t>(op_params["data"], shape, tensor));
  } else if (type == "int64") {
    RETURN_IF_NOT_OK(from_json_convert<int64_t>(op_params["data"], shape, tensor));
  } else if (type == "uint8") {
    RETURN_IF_NOT_OK(from_json_convert<uint8_t>(op_params["data"], shape, tensor));
  } else if (type == "uint16") {
    RETURN_IF_NOT_OK(from_json_convert<uint16_t>(op_params["data"], shape, tensor));
  } else if (type == "uint32") {
    RETURN_IF_NOT_OK(from_json_convert<uint32_t>(op_params["data"], shape, tensor));
  } else if (type == "uint64") {
    RETURN_IF_NOT_OK(from_json_convert<uint64_t>(op_params["data"], shape, tensor));
  } else if (type == "float32") {
    RETURN_IF_NOT_OK(from_json_convert<float>(op_params["data"], shape, tensor));
  } else if (type == "float64") {
    RETURN_IF_NOT_OK(from_json_convert<double>(op_params["data"], shape, tensor));
  } else if (type == "string") {
    RETURN_IF_NOT_OK(from_json_convert(op_params["data"], shape, DataType(DataType::DE_STRING), tensor));
  } else if (type == "bytes") {
    RETURN_IF_NOT_OK(from_json_convert(op_params["data"], shape, DataType(DataType::DE_BYTES), tensor));
  } else {
    return Status(StatusCode::kMDUnexpectedError, "Type is not supported for tensor");
  }
  return Status::OK();
}

template <typename T>
Status Tensor::from_json_convert(const nlohmann::json &json_data, const TensorShape &shape,
                                 std::shared_ptr<Tensor> *tensor) {
  std::vector<T> data = json_data;
  RETURN_IF_NOT_OK(CreateFromVector(data, shape, tensor));
  return Status::OK();
}

Status Tensor::from_json_convert(const nlohmann::json &json_data, const TensorShape &shape, const DataType &type,
                                 std::shared_ptr<Tensor> *tensor) {
  std::vector<std::string> data = json_data;
  RETURN_IF_NOT_OK(CreateFromVector(data, shape, type, tensor));
  return Status::OK();
}

template <typename T>
Status Tensor::GetItemAt(T *o, const std::vector<dsize_t> &index) const {
  RETURN_UNEXPECTED_IF_NULL(o);
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
  CHECK_FAIL_RETURN_UNEXPECTED(type_.IsString(), "Tensor type is not of string or bytes.");

  uchar *start = nullptr;
  offset_t length = 0;
  RETURN_IF_NOT_OK(GetItemPtr(&start, index, &length));
  std::string_view sv{reinterpret_cast<const char *>(start), length};
  o->swap(sv);
  return Status::OK();
}

#ifdef ENABLE_PYTHON
// return data as numpy, should return status
Status Tensor::GetDataAsNumpy(py::array *data) {
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
  } else if (type_.IsString()) {
    RETURN_IF_NOT_OK(GetDataAsNumpyStrings(data));
  } else {
    RETURN_STATUS_UNEXPECTED("Got unexpected type when returning numpy");
  }
  return Status::OK();
}

Status Tensor::GetDataAsNumpyStrings(py::array *data) {
  RETURN_UNEXPECTED_IF_NULL(data);
  if (type_ == DataType::DE_STRING) {
    RETURN_IF_NOT_OK(GetDataAsNumpyStrings<py::str>(data));
  } else if (type_ == DataType::DE_BYTES) {
    RETURN_IF_NOT_OK(GetDataAsNumpyStrings<py::bytes>(data));
  } else {
    RETURN_STATUS_UNEXPECTED("Can not convert a numeric Tensor to a string NumPy array.");
  }
  return Status::OK();
}

Status Tensor::GetDataAsPythonObject(py::dict *data) {
  RETURN_UNEXPECTED_IF_NULL(data);
  {
    py::gil_scoped_acquire gil_acquire;
    *data = python_dict_;
  }
  return Status::OK();
}
#endif

void Tensor::Squeeze() { shape_ = shape_.Squeeze(); }

template <typename T>
Status Tensor::GetUnsignedIntAt(T *o, const std::vector<dsize_t> &index) const {
  RETURN_UNEXPECTED_IF_NULL(o);
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
  RETURN_UNEXPECTED_IF_NULL(o);
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
  RETURN_UNEXPECTED_IF_NULL(o);
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
  CHECK_FAIL_RETURN_UNEXPECTED(type_.IsString(), "Type is not string or bytes.");
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
  RETURN_UNEXPECTED_IF_NULL(src);
  CHECK_FAIL_RETURN_UNEXPECTED(src->type() == type_, "Source Tensor has a different type");
  CHECK_FAIL_RETURN_UNEXPECTED(index.back() == 0, "Last dim in index should be 0");

  uint8_t type_size = type_.SizeInBytes();
  size_t len = std::min(src->shape()[-1], shape_[-1]) * type_size;
  dsize_t src_flat_ind = 0, dst_flat_ind = 0;
  RETURN_IF_NOT_OK(src->shape().ToFlatIndex(index, &src_flat_ind));
  RETURN_IF_NOT_OK(shape_.ToFlatIndex(index, &dst_flat_ind));

  const unsigned char *src_addr = src->GetBuffer() + src_flat_ind * type_size;
  unsigned char *dst_addr = GetMutableBuffer() + dst_flat_ind * type_size;
  CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(dst_addr, len, src_addr, len) == EOK, "memcpy error");
  return Status::OK();
}

Status Tensor::GetSliceOption(const SliceOption &slice_option, const int32_t &slice_index,
                              SliceOption *slice_option_ptr) {
  RETURN_UNEXPECTED_IF_NULL(slice_option_ptr);
  if (slice_option.indices_.empty() && !slice_option.slice_.valid()) {
    RETURN_STATUS_UNEXPECTED("Both indices and slices can not be empty.");
  }

  if (!slice_option.indices_.empty() && slice_option.slice_.valid()) {
    RETURN_STATUS_UNEXPECTED("Both indices and slices can not be given.");
  }

  CHECK_FAIL_RETURN_UNEXPECTED(shape_.Size() > slice_index, "Invalid shape, should be greater than slices index.");
  // if slice object was provided, indices should be empty. Generate indices from the slice object.
  if (slice_option.indices_.empty()) {
    // check if slice is valid
    mindspore::dataset::Slice slice_copy = slice_option.slice_;
    slice_copy.start_ = HandleNeg(slice_option.slice_.start_, shape_[slice_index]);
    slice_copy.stop_ = HandleNeg(slice_option.slice_.stop_, shape_[slice_index]);
    slice_copy.start_ = slice_copy.start_ < 0 ? 0 : slice_copy.start_;
    slice_copy.stop_ = slice_copy.stop_ < 0 ? 0 : slice_copy.stop_;
    dsize_t max_idx = shape_[slice_index];
    slice_copy.start_ = slice_copy.start_ > max_idx ? max_idx : slice_copy.start_;
    slice_copy.stop_ = slice_copy.stop_ > max_idx ? max_idx : slice_copy.stop_;
    *slice_option_ptr = SliceOption(slice_copy);
  } else {
    // indices validation
    std::vector<dsize_t> indices_copy;
    for (int j = 0; j < slice_option.indices_.size(); j++) {
      dsize_t index = HandleNeg(slice_option.indices_[j], shape_[slice_index]);
      CHECK_FAIL_RETURN_UNEXPECTED(index < shape_[slice_index] && index >= 0,
                                   "Index " + std::to_string(index) + " is out of bounds.");
      indices_copy.emplace_back(index);
    }
    *slice_option_ptr = SliceOption(indices_copy);
  }
  return Status::OK();
}

Status Tensor::Slice(std::shared_ptr<Tensor> *out, const std::vector<SliceOption> &slice_options) {
  RETURN_UNEXPECTED_IF_NULL(out);
  std::vector<SliceOption> converted_slice_objects;

  CHECK_FAIL_RETURN_UNEXPECTED(slice_options.size() <= static_cast<size_t>(std::numeric_limits<dsize_t>::max()),
                               "The size of slice_options_ must not be more than \"INT64_MAX\".");
  for (size_t k = 0; k < slice_options.size(); k++) {
    SliceOption slice_option = slice_options[k];

    if (slice_option.all_) {
      auto slice = mindspore::dataset::Slice(shape_[static_cast<dsize_t>(k)]);
      converted_slice_objects.emplace_back(slice);
      continue;
    }

    CHECK_FAIL_RETURN_UNEXPECTED(k <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                                 "GetSliceOption() can't function properly if there are "
                                 "more than \"INT32_MAX\" slice options");
    SliceOption slice_option_item(false);
    RETURN_IF_NOT_OK(GetSliceOption(slice_option, static_cast<int32_t>(k), &slice_option_item));
    converted_slice_objects.emplace_back(slice_option_item);
  }

  // partial slices, pass in the rest
  if (slice_options.size() != Rank()) {
    for (auto j = static_cast<dsize_t>(slice_options.size()); j < Rank(); j++) {
      mindspore::dataset::Slice slice = mindspore::dataset::Slice(0, shape_[j]);
      converted_slice_objects.emplace_back(SliceOption(slice));
    }
  }

  // determine final shape:
  TensorShape t = TensorShape({});
  dsize_t slice_len = slice_options.size();
  dsize_t slice_len_ind;
  for (int i = 0; i < shape_.Rank(); i++) {
    if (i < slice_len) {
      // if it's a slice
      if (converted_slice_objects[i].indices_.empty() && converted_slice_objects[i].slice_.step_ != 0) {
        slice_len_ind = (converted_slice_objects[i].slice_.stop_ - converted_slice_objects[i].slice_.start_) /
                        converted_slice_objects[i].slice_.step_;
        if ((converted_slice_objects[i].slice_.stop_ - converted_slice_objects[i].slice_.start_) %
              converted_slice_objects[i].slice_.step_ !=
            0) {
          slice_len_ind++;
        }
        // account for slices that would return no data
        slice_len_ind = slice_len_ind < 0 ? 0 : slice_len_ind;
        t = t.AppendDim(slice_len_ind);
      } else {
        // if its a vector of indices
        // need to introduce a way of handling indices and slices
        if (!converted_slice_objects[i].indices_.empty()) {
          t = t.AppendDim(converted_slice_objects[i].indices_.size());
        }
      }
    } else {
      // add in the rest of the dimensions
      slice_len_ind = shape_[i];
      t = t.AppendDim(slice_len_ind);
    }
  }

  std::vector<std::vector<dsize_t>> indices_vector = IndexGenerator(converted_slice_objects);

  if (indices_vector.empty()) {
    return CreateEmpty(t, type_, out);
  }
  if (type_.IsNumeric()) {
    return SliceNumeric(out, indices_vector, t);
  } else {
    return SliceString(out, indices_vector, t);
  }
}

Status Tensor::SliceNumeric(std::shared_ptr<Tensor> *out, const std::vector<std::vector<dsize_t>> &indices,
                            const TensorShape &shape) {
  RETURN_UNEXPECTED_IF_NULL(out);
  RETURN_IF_NOT_OK(CreateEmpty(shape, type_, out));

  RETURN_UNEXPECTED_IF_NULL(out);
  (*out)->GetMutableBuffer();
  dsize_t out_index = 0;
  std::vector<dsize_t> dim_length = shape_.AsVector();
  dsize_t type_size = type_.SizeInBytes();
  std::vector<dsize_t> src_start = HandleNegIndices(indices[0], dim_length);
  dsize_t src_start_index;
  RETURN_IF_NOT_OK(shape_.ToFlatIndex(src_start, &src_start_index));

  uchar *dst_addr = (*out)->data_;
  dsize_t count = 1;

  // to handle partial slices
  dsize_t current_stride = shape_.Strides()[indices[0].size() - 1];
  auto indices_size = static_cast<dsize_t>(indices.size());
  for (dsize_t i = 0; i < indices_size; i++) {
    std::vector<dsize_t> cur_index = HandleNegIndices(indices[i], dim_length);
    if (i < indices_size - 1) {
      std::vector<dsize_t> next_index = HandleNegIndices(indices[i + 1], dim_length);
      dsize_t flat_idx_curr;
      dsize_t flat_idx_next;

      RETURN_IF_NOT_OK(shape_.ToFlatIndex(cur_index, &flat_idx_curr));
      RETURN_IF_NOT_OK(shape_.ToFlatIndex(next_index, &flat_idx_next));

      if (flat_idx_next == flat_idx_curr + current_stride) {
        count++;
        continue;
      }
    }

    int return_code = memcpy_s(dst_addr + out_index * type_size, (*out)->SizeInBytes(),
                               data_ + src_start_index * type_size, count * type_size * current_stride);
    CHECK_FAIL_RETURN_UNEXPECTED(return_code == EOK, "memcpy_s failed in SliceNumeric");
    out_index += count * current_stride;
    if (i < indices_size - 1) {
      src_start = HandleNegIndices(indices[i + 1], dim_length);  // next index
      RETURN_IF_NOT_OK(shape_.ToFlatIndex(src_start, &src_start_index));
    }
    count = 1;
  }
  return Status::OK();
}

Status Tensor::SliceString(std::shared_ptr<Tensor> *out, const std::vector<std::vector<dsize_t>> &indices,
                           const TensorShape &shape) {
  RETURN_UNEXPECTED_IF_NULL(out);
  std::vector<dsize_t> dim_length = shape_.AsVector();
  std::vector<std::string> strings;

  for (const std::vector<dsize_t> &index : indices) {
    std::vector<dsize_t> cur_index = HandleNegIndices(index, dim_length);
    dsize_t cur_flat_index;
    RETURN_IF_NOT_OK(shape_.ToFlatIndex(cur_index, &cur_flat_index));
    std::string_view sv;
    RETURN_IF_NOT_OK(GetItemAt(&sv, {cur_index}));
    strings.emplace_back(sv);
  }
  return CreateFromVector(strings, shape, type_, out);
}

Status Tensor::CreateFromMSTensor(const MSTensor &in, TensorPtr *out) {
  if (in.Data() == nullptr) {
    *out = nullptr;
    return Status::OK();
  }
  return Tensor::CreateFromMemory(TensorShape(in.Shape()), MSTypeToDEType(static_cast<TypeId>(in.DataType())),
                                  (const uchar *)(in.Data().get()), in.DataSize(), out);
}
}  // namespace dataset
}  // namespace mindspore
