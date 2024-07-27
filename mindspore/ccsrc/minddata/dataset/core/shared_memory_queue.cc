/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/shared_memory_queue.h"

#include <algorithm>
#include <memory>
#include <string>

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/util/status.h"
#include "pybind11/pytypes.h"

namespace mindspore {
namespace dataset {
#if !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID)
SharedMemoryQueue::SharedMemoryQueue(const key_t &key)
    : key_(key), shm_id_(-1), shm_addr_(nullptr), shm_size_(0), release_flag_(true) {}

SharedMemoryQueue::~SharedMemoryQueue() {
  if (release_flag_ && shm_id_ != -1 && shmget(key_, 0, kShmPermission) == shm_id_) {
    auto rc = ReleaseCurrentShm();
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc.ToString();
    }
  }
}

Status SharedMemoryQueue::FromTensorRow(const TensorRow &in_row) {
  // calculate the size of the shared memory
  uint64_t request_shm_size = 0;
  RETURN_IF_NOT_OK(CalculateShmSize(in_row, &request_shm_size));

  // update the shm
  RETURN_IF_NOT_OK(UpdateShmBySize(request_shm_size));

  // Serialize the TensorRow
  RETURN_IF_NOT_OK(Serialize(in_row));

  return Status::OK();
}

Status SharedMemoryQueue::ToTensorRow(TensorRow *out_row, const int &shm_id, const uint64_t &shm_size) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  MS_LOG(DEBUG) << "In ToTensorRow, shm_id: " << shm_id << ", shm_size: " << shm_size;
  if (shm_id_ != shm_id) {
    // detach the old shm
    if (shm_id_ != -1) {
      std::stringstream ss;
      ss << shm_addr_;
      if (shmdt(shm_addr_) == -1) {
        RETURN_STATUS_UNEXPECTED("shmdt shm_addr: " + ss.str() + " error. Errno: " + std::to_string(errno));
      }
      shm_size_ = 0;
      shm_addr_ = nullptr;
      shm_id_ = -1;
    }

    // attach the new shm
    shm_id_ = shm_id;
    shm_addr_ = shmat(shm_id_, nullptr, 0);
    if (shm_addr_ == nullptr) {
      RETURN_STATUS_UNEXPECTED("shmat shm_id: " + std::to_string(shm_id_) + " error. Errno: " + std::to_string(errno));
    }
    shm_size_ = shm_size;
  }
  MS_LOG(DEBUG) << "Before Deserialize, shm_id: " << shm_id_ << ", shm_size: " << shm_size_;
  RETURN_IF_NOT_OK(Deserialize(out_row));
  return Status::OK();
}

Status SharedMemoryQueue::ToTensorRowWithNoCopy(TensorRow *out_row) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  MS_LOG(EXCEPTION) << "Not supported yet.";
  return Status::OK();
}

void SharedMemoryQueue::SetReleaseFlag(bool flag) { release_flag_ = flag; }

key_t SharedMemoryQueue::GetKey() { return key_; }

int SharedMemoryQueue::GetShmID() { return shm_id_; }

uint64_t SharedMemoryQueue::GetShmSize() { return shm_size_; }

Status SharedMemoryQueue::ReleaseCurrentShm() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  if (shm_addr_ != nullptr) {
    // detach the shm
    std::stringstream ss_addr;
    ss_addr << shm_addr_;
    if (shmdt(shm_addr_) == -1) {
      RETURN_STATUS_UNEXPECTED("shmdt shm_addr: " + ss_addr.str() + " error. Errno: " + std::to_string(errno));
    }
    MS_LOG(INFO) << "Detach shared memory: " << shm_addr_ << ", size: " << shm_size_ << ", thread id: " << ss.str()
                 << " successfully.";
    shm_size_ = 0;
    shm_addr_ = nullptr;
  }

  // del the shm
  if (shm_id_ != -1) {
    if (shmctl(shm_id_, IPC_RMID, NULL) == -1) {
      RETURN_STATUS_UNEXPECTED("shmctl shm_id: " + std::to_string(shm_id_) + " error. Errno: " + std::to_string(errno));
    }
    MS_LOG(INFO) << "Delete shared memory with shm_id: " << std::to_string(shm_id_) << ", thread id: " << ss.str()
                 << " successfully.";
    shm_id_ = -1;
  }

  return Status::OK();
}

Status SharedMemoryQueue::CreateShmBySize(const uint64_t &size) {
  if (shm_addr_ != nullptr) {
    RETURN_IF_NOT_OK(ReleaseCurrentShm());
  }

  shm_id_ = shmget(key_, size, IPC_CREAT | IPC_EXCL | kShmPermission);
  if (shm_id_ == -1) {
    RETURN_STATUS_UNEXPECTED("shmget key: " + std::to_string(key_) + " error. Errno: " + std::to_string(errno));
  }

  shm_addr_ = shmat(shm_id_, nullptr, 0);
  if (shm_addr_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("shmat shm_id: " + std::to_string(shm_id_) + " error. Errno: " + std::to_string(errno));
  }
  shm_size_ = size;

  std::stringstream ss;
  ss << std::this_thread::get_id();
  MS_LOG(INFO) << "Create new shared memory with shm_id: " << std::to_string(shm_id_)
               << ", size: " << std::to_string(shm_size_) << ", thread id: " << ss.str() << " successfully.";
  return Status::OK();
}

Status SharedMemoryQueue::UpdateShmBySize(const uint64_t &size) {
  // no need to create new shm
  if (shm_size_ >= size) {
    return Status::OK();
  }

  // release the old shm
  RETURN_IF_NOT_OK(ReleaseCurrentShm());

  // create new shm
  RETURN_IF_NOT_OK(CreateShmBySize(size));

  return Status::OK();
}

Status SharedMemoryQueue::CalculateShmSize(const TensorRow &in_row, uint64_t *size) {
  RETURN_UNEXPECTED_IF_NULL(size);
  *size = 0;
  *size += kTensorRowType;          // the type of the TensorRow
  *size += kTensorSizeInTensorRow;  // the size of tensor in the TensorRow
  for (auto &item : in_row) {
    *size += kTensorType;  // the type of the tensor
    if (!item->type().IsPython()) {
      *size += kTensorShapeDims + kTensorShapeType * item->shape().Size();  // the size of shape bytes
      *size += kTensorDataType;                                             // the data type of the tensor
      *size += kTensorDataLen + item->SizeInBytes();                        // the len of data and data
    } else {
      *size += kTensorDataLen + item->SizeInBytes();  // the len of data and data
    }
  }
  return Status::OK();
}

Status SharedMemoryQueue::Serialize(const TensorRow &in_row) {
  CHECK_FAIL_RETURN_UNEXPECTED(!in_row.wait(), "Cannot serialize tensor row with wait flag.");
  CHECK_FAIL_RETURN_UNEXPECTED(!in_row.quit(), "Cannot serialize tensor row with quit flag.");
  CHECK_FAIL_RETURN_UNEXPECTED(!in_row.skip(), "Cannot serialize tensor row with skip flag.");
  CHECK_FAIL_RETURN_UNEXPECTED(!in_row.error(), "Cannot serialize tensor row with error flag.");
  uint64_t offset = 0;

  uint32_t tensor_row_flag = in_row.Flags();
  int ret_code =
    memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, kTensorRowType, &tensor_row_flag, kTensorRowType);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the flag of TensorRow failed. err code: " + std::to_string(ret_code));
  offset += kTensorRowType;

  uint32_t tensor_row_size = in_row.size();
  ret_code = memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, kTensorSizeInTensorRow, &tensor_row_size,
                      kTensorSizeInTensorRow);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the size of TensorRow failed. err code: " + std::to_string(ret_code));
  offset += kTensorSizeInTensorRow;

  for (auto &item : in_row) {
    uint32_t tensor_inner_type = kNormalCTensor;
    if (item->type().IsPython()) {
      tensor_inner_type = kPythonDictObject;
    }
    ret_code = memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, kTensorType, &tensor_inner_type, kTensorType);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                 "memcpy_s the type of Tensor failed. err code: " + std::to_string(ret_code));
    offset += kTensorType;

    if (tensor_inner_type == kNormalCTensor) {  // tensor is normal data_ or python_array_
      // copy the TensorShape
      auto tensor_shape_size = item->shape().Size();
      ret_code =
        memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, kTensorShapeDims, &tensor_shape_size, kTensorShapeDims);
      CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                   "memcpy_s the size of Tensor's shape failed. err code: " + std::to_string(ret_code));
      offset += kTensorShapeDims;

      for (auto &shape : item->shape().AsVector()) {
        ret_code = memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, kTensorShapeType, &shape, kTensorShapeType);
        CHECK_FAIL_RETURN_UNEXPECTED(
          ret_code == EOK, "memcpy_s the shape of Tensor's shape[i] failed. err code: " + std::to_string(ret_code));
        offset += kTensorShapeType;
      }

      // copy the DataType
      auto tensor_data_type = (uint32_t)item->type().value();
      ret_code =
        memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, kTensorDataType, &tensor_data_type, kTensorDataType);
      CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                   "memcpy_s the data type of Tensor failed. err code: " + std::to_string(ret_code));
      offset += kTensorDataType;
    }

    // copy the data
    int64_t data_len = item->SizeInBytes();
    ret_code = memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, kTensorDataLen, &data_len, kTensorDataLen);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                 "memcpy_s the size of Tensor's data failed. err code: " + std::to_string(ret_code));
    offset += kTensorDataLen;

    if (data_len != 0) {
      if (data_len < SECUREC_MEM_MAX_LEN) {
        ret_code = memcpy_s(reinterpret_cast<char *>(shm_addr_) + offset, data_len,
                            reinterpret_cast<char *>(item->GetMutableBuffer()), data_len);
        CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                     "memcpy_s the data type of Tensor failed. err code: " + std::to_string(ret_code));
      } else {
        auto ret_memcpy = memcpy(reinterpret_cast<char *>(shm_addr_) + offset,
                                 reinterpret_cast<char *>(item->GetMutableBuffer()), data_len);
        CHECK_FAIL_RETURN_UNEXPECTED(ret_memcpy == item->GetMutableBuffer(), "memcpy the data type of Tensor failed.");
      }
      offset += static_cast<uint64_t>(data_len);
    }
    if (tensor_inner_type == kNormalCTensor) {
      // normal array
      MS_LOG(DEBUG) << "Row shape: " << item->shape() << ", row type: " << item->type()
                    << ", row data len: " << data_len << ", row inner type: " << tensor_inner_type;
    } else {
      // python dict
      MS_LOG(DEBUG) << "Row data len: " << data_len << ", row inner type: " << tensor_inner_type;
    }
  }

  return Status::OK();
}

Status SharedMemoryQueue::Deserialize(TensorRow *in_row) {
  RETURN_UNEXPECTED_IF_NULL(shm_addr_);
  RETURN_UNEXPECTED_IF_NULL(in_row);
  uint64_t offset = 0;

  uint32_t *tensor_row_flag = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(shm_addr_) + offset);
  if (*tensor_row_flag != static_cast<uint32_t>(TensorRow::TensorRowFlags::kFlagNone)) {
    *in_row = TensorRow(TensorRow::TensorRowFlags(*tensor_row_flag));
    return Status::OK();
  }
  in_row->reset();
  offset += kTensorRowType;

  uint32_t *row_size = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(shm_addr_) + offset);
  offset += kTensorSizeInTensorRow;

  for (int i = 0; i < *row_size; i++) {
    uint32_t *tensor_inner_type = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(shm_addr_) + offset);
    offset += kTensorType;

    if (*tensor_inner_type == kNormalCTensor) {  // tensor type is normal data_ / python_array_
      // copy the TensorShape
      uint32_t *shape_dims = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(shm_addr_) + offset);
      offset += kTensorShapeDims;

      std::vector<dsize_t> shape;
      std::transform(
        reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(shm_addr_) + offset),
        reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(shm_addr_) + offset + (*shape_dims) * kTensorShapeType),
        std::back_inserter(shape), [](int x) { return x; });
      offset += (*shape_dims) * kTensorShapeType;
      TensorShape tensor_shape(shape);

      // copy the DataType
      uint32_t *data_type = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(shm_addr_) + offset);
      offset += kTensorDataType;
      DataType tensor_type(DataType::Type((uint8_t)(*data_type)));

      // copy the data
      uint64_t *data_len = reinterpret_cast<uint64_t *>(reinterpret_cast<char *>(shm_addr_) + offset);
      offset += kTensorDataLen;

      std::shared_ptr<Tensor> output_tensor;
      uint64_t update_data_len = *data_len;
      if (*data_len == 0 && tensor_type.IsString()) {
        update_data_len = (tensor_shape.NumOfElements() + 1) * kOffsetSize + tensor_shape.NumOfElements();
      }
      RETURN_IF_NOT_OK(Tensor::CreateFromMemory(tensor_shape, tensor_type,
                                                reinterpret_cast<unsigned char *>(shm_addr_) + offset, update_data_len,
                                                &output_tensor));
      in_row->push_back(output_tensor);
      offset += *data_len;
      MS_LOG(DEBUG) << "Row shape: " << tensor_shape << ", row type: " << tensor_type << ", row data len: " << *data_len
                    << ", row inner type: " << *tensor_inner_type;
    } else {  // tensor type is python_dict_
      CHECK_FAIL_RETURN_UNEXPECTED(*tensor_inner_type == kPythonDictObject,
                                   "The type of tensor is not data_ / python_array_ or python_dict_.");

      // copy the data
      uint64_t *data_len = reinterpret_cast<uint64_t *>(reinterpret_cast<char *>(shm_addr_) + offset);
      offset += kTensorDataLen;

      // deserialize bytes to python object
      std::string str(reinterpret_cast<char *>(shm_addr_) + offset, *data_len);
      {
        py::gil_scoped_acquire gil_acquire;
        py::object shared_dict = py::module::import("pickle").attr("loads")(py::bytes(str));

        // construct tensor
        std::vector<dsize_t> shape{};
        DataType type = DataType(DataType::DE_PYTHON);
        std::shared_ptr<Tensor> output_tensor = std::make_shared<Tensor>(TensorShape({0}), type);
        RETURN_IF_NOT_OK(Tensor::CreateFromPythonObject(shared_dict, &output_tensor));
        in_row->push_back(output_tensor);
        offset += *data_len;
        MS_LOG(DEBUG) << "Row data len: " << *data_len << ", row inner type: " << *tensor_inner_type;
      }
    }
  }

  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
