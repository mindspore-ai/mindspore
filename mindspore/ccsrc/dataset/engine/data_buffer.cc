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
#include "dataset/engine/data_buffer.h"
#include "dataset/util/allocator.h"
#include "dataset/core/global_context.h"
#include "dataset/core/tensor.h"
#include "dataset/engine/datasetops/source/storage_client.h"
#include "dataset/engine/datasetops/source/tf_buffer.h"

namespace mindspore {
namespace dataset {
// Name: Constructor #1
// Description: This is the main constructor that is used for making a buffer
DataBuffer::DataBuffer(int32_t id, BufferFlags flags) : buffer_id_(id), tensor_table_(nullptr), buffer_flags_(flags) {}

// Name: CreateDataBuffer()
// Description: A static factory method to create the appropriate type of derived class
//              buffer.  Returns the base class reference for DataBuffer.
Status DataBuffer::CreateDataBuffer(
  int32_t id,                                     // In: The id for the new buffer
  std::shared_ptr<StorageClient> storage_client,  // In: The storage client that is related to this buffer type
  std::unique_ptr<DataBuffer> *ptr) {
  std::unique_ptr<DataBuffer> new_data_buffer;
  try {
    DatasetType ds_type = storage_client->schema()->dataset_type();
    switch (ds_type) {
      case DatasetType::kTf: {
        // This type of buffer is for TF record data.
        // Allocate derived class version for a TF buffers
        new_data_buffer = mindspore::make_unique<TFBuffer>(id, kDeBFlagNone, storage_client);
        break;
      }
      default: {
        std::string errMsg("Invalid buffer type");
        RETURN_STATUS_UNEXPECTED(errMsg);
      }
    }
  } catch (std::bad_alloc &e) {
    return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__, e.what());
  } catch (std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  *ptr = std::move(new_data_buffer);
  return Status::OK();
}

// Name: print()
// Description: A function that prints info about the DataBuffer (base class version)
void DataBuffer::Print(std::ostream &out,      // In: The output stream to print to
                       bool show_all) const {  // In: T/F if it should show everything
  out << "bufferId: " << buffer_id_ << "\nflags: " << std::hex << buffer_flags_ << std::dec << "\n";

  // If the column counts are set then it means that data has been set into
  // the tensor table.  Display the tensor table here.
  if (this->NumCols() > 0) {
    out << "Tensor table:\n";
    for (int32_t row = 0; row < DataBuffer::NumRows(); ++row) {
      out << "Row #   : " << row << "\n";
      TensorRow currRow = (*tensor_table_)[row];
      for (int32_t col = 0; col < this->NumCols(); ++col) {
        out << "Column #: " << col << "\n";  // Should add the column name here as well?
        // Call the tensor display
        out << *(currRow[col]) << "\n";
      }
    }
  }
}

Status DataBuffer::Load() {
  std::string err_msg = "Base class load called, but it does not have an implementation!";
  RETURN_STATUS_UNEXPECTED(err_msg);
}

// Remove me!! Callers should fetch rows via pop
Status DataBuffer::GetTensor(std::shared_ptr<Tensor> *ptr, int32_t row_id, int32_t col_id) const {
  if (row_id < tensor_table_->size() && col_id < tensor_table_->at(row_id).size()) {
    *ptr = (tensor_table_->at(row_id)).at(col_id);
  } else {
    std::string err_msg =
      "indices for mTensorTable out of range: (" + std::to_string(row_id) + "," + std::to_string(col_id) + ").";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

// Remove me!! Callers should fetch rows via pop
Status DataBuffer::GetRow(int32_t row_id, TensorRow *ptr) const {
  if (row_id < tensor_table_->size()) {
    *ptr = tensor_table_->at(row_id);
  } else {
    std::string err_msg = "rowId for mTensorTable out of range: " + std::to_string(row_id);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

Status DataBuffer::PopRow(TensorRow *ptr) {
  if (tensor_table_ && !tensor_table_->empty()) {
    *ptr = std::move(tensor_table_->front());
    tensor_table_->pop_front();
  }

  return Status::OK();
}

Status DataBuffer::SliceOff(int64_t number_of_rows) {
  while (number_of_rows > 0) {
    tensor_table_->pop_back();
    number_of_rows--;
  }

  return Status::OK();
}

// Destructor
DataBuffer::~DataBuffer() {}
}  // namespace dataset
}  // namespace mindspore
