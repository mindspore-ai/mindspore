/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATA_BUFFER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATA_BUFFER_H_

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"

namespace mindspore {
namespace dataset {
/// \brief The DataBuffer class is a container of tensor data and is the unit of transmission between
///     connectors of dataset operators.  Inside the buffer, tensors are organized into a table-like format
///     where n TensorRows may consist of m tensors (columns).
class DataBuffer {
 public:
  // Buffer flags
  enum BufferFlags : uint32_t {
    kDeBFlagNone = 0,
    kDeBFlagEOF = 1,         // The buffer is an eof end-of-data msg
    kDeBFlagEOE = 1u << 1,   // The buffer is an eoe end-of-epoch msg
    kDeBFlagWait = 1u << 2,  // The buffer is an control signal for workers to suspend operations
    kDeBFlagQuit = 1u << 3   // The buffer is a control signal for workers to quit
  };

  // Name: Constructor #1
  // Description: This is the main constructor that is used for making a buffer
  DataBuffer(int32_t id, BufferFlags flags);

  /// \brief default destructor
  ~DataBuffer() = default;

  /// \brief A method for debug printing of the buffer
  /// \param[in/out] out The stream to write to
  /// \param[in] show_all A boolean to toggle between details and summary printing
  void Print(std::ostream &out, bool show_all) const;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const DataBuffer &cb) {
    cb.Print(out, false);
    return out;
  }

  // Convenience getter functions for flag checking
  bool eof() const { return (static_cast<uint32_t>(buffer_flags_) & static_cast<uint32_t>(kDeBFlagEOF)); }

  bool eoe() const { return (static_cast<uint32_t>(buffer_flags_) & static_cast<uint32_t>(kDeBFlagEOE)); }

  bool wait() const { return (static_cast<uint32_t>(buffer_flags_) & static_cast<uint32_t>(kDeBFlagWait)); }

  bool quit() const { return (static_cast<uint32_t>(buffer_flags_) & static_cast<uint32_t>(kDeBFlagQuit)); }

  // Simple getter funcs
  int32_t id() const { return buffer_id_; }

  void set_id(int32_t id) { buffer_id_ = id; }

  int32_t NumRows() const { return ((tensor_table_) ? tensor_table_->size() : 0); }

  int32_t NumCols() const {
    return (tensor_table_ == nullptr || tensor_table_->empty()) ? 0 : tensor_table_->at(0).size();
  }

  BufferFlags buffer_flags() const { return buffer_flags_; }

  // Remove me!! Callers should fetch rows via pop
  Status GetTensor(std::shared_ptr<Tensor> *, int32_t row_id, int32_t col_id) const;

  // Remove me!! Callers should drain rows via pop.
  Status GetRow(int32_t row_id, TensorRow *) const;

  // Get a row from the TensorTable
  Status PopRow(TensorRow *);

  Status SliceOff(int64_t number_of_rows);

  // Replacing mTensorTable, the unique_ptr assignment will release the old TensorTable.
  void set_tensor_table(std::unique_ptr<TensorQTable> new_table) { tensor_table_ = std::move(new_table); }

  void set_flag(BufferFlags in_flag) {
    buffer_flags_ = static_cast<BufferFlags>(static_cast<uint32_t>(buffer_flags_) | static_cast<uint32_t>(in_flag));
  }

  void Shuffle() {}  // does nothing right now.  possibly remove later

 protected:
  int32_t buffer_id_;                           // An id for the buffer.
  std::unique_ptr<TensorQTable> tensor_table_;  // A table (row major) of Tensors
  BufferFlags buffer_flags_;                    // bit mask for various buffer properties
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATA_BUFFER_H_
