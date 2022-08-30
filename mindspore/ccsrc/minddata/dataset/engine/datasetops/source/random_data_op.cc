/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/random_data_op.h"

#include <algorithm>
#include <iomanip>
#include <random>
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

namespace mindspore {
namespace dataset {
// Constructor for RandomDataOp
RandomDataOp::RandomDataOp(int32_t num_workers, int32_t op_connector_size, int64_t total_rows,
                           std::unique_ptr<DataSchema> data_schema)
    : MappableLeafOp(num_workers, op_connector_size, std::make_shared<SequentialSamplerRT>(0, 0)),
      total_rows_(total_rows),
      data_schema_(std::move(data_schema)) {
  rand_gen_.seed(GetSeed());  // seed the random generator
  // If total rows was not given, then randomly pick a number
  if (total_rows_ == 0) {
    total_rows_ = GenRandomInt(1, kMaxTotalRows);
  }
  // If the user did not provide a schema, then we will ask the op to generate a pseudo-random schema.
  // See details of generateSchema function to learn what type of schema it will create.
  if (data_schema_ == nullptr) {
    GenerateSchema();
  }
}

// A print method typically used for debugging
void RandomDataOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [total rows: " << num_rows_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nTotal_rows: " << num_rows_ << " \nSchema:\n" << *data_schema_ << "\n\n";
  }
}

// Helper function to produce a default/random schema if one didn't exist
void RandomDataOp::GenerateSchema() {
  // To randomly create a schema, we need to choose:
  // a) how many columns
  // b) the type of each column
  // c) the shape of each column (number of dimensions i.e. rank)
  // d) the shape of each column (dimension values)
  data_schema_ = std::make_unique<DataSchema>();
  std::unique_ptr<TensorShape> new_shape;
  std::unique_ptr<ColDescriptor> new_col;

  // Loop over the number of chosen columns
  int32_t numColumns = GenRandomInt(1, kMaxNumColumns);
  for (int32_t i = 0; i < numColumns; i++) {
    // For each column:
    // - choose a datatype
    // - generate a shape that randomly chooses the number of dimensions and the dimension values.
    auto newType = static_cast<DataType::Type>(GenRandomInt(1, DataType::DE_STRING - 1));
    int32_t rank = GenRandomInt(1, kMaxRank);
    std::vector<dsize_t> dims;
    for (int32_t d = 0; d < rank; d++) {
      // 0 is not a valid dimension value.  however, we can support "*" or unknown, so map the random
      // 0 value to the unknown attribute if 0 is chosen
      auto dim_value = static_cast<dsize_t>(GenRandomInt(0, kMaxDimValue));
      if (dim_value == 0) {
        dim_value = TensorShape::kDimUnknown;
      }
      dims.push_back(dim_value);
    }
    new_shape = std::make_unique<TensorShape>(dims);

    // Create the column descriptor
    std::string col_name = "c" + std::to_string(i);
    new_col =
      std::make_unique<ColDescriptor>(col_name, DataType(newType), TensorImpl::kFlexible, rank, new_shape.get());

    Status rc = data_schema_->AddColumn(*new_col);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "[Internal ERROR] Failed to generate a schema. Message:" << rc;
    }
  }
}

// A helper function to create random data for the row
Status RandomDataOp::CreateRandomRow(TensorRow *new_row) {
  if (new_row == nullptr) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Missing tensor row output.");
  }

  // Create a tensor for each column, then add the tensor to the row
  for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    const ColDescriptor current_col = data_schema_->Column(i);
    std::vector<dsize_t> current_shape = current_col.Shape().AsVector();
    std::unique_ptr<TensorShape> new_shape = nullptr;
    std::unique_ptr<unsigned char[]> buf = nullptr;
    std::shared_ptr<Tensor> new_tensor = nullptr;

    // We need to resolve the shape to fill in any unknown dimensions with random
    // values, then use that as our shape for this tensor.
    for (int j = 0; j < current_shape.size(); ++j) {
      if (current_shape[j] == TensorShape::kDimUnknown) {
        current_shape[j] = static_cast<dsize_t>(GenRandomInt(1, kMaxDimValue));
      }
    }

    new_shape = std::make_unique<TensorShape>(current_shape);
    int64_t size_in_bytes = new_shape->NumOfElements() * current_col.Type().SizeInBytes();

    // Generate a random byte of data.  This may cause some funny data for things like doubles,floats, bools
    // however the random data op is not too concerned about the physical data itself.
    std::uniform_int_distribution<uint32_t> uniDist(0, UINT8_MAX);
    uint8_t random_byte = static_cast<uint8_t>(uniDist(rand_gen_));

    // Now, create a chunk of memory for the entire tensor and copy this byte in repeatedly.
    buf = std::make_unique<unsigned char[]>(size_in_bytes);
    int ret_code = memset_s(buf.get(), size_in_bytes, random_byte, size_in_bytes);
    if (ret_code != EOK) {
      std::string error_msg = "RandomData: failed to set random data, ";
      if (ret_code == ERANGE) {
        RETURN_STATUS_UNEXPECTED(error_msg + "memory size of total data can not be zero or exceed " +
                                 std::to_string(SECUREC_MEM_MAX_LEN) + ", but got: " + std::to_string(size_in_bytes));
      } else {
        RETURN_STATUS_UNEXPECTED("memset_s method failed with errno_t: " + std::to_string(ret_code));
      }
    }

    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(*new_shape, current_col.Type(), buf.get(), &new_tensor));

    // Add this tensor to the tensor row for output
    (*new_row).push_back(std::move(new_tensor));
  }
  return Status::OK();
}

Status RandomDataOp::ComputeColMap() {
  // Extract the column name mapping from the schema and save it in the class.
  if (column_name_id_map_.empty()) {
    RETURN_IF_NOT_OK(data_schema_->GetColumnNameMap(&(column_name_id_map_)));
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status RandomDataOp::LoadTensorRow(row_id_type row_id, TensorRow *row) {
  CHECK_FAIL_RETURN_UNEXPECTED(row_id < total_rows_, "Wrong index.");
  for (const auto &tensor : rows_[static_cast<size_t>(row_id)]) {
    TensorPtr new_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateFromTensor(tensor, &new_tensor));
    row->emplace_back(new_tensor);
  }
  return Status::OK();
}

Status RandomDataOp::PrepareData() {
  for (int64_t i = 0; i < total_rows_; i++) {
    TensorRow row;
    RETURN_IF_NOT_OK(CreateRandomRow(&row));
    rows_.emplace_back(row);
  }
  num_rows_ = total_rows_;
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
