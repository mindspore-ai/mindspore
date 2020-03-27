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
#include "dataset/engine/datasetops/source/tf_buffer.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "common/utils.h"
#include "utils/log_adapter.h"

#include "dataset/engine/datasetops/source/tf_client.h"
#include "dataset/core/data_type.h"
#include "dataset/engine/datasetops/source/storage_client.h"
#include "dataset/engine/data_schema.h"
#include "dataset/util/make_unique.h"

namespace mindspore {
namespace dataset {
// constructor
TFBuffer::TFBuffer(
  uint32_t id,                                           // In: The id for this buffer
  BufferFlags flags,                                     // In: The flags for this buffer
  const std::shared_ptr<StorageClient> &storage_client)  // In: Storage client that is related to this buffer type
    : DataBuffer(id, flags), storage_client_(storage_client) {
  // Initializing mColumnNameMap from the schema file
  const DataSchema *the_schema = storage_client_->schema();
  for (int32_t i = 0; i < the_schema->NumColumns(); ++i) {
    column_name_map_[the_schema->column(i).name()] = i;
  }
}

// destructor
TFBuffer::~TFBuffer() {}

// Name: print()
// Description: A function that prints info
void TFBuffer::Print(std::ostream &out,      // In: The output stream to print to
                     bool show_all) const {  // In: T/F if it should print everything
  out << "TFBuffer print\n";

  // Call base class printer
  DataBuffer::Print(out, show_all);
}

// Name: load()
// Description: populates the DataBuffer with data
//              Overrides base-class method.
Status TFBuffer::Load() {
  const DataSchema *the_schema = storage_client_->schema();
  uint32_t num_columns = the_schema->NumColumns();
  uint32_t num_rows_requested = storage_client_->rows_per_buffer();
  uint32_t remaining_rows = storage_client_->num_rows() > buffer_id_ * storage_client_->rows_per_buffer()
                              ? storage_client_->num_rows() - buffer_id_ * storage_client_->rows_per_buffer()
                              : 0;
  if (remaining_rows < num_rows_requested) {
    num_rows_requested = remaining_rows;
  }

  // Construct the Tensor table for this buffer.
  tensor_table_ = mindspore::make_unique<TensorQTable>();

  // At each position in the tensor table, instantiate the shared pointer to it's Tensor.
  uint32_t row = 0;
  while (row < num_rows_requested && (cur_reader_.peek() != EOF || storage_client_->IsMoreData(buffer_id_))) {
    TensorRow new_row;

    // Read the data from storage into a tf_file format
    dataengine::Example tf_file;
    RETURN_IF_NOT_OK(ParseSingleExample(&tf_file));
    for (uint32_t col = 0; col < num_columns; ++col) {
      std::shared_ptr<Tensor> new_t;
      const ColDescriptor current_col = the_schema->column(col);
      const dataengine::Features &example_features = tf_file.features();
      const google::protobuf::Map<std::string, dataengine::Feature> &feature_map = example_features.feature();
      const dataengine::Feature &column_values_list = feature_map.at(current_col.name());
      const dataengine::Feature::KindCase column_list_type = column_values_list.kind_case();
      RETURN_IF_NOT_OK(LoadFeature(column_list_type, column_values_list, current_col, &new_t));

      // Add the column to the current tensor row
      new_row.push_back(std::move(new_t));
    }

    // Add the new row of tensors to the end of our tensor table
    tensor_table_->push_back(new_row);
    row++;
  }
  cur_reader_.close();
  return Status::OK();
}

// Name: ParseSingleExample()
// Description: Drives the calls to TFClient for fetching the tf_file info from
//              the tf_file files.  Returns a single row of data from the tf_file
//              files.
Status TFBuffer::ParseSingleExample(dataengine::Example *ptr) {
  if (cur_reader_.peek() == EOF) {
    auto client = std::dynamic_pointer_cast<TFClient>(storage_client_);
    if (client == nullptr) {
      std::string errMsg = "Unexpected storage client type for TFBuffer";
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
    RETURN_IF_NOT_OK(client->NextFileInfo(buffer_id_, &cur_f_info_));
    cur_reader_.close();
    cur_reader_.open(cur_f_info_.fileName);
    // Seek to the offset
    (void)cur_reader_.seekg(static_cast<std::streamsize>(cur_f_info_.startOffset));
    MS_LOG(INFO) << "got new file " << cur_f_info_.fileName << ".";
  }

  // one record in tf_file looks like:
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  // read length
  if (cur_reader_.peek() == EOF) {
    MS_LOG(ERROR) << "ParseSingleExample failed";
  }

  dataengine::Example tf_file;
  try {
    uint64_t record_length = 0;
    (void)cur_reader_.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(uint64_t)));

    // ignore crc header
    (void)cur_reader_.ignore(static_cast<std::streamsize>(sizeof(uint32_t)));

    // read serialized Example
    std::string serialized_example;
    serialized_example.resize(record_length);
    (void)cur_reader_.read(&serialized_example[0], static_cast<std::streamsize>(record_length));

    // ignore crc footer
    (void)cur_reader_.ignore(static_cast<std::streamsize>(sizeof(uint32_t)));

    if (!tf_file.ParseFromString(serialized_example)) {
      std::string err_msg = "parse tf_file failed";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  } catch (const std::exception &err) {
    std::string err_msg = "Please check if the data file is complete!";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  *ptr = tf_file;
  return Status::OK();
}

// Name: LoadFeature()
// Description: Given the column type of the tf record and the values list,
//              constructs the tensor and returns it.
Status TFBuffer::LoadFeature(const dataengine::Feature::KindCase &column_list_type,
                             const dataengine::Feature &column_values_list, const ColDescriptor &current_col,
                             std::shared_ptr<Tensor> *out_tensor) {
  std::string element_str;                  // For staging data from protobuf deserialization
  std::unique_ptr<int64_t[]> int_array;     // For staging data from protobuf deserialization
  std::unique_ptr<float[]> float_array;     // For staging data from protobuf deserialization
  const unsigned char *data_ptr = nullptr;  // Generic pointer used for populating the Tensor
  // This variable will point into the above staging
  // variables.
  uint32_t num_elements = 0;  // Generic counter used for setting shape attributes

  // Depending on the type of data from the tf_file, we want to extract 2 things:
  // 1) A pointer to the data as a const unsigned char *
  // 2) The number of elements of the data
  // After those are determined, we can then build the tensor to represent this data.

  switch (column_list_type) {
    // CASE : TF record type: kBytesList
    case dataengine::Feature::KindCase::kBytesList: {
      RETURN_IF_NOT_OK(LoadBytesList(current_col, column_values_list, &element_str));

      // Get the const pointer representation of this data, and the number of elements
      // (number of bytes) for this tensor.
      data_ptr = reinterpret_cast<const unsigned char *>(common::SafeCStr(element_str));
      num_elements = element_str.length();
      break;
    }

      // CASE : TF record type: kFloatList
    case dataengine::Feature::KindCase::kFloatList: {
      RETURN_IF_NOT_OK(LoadFloatList(current_col, column_values_list, &num_elements, &float_array));

      data_ptr = reinterpret_cast<const unsigned char *>(float_array.get());
      break;
    }

      // CASE : TF record type: kInt64List
    case dataengine::Feature::KindCase::kInt64List: {
      RETURN_IF_NOT_OK(LoadIntList(current_col, column_values_list, &num_elements, &int_array));

      data_ptr = reinterpret_cast<const unsigned char *>(int_array.get());
      break;
    }
    case dataengine::Feature::KindCase::KIND_NOT_SET: {
      std::string errMsg = "tf_file column list type enum is KIND_NOT_SET";
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
    default: {
      std::string errMsg = "tf_file column list type enum does not match any known DE type";
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
  }

  // At this point we have a raw pointer to the data, and we have the number of elements.
  // Along with the tensor implementation type and the data type from the schema, we
  // enough info to construct the Tensor for it.
  TensorShape current_shape = TensorShape::CreateUnknownRankShape();
  RETURN_IF_NOT_OK(CreateTensorShapeForColumn(current_col, num_elements, &current_shape));

  // Now, create this tensor directly into the appropriate slot in our tensor
  // table.
  RETURN_IF_NOT_OK(
    Tensor::CreateTensor(out_tensor, current_col.tensorImpl(), current_shape, current_col.type(), data_ptr));

  return Status::OK();
}

Status TFBuffer::LoadBytesList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                               std::string *element_str) {
  // kBytesList can map to the following DE types ONLY!
  // DE_UINT8, DE_INT8
  // Must be single byte type for each element!
  if (current_col.type() != DataType::DE_UINT8 && current_col.type() != DataType::DE_INT8) {
    std::string err_msg = "Invalid datatype for Tensor at column: " + current_col.name();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  const dataengine::BytesList &bytes_list = column_values_list.bytes_list();

  // A bytesList is a special case where the entire list of data can be
  // deserialized into a single string. For example, it is not a list
  // of bytes, it is a list of strings, where each string represents
  // a list of bytes (this is different from the other cases like IntList etc)
  // As such, if there is more than one string in this list, that is invalid.
  if (bytes_list.value_size() > 1) {
    std::string err_msg = "Bytes list contains more than one element for column: " + current_col.name();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Extract the string that contains the bytes we need.  Position 0 is the only
  // valid string here.
  *element_str = bytes_list.value(0);

  return Status::OK();
}

Status TFBuffer::LoadFloatList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                               uint32_t *num_elements, std::unique_ptr<float[]> *float_array) {
  // KFloatList can only map to DE types:
  // DE_FLOAT32
  if (current_col.type() != DataType::DE_FLOAT32) {
    std::string err_msg = "Invalid datatype for Tensor at column: " + current_col.name();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  const dataengine::FloatList &float_list = column_values_list.float_list();

  // Identify how many values we have and then create a local array of these
  // to deserialize into
  *num_elements = float_list.value_size();
  *float_array = mindspore::make_unique<float[]>(*num_elements);
  for (int i = 0; i < float_list.value_size(); i++) {
    (*float_array)[i] = float_list.value(i);
  }

  return Status::OK();
}

Status TFBuffer::LoadIntList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                             uint32_t *num_elements, std::unique_ptr<int64_t[]> *int_array) {
  // KInt64List can only map to DE types:
  // DE_UINT64, DE_INT64, DE_UINT32, DE_INT32, DE_UINT16, DE_INT16, DE_UINT8, DE_INT8
  if (!(current_col.type().IsInt())) {
    std::string err_msg = "Invalid datatype/rank for column label in TFBuffer.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  const dataengine::Int64List &int64_list = column_values_list.int64_list();

  // Identify how many values we have and then create a local array of these
  // to deserialize into
  *num_elements = int64_list.value_size();
  *int_array = mindspore::make_unique<int64_t[]>(*num_elements);
  for (int i = 0; i < int64_list.value_size(); i++) {
    (*int_array)[i] = int64_list.value(i);
  }

  return Status::OK();
}

Status TFBuffer::CreateTensorShapeForColumn(const ColDescriptor &current_col, uint32_t num_elements,
                                            TensorShape *current_shape) {
  // If the shape is assigned by user, we have an assumption that the data is
  // already in the appropriate format that we can copy into the Tensor as-is.
  if (current_col.hasShape()) {
    *current_shape = current_col.shape();
  } else if (current_col.rank() == 1) {
    // If shape was not given, then we support 2 possible shapes.
    // 1) It's a scalar (rank 0), in which case the shape is empty but we need to flag
    //    it as a scalar value (empty shape but has a single value)
    // 2) It's a rank 1 shape, and the dimension value for that single dimension will
    //    be comprised of the entire bytes-size of the input data.
    *current_shape = TensorShape({num_elements});
  } else if (current_col.rank() == 0) {
    // Make this shape into a single value scalar.
    *current_shape = TensorShape::CreateScalar();
  } else if (current_col.rank() > 1) {
    // All other ranks, except for 0, are invalid because we cannot guess
    // what the shape will be.  For example, if we have rank 3 and 12 bytes
    // of data, is it shape {2,2,3} or is it {2,6,1}.  We can't guess at
    // the shape dimensions.
    const std::string kErrMsg = "Invalid rank (rank>1) for dynamic shape construction. Specify shape in schema.";
    RETURN_STATUS_UNEXPECTED(kErrMsg);
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
