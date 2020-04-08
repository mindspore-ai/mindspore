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

#define MAX_INTEGER_INT32 2147483647

#include <iostream>
#include <memory>
#include <utility>
#include <nlohmann/json.hpp>
#include "dataset/core/constants.h"
#include "dataset/engine/datasetops/source/storage_client.h"
#include "dataset/engine/datasetops/source/storage_op.h"
#include "dataset/engine/datasetops/source/tf_client.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Name: Constructor
// Description:
StorageClient::StorageClient(std::unique_ptr<DataSchema> schema,  // In: The schema for this storage client.
                             StorageOp *store_op)                 // In: The StorageOp that's using this client
    : data_schema_(std::move(schema)), num_rows_in_dataset_(0), storage_op_(store_op), num_classes_(0) {}

// Name: Print()
// Description: A function that prints info about the StorageClient
// In: The output stream to print to
void StorageClient::Print(std::ostream &out) const {
  // not much to show here folks!
  // out << "Storage client:\n";
}

// This is a local-only static function to drive the switch statement for creating
// the storage client (not a static member function)
static Status CreateStorageClientSwitch(
  std::unique_ptr<DataSchema> schema,            // In: The schema  to set into the client
  StorageOp *store_op,                           // In: The StorageOp we are operating on
  std::shared_ptr<StorageClient> *out_client) {  // Out: the created storage client
  switch (schema->dataset_type()) {
    case DatasetType::kArrow: {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                    "Storage client not implemented yet for arrow dataset type.");
    }
    case DatasetType::kTf: {
      // Construct the derived class TFClient, stored as base class StorageClient
      store_op->set_rows_per_buffer(32);
      *out_client = std::make_unique<TFClient>(std::move(schema), store_op);
      break;
    }
    case DatasetType::kUnknown:
    default: {
      RETURN_STATUS_UNEXPECTED("Invalid dataset type.");
    }
  }
  if (*out_client) {
    RETURN_IF_NOT_OK((*out_client)->Init());
  }
  return Status::OK();
}

// Name: CreateStorageClient()
// Description: A factory method to create the derived storage client.
//              Every dataset has a required field for the dataset type in a config
//              file.  This type will determine the child class to return for the
//              type of storage client.  It also creates the schema and sticks it
//              into the cache.
Status StorageClient::CreateStorageClient(
  StorageOp *store_op,                           // In: A backpointer to the owning cache for this client.
  std::string dataset_schema_path,               // In: The path to the schema
  std::shared_ptr<StorageClient> *out_client) {  // Out: the created storage client
  // Make a new schema first.  This only assigns the dataset type.  It does not
  // create the columns yet.
  auto new_schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(new_schema->LoadDatasetType(dataset_schema_path));
  RETURN_IF_NOT_OK(CreateStorageClientSwitch(std::move(new_schema), store_op, out_client));
  return Status::OK();
}

// Name: CreateStorageClient()
// Description: A factory method to create the derived storage client.
//              This creator is a user-override for the schema properties where
//              the user has input the layout of the data (typically used in testcases)
Status StorageClient::CreateStorageClient(
  StorageOp *store_op,                           // In: A backpointer to the owning cache for this client.
  DatasetType in_type,                           // In: The type of dataset
  std::shared_ptr<StorageClient> *out_client) {  // Out: the created storage client
  // The dataset type is passed in by the user.  Create an empty schema with only
  // only the dataset type filled in and then create the client with it.
  auto new_schema = std::make_unique<DataSchema>();
  new_schema->set_dataset_type(in_type);
  RETURN_IF_NOT_OK(CreateStorageClientSwitch(std::move(new_schema), store_op, out_client));
  return Status::OK();
}

// Name: LoadDatasetLayout()
// Description: There are 2 ways to define the properties of the data in the storage
//              layer: LoadDatasetLayout() and AssignDatasetLayout().
//              LoadDatasetLayout() will parse the json config file that comes with
//              the dataset.
Status StorageClient::LoadDatasetLayout() {
  // Access the json file to populate our schema, assume the json file is accessible
  // locally.
  RETURN_IF_NOT_OK(data_schema_->LoadSchemaFile(storage_op_->schema_file(), storage_op_->columns_to_load()));

  // The number of rows in the schema file is an optional config.  For example,
  // maybe the derived storage client will know how to determine the total number
  // of rows a different way rather than having it in the schema config json file.
  // Thus, mNumRowsInDataset can still be zero and force the derived class override
  // to determine it another way.
  uint32_t num_rows = 0;
  RETURN_IF_NOT_OK(this->numRowsFromFile(num_rows));
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows <= MAX_INTEGER_INT32, "numRows exceeds the boundary numRows>2147483647");
  if (num_rows_in_dataset_ == 0 || num_rows < num_rows_in_dataset_) {
    num_rows_in_dataset_ = num_rows;
  }

  return Status::OK();
}

// Name: AssignDatasetLayout()
// Description: There are 2 ways to define the properties of the data in the storage
//              layer: LoadDatasetLayout() and AssignDatasetLayout().
//              AssignDatasetLayout() will take input from the caller and assign that
//              info into the storage client.
Status StorageClient::AssignDatasetLayout(uint32_t num_rows,           // In: The number of rows in the dataset
                                          const DataSchema &schema) {  // In: The schema for the dataset
  // Since this is just an assignment into the storage client, you probably won't need
  // to override this one in a derived class.  First some sanity checks
  CHECK_FAIL_RETURN_UNEXPECTED(data_schema_->dataset_type() == schema.dataset_type(),
                               "Assigning a schema into StorageClient with mismatched dataset types!");
  CHECK_FAIL_RETURN_UNEXPECTED(data_schema_->NumColumns() == 0,
                               "Assigning a schema into StorageClient that already has non-empty schema!");

  // The current schema was just an empty one with only the dataset field populated.
  // Let's copy construct a new one that will be a copy of the input schema (releasing the old
  // one) and then set the number of rows that the user requested.
  data_schema_ = std::make_unique<DataSchema>(schema);
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows <= MAX_INTEGER_INT32, "numRows exceeds the boundary numRows>2147483647");
  num_rows_in_dataset_ = num_rows;

  return Status::OK();
}

// Name: numRowsFromFile()
// Description: Reads the schema json file to see if the optional numRows field has
//              been set and returns it.
Status StorageClient::numRowsFromFile(uint32_t &num_rows) const {
  std::string schemaFile = storage_op_->schema_file();
  try {
    std::ifstream in(schemaFile);
    nlohmann::json js;
    in >> js;
    num_rows = js.value("numRows", 0);
    if (num_rows == 0) {
      std::string err_msg =
        "Storage client has not properly done dataset "
        "handshake to initialize schema and number of rows.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  // Catch any exception and rethrow it as our own
  catch (const std::exception &err) {
    std::ostringstream ss;
    ss << "Schema file failed to load:\n" << err.what();
    std::string err_msg = ss.str();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

// Get'r function
DataSchema *StorageClient::schema() const { return data_schema_.get(); }
}  // namespace dataset
}  // namespace mindspore
