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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_STORAGE_CLIENT_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_STORAGE_CLIENT_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "dataset/engine/data_schema.h"
#include "dataset/engine/datasetops/source/storage_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
// The Storage Client is the interface and base class that the StorageOp
// will use to perform any interactions with the storage layer.
// The different types of datasets will have different derived classes
// under that storage client super class.
class StorageClient {
 public:
  // Name: Constructor
  // Description:
  StorageClient(std::unique_ptr<DataSchema> schema,  // In: The schema for this storage client.
                StorageOp *store_op);                // In: The StorageOp that's using this client

  // Destructor
  virtual ~StorageClient() { storage_op_ = nullptr; }

  virtual Status Init() { return Status::OK(); }

  // Name: CreateStorageClient()
  // Description: A factory method to create the derived storage client.
  //              Every dataset has a required field for the dataset type in a config
  //              file.  This type will determine the child class to return for the
  //              type of storage client.
  static Status CreateStorageClient(StorageOp *store_op,  // In: A backpointer to the owning storage op for this client.
                                    std::string dataset_schema_path,              // In: The path to the dataset
                                    std::shared_ptr<StorageClient> *out_client);  // Out: the created storage client

  // Name: CreateStorageClient()
  // Description: A factory method to create the derived storage client.
  //              This creator is a user-override for the schema properties where
  //              the user has input the layout of the data (typically used in testcases)
  static Status CreateStorageClient(StorageOp *store_op,  // In: A backpointer to the owning cache for this client.
                                    DatasetType in_type,  // In: The type of dataset
                                    std::shared_ptr<StorageClient> *out_client);  // Out: the created storage client

  // Name: Print()
  // Description: A function that prints info about the StorageClient
  virtual void Print(std::ostream &out) const;  // In: The output stream to print to

  // Provide stream operator for displaying
  friend std::ostream &operator<<(std::ostream &out, const StorageClient &storage_client) {
    storage_client.Print(out);
    return out;
  }

  // Name: LoadDatasetLayout()
  // Description: There are 2 ways to define the properties of the data in the storage
  //              layer: LoadDatasetLayout() and AssignDatasetLayout().
  //              LoadDatasetLayout() will parse the json config file that comes with
  //              the dataset and internally populate row counts and schema.
  virtual Status LoadDatasetLayout();

  // Name: AssignDatasetLayout()
  // Description: There are 2 ways to define the properties of the data in the storage
  //              layer: LoadDatasetLayout() and AssignDatasetLayout().
  //              AssignDatasetLayout() will take input from the caller and assign that
  virtual Status AssignDatasetLayout(uint32_t num_rows,          // In: The number of rows in the dataset
                                     const DataSchema &schema);  // In: The schema for the dataset

  // Name: Reset()
  // Description: Resets any state info inside the client back to it's initialized
  //              state.
  virtual Status Reset() = 0;

  // Name: IsMoreData
  // Description: General routine to ask if more data exists in the storage side for
  //              a given buffer id.
  virtual bool IsMoreData(uint32_t id) { return true; }

  // Name: numRowsFromFile()
  // Description: Reads the schema json file to see if the optional numRows field has
  //              been set and returns it.
  Status numRowsFromFile(uint32_t &num_rows) const;

  // Get'r functions
  DataSchema *schema() const;

  uint32_t num_rows() const { return num_rows_in_dataset_; }

  // Name: rows_per_buffer()
  // Description: This default version simply gives you the count of the requested
  //              rows per buffer that the user defined in the storage op.
  //              However, if some condition down in the storage client layers
  //              could result in a buffer that has a different number of rows,
  //              then the derived class can override this method to provide their
  //              own implementation.
  virtual uint32_t rows_per_buffer() { return storage_op_->rows_per_buffer(); }

  // Description: Get the label classes num. Only manifest and Imagenet dataset support this parameter
  virtual uint32_t num_classes() const { return 0; }

 protected:
  std::unique_ptr<DataSchema> data_schema_;  // The schema for the data
  uint32_t num_rows_in_dataset_;             // The number of rows in the dataset
  StorageOp *storage_op_;                    // Back pointer to the owning storage operator.
  std::vector<std::string> col_names_;
  uint32_t num_classes_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_STORAGE_CLIENT_H_
