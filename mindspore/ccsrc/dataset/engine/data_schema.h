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
#ifndef DATASET_ENGINE_DATA_SCHEMA_H_
#define DATASET_ENGINE_DATA_SCHEMA_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include "dataset/core/constants.h"
#include "dataset/core/data_type.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
// A simple class to provide meta info about a column.
class ColDescriptor {
 public:
  // Constructor 1: Simple constructor that leaves things uninitialized.
  ColDescriptor();

  // Constructor 2: Main constructor
  // @param col_name - The name of the column
  // @param col_type - The DE Datatype of the column
  // @param tensor_impl - The (initial) type of tensor implementation for the column
  // @param rank - The number of dimension of the data
  // @param in_shape - option argument for input shape
  ColDescriptor(const std::string &col_name, DataType col_type, TensorImpl tensor_impl, int32_t rank,
                const TensorShape *in_shape = nullptr);

  // Explicit copy constructor is required
  // @param in_cd - the source ColDescriptor
  ColDescriptor(const ColDescriptor &in_cd);

  // Assignment overload
  // @param in_cd - the source ColDescriptor
  ColDescriptor &operator=(const ColDescriptor &in_cd);

  // Destructor
  ~ColDescriptor();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  void Print(std::ostream &out) const;

  // Given a number of elements, this function will compute what the actual Tensor shape would be.
  // If there is no starting TensorShape in this column, or if there is a shape but it contains
  // an unknown dimension, then the output shape returned shall resolve dimensions as needed.
  // @param num_elements - The number of elements in the data for a Tensor
  // @param out_shape - The materialized output Tensor shape
  // @return Status - The error code return
  Status MaterializeTensorShape(int32_t num_elements, TensorShape *out_shape) const;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param cd - reference to the ColDescriptor to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ColDescriptor &cd) {
    cd.Print(out);
    return out;
  }

  // getter function
  // @return The column's DataType
  DataType type() const { return type_; }

  // getter function
  // @return The column's rank
  int32_t rank() const { return rank_; }

  // getter function
  // @return The column's name
  std::string name() const { return col_name_; }

  // getter function
  // @return The column's shape
  TensorShape shape() const;

  // getter function
  // @return TF if the column has an assigned fixed shape.
  bool hasShape() const { return tensor_shape_ != nullptr; }

  // getter function
  // @return The column's tensor implementation type
  TensorImpl tensorImpl() const { return tensor_impl_; }

 private:
  DataType type_;                              // The columns type
  int32_t rank_;                               // The rank for this column (number of dimensions)
  TensorImpl tensor_impl_;                     // The initial flavour of the tensor for this column.
  std::unique_ptr<TensorShape> tensor_shape_;  // The fixed shape (if given by user)
  std::string col_name_;                       // The name of the column
};

// A list of the columns.
class DataSchema {
 public:
  // Constructor
  DataSchema();

  // Destructor
  ~DataSchema();

  // Populates the schema with a dataset type from a json file.  It does not populate any of the
  // column info. To populate everything, use loadSchema() afterwards.
  // @param schema_file_path - Absolute path to the schema file to use for getting dataset type info.
  Status LoadDatasetType(const std::string &schema_file_path);

  // Parses a schema json file and populates the columns and meta info.
  // @param schema_file_path - the schema file that has the column's info to load
  // @param columns_to_load - list of strings for columns to load. if empty, assumes all columns.
  // @return Status - The error code return
  Status LoadSchemaFile(const std::string &schema_file_path, const std::vector<std::string> &columns_to_load);

  // Parses a schema JSON string and populates the columns and meta info.
  // @param schema_json_string - the schema file that has the column's info to load
  // @param columns_to_load - list of strings for columns to load. if empty, assumes all columns.
  // @return Status - The error code return
  Status LoadSchemaString(const std::string &schema_json_string, const std::vector<std::string> &columns_to_load);

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  void Print(std::ostream &out) const;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param ds - reference to the DataSchema to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DataSchema &ds) {
    ds.Print(out);
    return out;
  }

  // Adds a column descriptor to the schema
  // @param cd - The ColDescriptor to add
  // @return Status - The error code return
  Status AddColumn(const ColDescriptor &cd);

  // Setter
  // @param in_type - The Dataset type to set into the schema
  void set_dataset_type(DatasetType in_type) { dataset_type_ = in_type; }

  // getter
  // @return The dataset type of the schema
  DatasetType dataset_type() const { return dataset_type_; }

  // getter
  // @return The reference to a ColDescriptor to get (const version)
  const ColDescriptor &column(int32_t idx) const;

  // getter
  // @return The number of columns in the schema
  int32_t NumColumns() const { return col_descs_.size(); }

  bool Empty() const { return NumColumns() == 0; }

  std::string dir_structure() const { return dir_structure_; }

  std::string dataset_type_str() const { return dataset_type_str_; }

  int64_t num_rows() const { return num_rows_; }

  static const char DEFAULT_DATA_SCHEMA_FILENAME[];

  // Loops through all columns in the schema and returns a map with the column
  // name to column index number.
  // @param out_column_name_map - The output map of columns names to column index
  // @return Status - The error code return
  Status GetColumnNameMap(std::unordered_map<std::string, int32_t> *out_column_name_map);

 private:
  // Internal helper function. Parses the json schema file in any order and produces a schema that
  // does not follow any particular order (json standard does not enforce any ordering protocol).
  // This one produces a schema that contains all of the columns from the schema file.
  // @param column_tree - The nlohmann tree from the json file to parse
  // @return Status - The error code return
  Status AnyOrderLoad(nlohmann::json column_tree);

  // Internal helper function. For each input column name, perform a lookup to the json document to
  // find the matching column.  When the match is found, process that column to build the column
  // descriptor and add to the schema in the order in which the input column names are given.
  // @param column_tree - The nlohmann tree from the json file to parse
  // @param columns_to_load - list of strings for the columns to add to the schema
  // @return Status - The error code return
  Status ColumnOrderLoad(nlohmann::json column_tree, const std::vector<std::string> &columns_to_load);

  // Internal helper function. Given the json tree for a given column, load it into our schema.
  // @param columnTree - The nlohmann child tree for a given column to load.
  // @param col_name - The string name of the column for that subtree.
  // @return Status - The error code return
  Status ColumnLoad(nlohmann::json column_child_tree, const std::string &col_name);

  // Internal helper function. Performs sanity checks on the json file setup.
  // @param js - The nlohmann tree for the schema file
  // @return Status - The error code return
  Status PreLoadExceptionCheck(const nlohmann::json &js);

  DatasetType GetDatasetTYpeFromString(const std::string &type) const;

  std::vector<ColDescriptor> col_descs_;  // Vector of column descriptors
  std::string dataset_type_str_;          // A string that represents the type of dataset
  DatasetType dataset_type_;              // The numeric form of the dataset type from enum
  std::string dir_structure_;             // Implicit or flatten
  int64_t num_rows_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATA_SCHEMA_H_
