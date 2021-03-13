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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATA_SCHEMA_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATA_SCHEMA_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// \class ColDescriptor data_schema.h
/// \brief A simple class to provide meta info about a column.
class ColDescriptor {
 public:
  /// \brief Constructor 1: Simple constructor that leaves things uninitialized.
  ColDescriptor();

  /// \brief Constructor 2: Main constructor
  /// \param[in] col_name - The name of the column
  /// \param[in] col_type - The DE Datatype of the column
  /// \param[in] tensor_impl - The (initial) type of tensor implementation for the column
  /// \param[in] rank - The number of dimension of the data
  /// \param[in] in_shape - option argument for input shape
  ColDescriptor(const std::string &col_name, DataType col_type, TensorImpl tensor_impl, int32_t rank,
                const TensorShape *in_shape = nullptr);

  /// \brief Explicit copy constructor is required
  /// \param[in] in_cd - the source ColDescriptor
  ColDescriptor(const ColDescriptor &in_cd);

  /// \brief Assignment overload
  /// \param in_cd - the source ColDescriptor
  ColDescriptor &operator=(const ColDescriptor &in_cd);

  /// \brief Destructor
  ~ColDescriptor();

  /// \brief A print method typically used for debugging
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const;

  /// \brief Given a number of elements, this function will compute what the actual Tensor shape would be.
  ///     If there is no starting TensorShape in this column, or if there is a shape but it contains
  ///     an unknown dimension, then the output shape returned shall resolve dimensions as needed.
  /// \param[in] num_elements - The number of elements in the data for a Tensor
  /// \param[in/out] out_shape - The materialized output Tensor shape
  /// \return Status The status code returned
  Status MaterializeTensorShape(int32_t num_elements, TensorShape *out_shape) const;

  /// \brief << Stream output operator overload
  ///     This allows you to write the debug print info using stream operators
  /// \param[in] out - reference to the output stream being overloaded
  /// \param[in] cd - reference to the ColDescriptor to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ColDescriptor &cd) {
    cd.Print(out);
    return out;
  }

  /// \brief getter function
  /// \return The column's DataType
  DataType type() const { return type_; }

  /// \brief getter function
  /// \return The column's rank
  int32_t rank() const { return rank_; }

  /// \brief getter function
  /// \return The column's name
  std::string name() const { return col_name_; }

  /// \brief getter function
  /// \return The column's shape
  TensorShape shape() const;

  /// \brief getter function
  /// \return TF if the column has an assigned fixed shape.
  bool hasShape() const { return tensor_shape_ != nullptr; }

  /// \brief getter function
  /// \return The column's tensor implementation type
  TensorImpl tensorImpl() const { return tensor_impl_; }

 private:
  DataType type_;                              // The columns type
  int32_t rank_;                               // The rank for this column (number of dimensions)
  TensorImpl tensor_impl_;                     // The initial flavour of the tensor for this column
  std::unique_ptr<TensorShape> tensor_shape_;  // The fixed shape (if given by user)
  std::string col_name_;                       // The name of the column
};

/// \class DataSchema data_schema.h
/// \brief A list of the columns.
class DataSchema {
 public:
  /// \brief Constructor
  DataSchema();

  /// \brief Destructor
  ~DataSchema();

  /// \brief Parses a schema json file and populates the columns and meta info.
  /// \param[in] schema_file_path - the schema file that has the column's info to load
  /// \param[in] columns_to_load - list of strings for columns to load. if empty, assumes all columns.
  /// \return Status The status code returned
  Status LoadSchemaFile(const std::string &schema_file_path, const std::vector<std::string> &columns_to_load);

  /// \brief Parses a schema JSON string and populates the columns and meta info.
  /// \param[in] schema_json_string - the schema file that has the column's info to load
  /// \param[in] columns_to_load - list of strings for columns to load. if empty, assumes all columns.
  /// \return Status The status code returned
  Status LoadSchemaString(const std::string &schema_json_string, const std::vector<std::string> &columns_to_load);

  /// \brief A print method typically used for debugging
  /// \param[in] out - The output stream to write output to
  void Print(std::ostream &out) const;

  /// \brief << Stream output operator overload. This allows you to write the debug print info using stream operators
  /// \param[in] out - reference to the output stream being overloaded
  /// \param[in] ds - reference to the DataSchema to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DataSchema &ds) {
    ds.Print(out);
    return out;
  }

  /// \brief Adds a column descriptor to the schema
  /// \param[in] cd - The ColDescriptor to add
  /// \return Status The status code returned
  Status AddColumn(const ColDescriptor &cd);

  /// \brief getter
  /// \return The reference to a ColDescriptor to get (const version)
  const ColDescriptor &column(int32_t idx) const;

  /// \brief getter
  /// \return The number of columns in the schema
  int32_t NumColumns() const { return col_descs_.size(); }

  bool Empty() const { return NumColumns() == 0; }

  /// \brief getter
  /// \return The number of rows read from schema
  int64_t num_rows() const { return num_rows_; }

  static const char DEFAULT_DATA_SCHEMA_FILENAME[];

  /// \brief Loops through all columns in the schema and returns a map with the column name to column index number.
  /// \param[in/out] out_column_name_map - The output map of columns names to column index
  /// \return Status The status code returned
  Status GetColumnNameMap(std::unordered_map<std::string, int32_t> *out_column_name_map);

 private:
  /// \brief Internal helper function. Parses the json schema file in any order and produces a schema that
  ///     does not follow any particular order (json standard does not enforce any ordering protocol).
  ///     This one produces a schema that contains all of the columns from the schema file.
  /// \param[in] column_tree - The nlohmann tree from the json file to parse
  /// \return Status The status code returned
  Status AnyOrderLoad(nlohmann::json column_tree);

  /// \brief Internal helper function. For each input column name, perform a lookup to the json document to
  ///     find the matching column.  When the match is found, process that column to build the column
  ///     descriptor and add to the schema in the order in which the input column names are given.
  /// \param[in] column_tree - The nlohmann tree from the json file to parse
  /// \param[in] columns_to_load - list of strings for the columns to add to the schema
  /// \return Status The status code returned
  Status ColumnOrderLoad(nlohmann::json column_tree, const std::vector<std::string> &columns_to_load);

  /// \brief Internal helper function. Given the json tree for a given column, load it into our schema.
  /// \param[in] columnTree - The nlohmann child tree for a given column to load.
  /// \param[in] col_name - The string name of the column for that subtree.
  /// \return Status The status code returned
  Status ColumnLoad(nlohmann::json column_child_tree, const std::string &col_name);

  /// \brief Internal helper function. Performs sanity checks on the json file setup.
  /// \param[in] js - The nlohmann tree for the schema file
  /// \return Status The status code returned
  Status PreLoadExceptionCheck(const nlohmann::json &js);

  std::vector<ColDescriptor> col_descs_;  // Vector of column descriptors
  int64_t num_rows_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATA_SCHEMA_H_
