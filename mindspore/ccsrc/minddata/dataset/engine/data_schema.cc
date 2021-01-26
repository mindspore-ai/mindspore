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
#include "minddata/dataset/engine/data_schema.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <nlohmann/json.hpp>

#include "utils/ms_utils.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
// A macro for converting an input string representing the column type to it's actual
// numeric column type.
#define STR_TO_TENSORIMPL(in_col_str, out_type) \
  do {                                          \
    if (in_col_str == "cvmat") {                \
      out_type = TensorImpl::kCv;               \
    } else if (in_col_str == "flex") {          \
      out_type = TensorImpl::kFlexible;         \
    } else if (in_col_str == "np") {            \
      out_type = TensorImpl::kNP;               \
    } else {                                    \
      out_type = TensorImpl::kNone;             \
    }                                           \
  } while (false)

// Constructor 1: Simple constructor that leaves things uninitialized.
ColDescriptor::ColDescriptor()
    : type_(DataType::DE_UNKNOWN), rank_(0), tensor_impl_(TensorImpl::kNone), tensor_shape_(nullptr) {}

// Constructor 2: Main constructor
ColDescriptor::ColDescriptor(const std::string &col_name, DataType col_type, TensorImpl tensor_impl, int32_t rank,
                             const TensorShape *in_shape)
    : type_(col_type), rank_(rank), tensor_impl_(tensor_impl), col_name_(col_name) {
  // If a shape was provided, create unique pointer for it and copy construct it into
  // our shape.  Otherwise, set our shape to be empty.
  if (in_shape != nullptr) {
    // Create a shape and copy construct it into our column's shape.
    tensor_shape_ = std::make_unique<TensorShape>(*in_shape);
  } else {
    tensor_shape_ = nullptr;
  }
  // If the user input a shape, then the rank of the input shape needs to match
  // the input rank
  if (in_shape != nullptr && in_shape->known() && in_shape->Size() != rank_) {
    rank_ = in_shape->Size();
    MS_LOG(WARNING) << "Rank does not match the number of dimensions in the provided shape."
                    << " Overriding rank with the number of dimensions in the provided shape.";
  }
}

// Explicit copy constructor is required
ColDescriptor::ColDescriptor(const ColDescriptor &in_cd)
    : type_(in_cd.type_), rank_(in_cd.rank_), tensor_impl_(in_cd.tensor_impl_), col_name_(in_cd.col_name_) {
  // If it has a tensor shape, make a copy of it with our own unique_ptr.
  tensor_shape_ = in_cd.hasShape() ? std::make_unique<TensorShape>(in_cd.shape()) : nullptr;
}

// Assignment overload
ColDescriptor &ColDescriptor::operator=(const ColDescriptor &in_cd) {
  if (&in_cd != this) {
    type_ = in_cd.type_;
    rank_ = in_cd.rank_;
    tensor_impl_ = in_cd.tensor_impl_;
    col_name_ = in_cd.col_name_;
    // If it has a tensor shape, make a copy of it with our own unique_ptr.
    tensor_shape_ = in_cd.hasShape() ? std::make_unique<TensorShape>(in_cd.shape()) : nullptr;
  }
  return *this;
}

// Destructor
ColDescriptor::~ColDescriptor() = default;

// A print method typically used for debugging
void ColDescriptor::Print(std::ostream &out) const {
  out << "  Name          : " << col_name_ << "\n  Type          : " << type_ << "\n  Rank          : " << rank_
      << "\n  Shape         : (";
  if (tensor_shape_) {
    out << *tensor_shape_ << ")\n";
  } else {
    out << "no shape provided)\n";
  }
}

// Given a number of elements, this function will compute what the actual Tensor shape would be.
// If there is no starting TensorShape in this column, or if there is a shape but it contains
// an unknown dimension, then the output shape returned shall resolve dimensions as needed.
Status ColDescriptor::MaterializeTensorShape(int32_t num_elements, TensorShape *out_shape) const {
  if (out_shape == nullptr) {
    RETURN_STATUS_UNEXPECTED("Unexpected null output shape argument.");
  }

  // If the shape is not given in this column, then we assume the shape will be: {numElements}
  if (tensor_shape_ == nullptr) {
    if (this->rank() == 0 && num_elements == 1) {
      *out_shape = TensorShape::CreateScalar();
      return Status::OK();
    }
    *out_shape = TensorShape({num_elements});
    return Status::OK();
  }

  // Build the real TensorShape based on the requested shape and the number of elements in the data.
  // If there are unknown dimensions, then the unknown dimension needs to be filled in.
  // Example: requestedShape: {?,4,3}.
  // If numElements is 24, then the output shape can be computed to: {2,4,3}
  std::vector<dsize_t> requested_shape = tensor_shape_->AsVector();
  int64_t num_elements_of_shape = 1;  // init to 1 as a starting multiplier.

  // unknownDimPosition variable is overloaded to provide 2 meanings:
  // 1) If it's set to DIM_UNKNOWN, then it provides a boolean knowledge to tell us if there are
  //    any unknown dimensions.  i.e. if it's set to unknown, then there are no unknown dimensions.
  // 2) If it's set to a numeric value, then this is the vector index position within the shape
  //    where the single unknown dimension can be found.
  int64_t unknown_dim_position = TensorShape::kDimUnknown;  // Assume there are no unknown dims to start

  for (int i = 0; i < requested_shape.size(); ++i) {
    // If we already had an unknown dimension, then we cannot have a second unknown dimension.
    // We only support the compute of a single unknown dim.
    if (requested_shape[i] == TensorShape::kDimUnknown && unknown_dim_position != TensorShape::kDimUnknown) {
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                    "Requested shape has more than one unknown dimension!");
    }

    // If the current dimension in the requested shape is a known value, then compute the number of
    // elements so far.
    if (requested_shape[i] != TensorShape::kDimUnknown) {
      num_elements_of_shape *= requested_shape[i];
    } else {
      // This dimension is unknown so track which dimension position has it.
      unknown_dim_position = i;
    }
  }

  // Sanity check the the computed element counts divide evenly into the input element count
  if (num_elements < num_elements_of_shape || num_elements_of_shape == 0 || num_elements % num_elements_of_shape != 0) {
    RETURN_STATUS_UNEXPECTED("Requested shape has an invalid element count!");
  }

  // If there was any unknown dimensions, then update the requested shape to fill in the unknown
  // dimension with the correct value.  If there were no unknown dim's then the output shape will
  // remain to be the same as the requested shape.
  if (unknown_dim_position != TensorShape::kDimUnknown) {
    requested_shape[unknown_dim_position] = (num_elements / num_elements_of_shape);
  }

  // Any unknown dimension is filled in now.  Set the output shape
  *out_shape = TensorShape(requested_shape);
  return Status::OK();
}

// getter function for the shape
TensorShape ColDescriptor::shape() const {
  if (tensor_shape_ != nullptr) {
    return *tensor_shape_;  // copy construct a shape to return
  } else {
    return TensorShape::CreateUnknownRankShape();  // empty shape to return
  }
}

const char DataSchema::DEFAULT_DATA_SCHEMA_FILENAME[] = "datasetSchema.json";

// Constructor 1: Simple constructor that leaves things uninitialized.
DataSchema::DataSchema() : num_rows_(0) {}

// Internal helper function. Parses the json schema file in any order and produces a schema that
// does not follow any particular order (json standard does not enforce any ordering protocol).
// This one produces a schema that contains all of the columns from the schema file.
Status DataSchema::AnyOrderLoad(nlohmann::json column_tree) {
  // Iterate over the json file.  Each parent json node is the column name,
  // followed by the column properties in the child tree under the column.
  // Outer loop here iterates over the parents (i.e. the column name)
  if (!column_tree.is_array()) {
    for (nlohmann::json::iterator it = column_tree.begin(); it != column_tree.end(); ++it) {
      std::string col_name = it.key();
      nlohmann::json column_child_tree = it.value();
      RETURN_IF_NOT_OK(ColumnLoad(column_child_tree, col_name));
    }
  } else {
    // Case where the schema is a list of columns not a dict
    for (nlohmann::json::iterator it = column_tree.begin(); it != column_tree.end(); ++it) {
      nlohmann::json column_child_tree = it.value();
      RETURN_IF_NOT_OK(ColumnLoad(column_child_tree, ""));
    }
  }
  return Status::OK();
}

// Internal helper function. For each input column name, perform a lookup to the json document to
// find the matching column.  When the match is found, process that column to build the column
// descriptor and add to the schema in the order in which the input column names are given.id
Status DataSchema::ColumnOrderLoad(nlohmann::json column_tree, const std::vector<std::string> &columns_to_load) {
  if (!column_tree.is_array()) {
    // the json file is dict (e.g., {image: ...})
    // Loop over the column name list
    for (const auto &curr_col_name : columns_to_load) {
      // Find the column in the json document
      auto column_info = column_tree.find(common::SafeCStr(curr_col_name));
      if (column_info == column_tree.end()) {
        RETURN_STATUS_UNEXPECTED("Invalid data, failed to find column name: " + curr_col_name);
      }
      // At this point, columnInfo.value() is the subtree in the json document that contains
      // all of the data for a given column.  This data will formulate our schema column.
      const std::string &col_name = column_info.key();
      nlohmann::json column_child_tree = column_info.value();
      RETURN_IF_NOT_OK(ColumnLoad(column_child_tree, col_name));
    }
  } else {
    // the json file is array (e.g., [name: image...])
    // Loop over the column name list
    for (const auto &curr_col_name : columns_to_load) {
      // Find the column in the json document
      int32_t index = -1;
      int32_t i = 0;
      for (const auto &it_child : column_tree.items()) {
        auto name = it_child.value().find("name");
        if (name == it_child.value().end()) {
          RETURN_STATUS_UNEXPECTED("Name field is missing for this column.");
        }
        if (name.value() == curr_col_name) {
          index = i;
          break;
        }
        i++;
      }
      if (index == -1) {
        RETURN_STATUS_UNEXPECTED("Invalid data, failed to find column name: " + curr_col_name);
      }
      nlohmann::json column_child_tree = column_tree[index];
      RETURN_IF_NOT_OK(ColumnLoad(column_child_tree, curr_col_name));
    }
  }
  return Status::OK();
}

// Internal helper function for parsing shape info and building a vector for the shape construction.
static Status buildShape(const nlohmann::json &shapeVal, std::vector<dsize_t> *outShape) {
  if (outShape == nullptr) {
    RETURN_STATUS_UNEXPECTED("null output shape");
  }
  if (shapeVal.empty()) return Status::OK();

  // Iterate over the integer list and add those values to the output shape tensor
  auto items = shapeVal.items();
  using it_type = decltype(items.begin());
  (void)std::transform(items.begin(), items.end(), std::back_inserter(*outShape), [](it_type j) { return j.value(); });
  return Status::OK();
}

// Internal helper function. Given the json tree for a given column, load it into our schema.
Status DataSchema::ColumnLoad(nlohmann::json column_child_tree, const std::string &col_name) {
  int32_t rank_value = -1;
  TensorImpl t_impl_value = TensorImpl::kFlexible;
  std::string name, type_str;
  std::vector<dsize_t> tmp_shape = {};
  bool shape_field_exists = false;
  // Iterate over this column's attributes.
  // Manually iterating each of the child nodes/trees here so that we can provide our own error handling.
  for (const auto &it_child : column_child_tree.items()) {
    // Save the data for each of the attributes into variables. We'll use these to construct later.
    if (it_child.key() == "name") {
      name = it_child.value();
    } else if (it_child.key() == "type") {
      type_str = it_child.value();
    } else if (it_child.key() == "rank") {
      rank_value = it_child.value();
    } else if (it_child.key() == "t_impl") {
      STR_TO_TENSORIMPL(it_child.value(), t_impl_value);
    } else if (it_child.key() == "shape") {
      shape_field_exists = true;
      RETURN_IF_NOT_OK(buildShape(it_child.value(), &tmp_shape));
    } else {
      std::string err_msg = "Unexpected column attribute " + it_child.key() + " for column " + col_name;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  if (!name.empty()) {
    if (!col_name.empty() && col_name != name) {
      std::string err_msg =
        "json schema file for column " + col_name + " has column name that does not match columnsToLoad";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  } else {
    if (col_name.empty()) {
      std::string err_msg = "json schema file for column " + col_name + " has invalid or missing column name.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else {
      name = col_name;
    }
  }
  // data type is mandatory field
  if (type_str.empty())
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "json schema file for column " + col_name + " has invalid or missing column type.");

  // rank number is mandatory field
  if (rank_value <= -1)
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "json schema file for column " + col_name + " must define a positive rank value.");

  // Create the column descriptor for this column from the data we pulled from the json file
  TensorShape col_shape = TensorShape(tmp_shape);
  if (shape_field_exists)
    (void)this->AddColumn(ColDescriptor(name, DataType(type_str), t_impl_value, rank_value, &col_shape));
  else
    // Create a column descriptor that doesn't have a shape
    (void)this->AddColumn(ColDescriptor(name, DataType(type_str), t_impl_value, rank_value));
  return Status::OK();
}

// Parses a schema json file and populates the columns and meta info.
Status DataSchema::LoadSchemaFile(const std::string &schema_file_path,
                                  const std::vector<std::string> &columns_to_load) {
  try {
    std::ifstream in(schema_file_path);

    nlohmann::json js;
    in >> js;
    RETURN_IF_NOT_OK(PreLoadExceptionCheck(js));
    try {
      num_rows_ = js.at("numRows").get<int64_t>();
    } catch (nlohmann::json::out_of_range &e) {
      num_rows_ = 0;
    } catch (nlohmann::json::exception &e) {
      RETURN_STATUS_UNEXPECTED("Unable to parse \"numRows\" from schema");
    }
    nlohmann::json column_tree = js.at("columns");
    if (column_tree.empty()) {
      RETURN_STATUS_UNEXPECTED("columns is null");
    }
    if (columns_to_load.empty()) {
      // Parse the json tree and load the schema's columns in whatever order that the json
      // layout decides
      RETURN_IF_NOT_OK(this->AnyOrderLoad(column_tree));
    } else {
      RETURN_IF_NOT_OK(this->ColumnOrderLoad(column_tree, columns_to_load));
    }
  } catch (const std::exception &err) {
    // Catch any exception and convert to Status return code
    RETURN_STATUS_UNEXPECTED("Schema file failed to load");
  }
  return Status::OK();
}

// Parses a schema json string and populates the columns and meta info.
Status DataSchema::LoadSchemaString(const std::string &schema_json_string,
                                    const std::vector<std::string> &columns_to_load) {
  try {
    nlohmann::json js = nlohmann::json::parse(schema_json_string);
    RETURN_IF_NOT_OK(PreLoadExceptionCheck(js));
    num_rows_ = js.value("numRows", 0);
    nlohmann::json column_tree = js.at("columns");
    if (column_tree.empty()) {
      RETURN_STATUS_UNEXPECTED("columns is null");
    }
    if (columns_to_load.empty()) {
      // Parse the json tree and load the schema's columns in whatever order that the json
      // layout decides
      RETURN_IF_NOT_OK(this->AnyOrderLoad(column_tree));
    } else {
      RETURN_IF_NOT_OK(this->ColumnOrderLoad(column_tree, columns_to_load));
    }
  } catch (const std::exception &err) {
    // Catch any exception and convert to Status return code
    RETURN_STATUS_UNEXPECTED("Schema file failed to load");
  }
  return Status::OK();
}

// Destructor
DataSchema::~DataSchema() = default;

// Getter for the ColDescriptor by index
const ColDescriptor &DataSchema::column(int32_t idx) const {
  MS_ASSERT(idx < static_cast<int>(col_descs_.size()));
  return col_descs_[idx];
}

// A print method typically used for debugging
void DataSchema::Print(std::ostream &out) const {
  out << "Dataset schema: (";
  for (const auto &col_desc : col_descs_) {
    out << col_desc << "\n";
  }
}

// Adds a column descriptor to the schema
Status DataSchema::AddColumn(const ColDescriptor &cd) {
  // Sanity check there's not a duplicate name before adding the column
  for (int32_t i = 0; i < col_descs_.size(); ++i) {
    if (col_descs_[i].name() == cd.name()) {
      std::ostringstream ss;
      ss << "column name '" << cd.name() << "' already exists in schema.";
      std::string err_msg = ss.str();
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  col_descs_.push_back(cd);
  return Status::OK();
}

// Internal helper function. Performs sanity checks on the json file setup.
Status DataSchema::PreLoadExceptionCheck(const nlohmann::json &js) {
  // Check if columns node exists.  It is required for building schema from file.
  if (js.find("columns") == js.end())
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "\"columns\" node is required in the schema json file.");
  return Status::OK();
}

// Loops through all columns in the schema and returns a map with the column
// name to column index number.
Status DataSchema::GetColumnNameMap(std::unordered_map<std::string, int32_t> *out_column_name_map) {
  if (out_column_name_map == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "unexpected null output column name map.");
  }

  for (int32_t i = 0; i < col_descs_.size(); ++i) {
    if (col_descs_[i].name().empty()) {
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                    "Constructing column name map from schema, but found empty column name.");
    }
    (*out_column_name_map)[col_descs_[i].name()] = i;
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
