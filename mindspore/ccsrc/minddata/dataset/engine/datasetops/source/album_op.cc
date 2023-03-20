/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/album_op.h"
#include <fstream>
#include <iomanip>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif

namespace mindspore {
namespace dataset {
AlbumOp::AlbumOp(int32_t num_wkrs, std::string file_dir, int32_t queue_size, bool do_decode,
                 const std::set<std::string> &exts, std::unique_ptr<DataSchema> data_schema,
                 std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_wkrs, queue_size, std::move(sampler)),
      folder_path_(std::move(file_dir)),
      decode_(do_decode),
      extensions_(exts),
      data_schema_(std::move(data_schema)),
      sampler_ind_(0),
      dirname_offset_(0) {
  // Set the column name map (base class field)
  for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    column_name_id_map_[data_schema_->Column(i).Name()] = i;
  }
}

// Helper function for string comparison
// album sorts the files via numerical values, so this is not a simple string comparison
bool StrComp(const std::string &a, const std::string &b) {
  // returns 1 if string "a" represent a numeric value less than string "b"
  // the following will always return name, provided there is only one "." character in name
  // "." character is guaranteed to exist since the extension is checked before this function call.
  int64_t value_a = std::stoi(a.substr(1, a.find(".")).c_str());
  int64_t value_b = std::stoi(b.substr(1, b.find(".")).c_str());
  return value_a < value_b;
}

// Single thread to go through the folder directory and gets all file names
// calculate numRows then return
Status AlbumOp::PrepareData() {
  Path folder(folder_path_);
  dirname_offset_ = folder_path_.length();
  std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&folder);
  if (!folder.Exists() || dirItr == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid folder, " + folder_path_ + " does not exist or permission denied.");
  }
  MS_LOG(INFO) << "Album folder Path found: " << folder_path_ << ".";

  while (dirItr->HasNext()) {
    Path file = dirItr->Next();
    if (extensions_.empty() || extensions_.find(file.Extension()) != extensions_.end()) {
      (void)image_rows_.push_back(file.ToString().substr(dirname_offset_));
    } else {
      MS_LOG(INFO) << "Album operator unsupported file found: " << file.ToString()
                   << ", extension: " << file.Extension() << ".";
    }
  }

  std::sort(image_rows_.begin(), image_rows_.end(), StrComp);
  num_rows_ = image_rows_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, AlbumDataset API can't read the data file (interface mismatch or no data found). "
      "Check file path:" +
      folder_path_ + ".");
  }
  return Status::OK();
}

// Only support JPEG/PNG/GIF/BMP
// Optimization: Could take in a tensor
// This function does not return status because we want to just skip bad input, not crash
bool AlbumOp::CheckImageType(const std::string &file_name, bool *valid) {
  if (valid == nullptr) {
    MS_LOG(ERROR) << "[Internal ERROR] Album parameter can't be nullptr.";
    return false;
  }
  std::ifstream file_handle;
  constexpr int read_num = 3;
  *valid = false;
  file_handle.open(file_name, std::ios::binary | std::ios::in);
  if (!file_handle.is_open()) {
    return false;
  }
  unsigned char file_type[read_num];
  (void)file_handle.read(reinterpret_cast<char *>(file_type), read_num);

  if (file_handle.fail()) {
    file_handle.close();
    return false;
  }
  file_handle.close();
  if (file_type[0] == 0xff && file_type[1] == 0xd8 && file_type[2] == 0xff) {
    // Normal JPEGs start with \xff\xd8\xff\xe0
    // JPEG with EXIF stats with \xff\xd8\xff\xe1
    // Use \xff\xd8\xff to cover both.
    *valid = true;
  }
  return true;
}

Status AlbumOp::LoadImageTensor(const std::string &image_file_path, int32_t col_num, TensorRow *row) {
  TensorPtr image;
  std::ifstream fs;
  fs.open(image_file_path, std::ios::binary | std::ios::in);
  if (fs.fail()) {
    MS_LOG(WARNING) << "File not found:" << image_file_path << ".";
    // If file doesn't exist, we don't flag this as error in input check, simply push back empty tensor
    RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, row));
    return Status::OK();
  }
  fs.close();
  // Hack logic to replace png images with empty tensor
  Path file(image_file_path);
  std::set<std::string> png_ext = {".png", ".PNG"};
  if (png_ext.find(file.Extension()) != png_ext.end()) {
    // load empty tensor since image is not jpg
    MS_LOG(INFO) << "PNG!" << image_file_path << ".";
    RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, row));
    return Status::OK();
  }
  // treat bin files separately
  std::set<std::string> bin_ext = {".bin", ".BIN"};
  if (bin_ext.find(file.Extension()) != bin_ext.end()) {
    // load empty tensor since image is not jpg
    MS_LOG(INFO) << "Bin file found" << image_file_path << ".";
    RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_file_path, &image));
    row->push_back(std::move(image));
    return Status::OK();
  }

  // check that the file is an image before decoding
  bool valid = false;
  bool check_success = CheckImageType(image_file_path, &valid);
  if (!check_success || !valid) {
    RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, row));
    return Status::OK();
  }
  // if it is a jpeg image, load and try to decode
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_file_path, &image));
  if (decode_ && valid) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, row));
      return Status::OK();
    }
  }
  row->push_back(std::move(image));
  return Status::OK();
}

Status AlbumOp::LoadStringArrayTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row) {
  std::vector<std::string> data = json_obj;

  MS_LOG(INFO) << "String array label found: " << data << ".";
  TensorPtr label;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &label));
  row->push_back(std::move(label));
  return Status::OK();
}

Status AlbumOp::LoadStringTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row) {
  std::string data = json_obj;
  // now we iterate over the elements in json

  MS_LOG(INFO) << "String label found: " << data << ".";
  TensorPtr label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar<std::string>(data, &label));
  row->push_back(std::move(label));
  return Status::OK();
}

Status AlbumOp::LoadIntArrayTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row) {
  TensorPtr label;
  // consider templating this function to handle all ints
  if (data_schema_->Column(col_num).Type() == DataType::DE_INT64) {
    std::vector<int64_t> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &label));
  } else if (data_schema_->Column(col_num).Type() == DataType::DE_INT32) {
    std::vector<int32_t> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items1 = json_obj.items();
    using it_type = decltype(items1.begin());
    (void)std::transform(items1.begin(), items1.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &label));
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid column type, column type of " + data_schema_->Column(col_num).Name() +
                             " should be int32 or int64, but got " + data_schema_->Column(col_num).Type().ToString());
  }
  row->push_back(std::move(label));
  return Status::OK();
}

Status AlbumOp::LoadFloatArrayTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row) {
  TensorPtr float_array;
  // consider templating this function to handle all ints
  if (data_schema_->Column(col_num).Type() == DataType::DE_FLOAT64) {
    std::vector<double> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &float_array));
  } else if (data_schema_->Column(col_num).Type() == DataType::DE_FLOAT32) {
    std::vector<float> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items1 = json_obj.items();
    using it_type = decltype(items1.begin());
    (void)std::transform(items1.begin(), items1.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &float_array));
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid column type, column type of " + data_schema_->Column(col_num).Name() +
                             " should be float32 nor float64, but got " +
                             data_schema_->Column(col_num).Type().ToString());
  }
  row->push_back(std::move(float_array));
  return Status::OK();
}

Status AlbumOp::LoadIDTensor(const std::string &file, int32_t col_num, TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  if (data_schema_->Column(col_num).Type() == DataType::DE_STRING) {
    TensorPtr id;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<std::string>(file, &id));
    row->push_back(std::move(id));
    return Status::OK();
  }
  // hack to get the file name without extension, the 1 is to get rid of the backslash character
  int64_t image_id = std::stoi(file.substr(1, file.find(".")).c_str());
  TensorPtr id;
  RETURN_IF_NOT_OK(Tensor::CreateScalar<int64_t>(image_id, &id));
  MS_LOG(INFO) << "File ID " << image_id << ".";
  row->push_back(std::move(id));
  return Status::OK();
}

Status AlbumOp::LoadEmptyTensor(int32_t col_num, TensorRow *row) {
  // hack to get the file name without extension, the 1 is to get rid of the backslash character
  TensorPtr empty_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({0}), data_schema_->Column(col_num).Type(), &empty_tensor));
  row->push_back(std::move(empty_tensor));
  return Status::OK();
}

// Loads a tensor with float value, issue with float64, we don't have reverse look up to the type
// So we actually have to check what type we want to fill the tensor with.
// Float64 doesn't work with reinterpret cast here. Otherwise we limit the float in the schema to
// only be float32, seems like a weird limitation to impose
Status AlbumOp::LoadFloatTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row) {
  TensorPtr float_tensor;
  if (data_schema_->Column(col_num).Type() == DataType::DE_FLOAT64) {
    double data = json_obj;
    MS_LOG(INFO) << "double found: " << json_obj << ".";
    RETURN_IF_NOT_OK(Tensor::CreateScalar<double>(data, &float_tensor));
  } else if (data_schema_->Column(col_num).Type() == DataType::DE_FLOAT32) {
    float data1 = json_obj;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<float>(data1, &float_tensor));
    MS_LOG(INFO) << "float found: " << json_obj << ".";
  }
  row->push_back(std::move(float_tensor));
  return Status::OK();
}

// Loads a tensor with int value, we have to cast the value to type specified in the schema.
Status AlbumOp::LoadIntTensor(const nlohmann::json &json_obj, int32_t col_num, TensorRow *row) {
  TensorPtr int_tensor;
  if (data_schema_->Column(col_num).Type() == DataType::DE_INT64) {
    int64_t data = json_obj;
    MS_LOG(INFO) << "int64 found: " << json_obj << ".";
    RETURN_IF_NOT_OK(Tensor::CreateScalar<int64_t>(data, &int_tensor));
  } else if (data_schema_->Column(col_num).Type() == DataType::DE_INT32) {
    int32_t data = json_obj;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<int32_t>(data, &int_tensor));
    MS_LOG(INFO) << "int32 found: " << json_obj << ".";
  }
  row->push_back(std::move(int_tensor));
  return Status::OK();
}

// Load 1 TensorRow (image,label) using 1 ImageColumns. 1 function call produces 1 TensorRow
// possible optimization: the helper functions of LoadTensorRow should be optimized
// to take a reference to a column descriptor?
// the design of this class is to make the code more readable, forgoing minor performance gain like
// getting rid of duplicated checks
Status AlbumOp::LoadTensorRow(row_id_type row_id, TensorRow *row) {
  std::string file = image_rows_[row_id];
  // testing here is to just print out file path
  (*row) = TensorRow(row_id, {});
  MS_LOG(INFO) << "Image row file: " << file << ".";

  std::ifstream file_handle(folder_path_ + file);
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid json file, " + folder_path_ + file + " does not exist or permission denied.");
  }
  std::string line;
  while (getline(file_handle, line)) {
    try {
      nlohmann::json js = nlohmann::json::parse(line);
      MS_LOG(INFO) << "This Line: " << line << ".";

      // note if take a schema here, then we have to iterate over all column descriptors in schema and check for key
      // get columns in schema:
      int32_t columns = data_schema_->NumColumns();

      // loop over each column descriptor, this can optimized by switch cases
      for (int32_t i = 0; i < columns; i++) {
        file_handle.close();
        RETURN_IF_NOT_OK(loadColumnData(file, i, js, row));
      }
    } catch (const std::exception &err) {
      file_handle.close();
      RETURN_STATUS_UNEXPECTED("Invalid file, " + folder_path_ + file + " load failed: " + std::string(err.what()));
    }
  }
  file_handle.close();
  std::vector<std::string> path(row->size(), folder_path_ + file);
  row->setPath(path);
  return Status::OK();
}

Status AlbumOp::loadColumnData(const std::string &file, int32_t index, nlohmann::json js, TensorRow *row) {
  int32_t i = index;
  // special case to handle
  if (data_schema_->Column(i).Name() == "id") {
    // id is internal, special case to load from file
    return LoadIDTensor(file, i, row);
  }
  // find if key does not exist, insert placeholder nullptr if not found
  if (js.find(data_schema_->Column(i).Name()) == js.end()) {
    // iterator not found, push nullptr as placeholder
    MS_LOG(INFO) << "Pushing empty tensor for column: " << data_schema_->Column(i).Name() << ".";
    return LoadEmptyTensor(i, row);
  }
  nlohmann::json column_value = js.at(data_schema_->Column(i).Name());
  MS_LOG(INFO) << "This column is: " << data_schema_->Column(i).Name() << ".";
  bool is_array = column_value.is_array();
  // load single string
  if (column_value.is_string() && data_schema_->Column(i).Type() == DataType::DE_STRING) {
    return LoadStringTensor(column_value, i, row);
  }
  // load string array
  if (is_array && data_schema_->Column(i).Type() == DataType::DE_STRING) {
    return LoadStringArrayTensor(column_value, i, row);
  }
  // load image file
  if (column_value.is_string() && data_schema_->Column(i).Type() != DataType::DE_STRING) {
    std::string image_file_path = column_value;
    return LoadImageTensor(image_file_path, i, row);
  }
  // load float value
  bool judge_float = (data_schema_->Column(i).Type() == DataType::DE_FLOAT32) ||
                     (data_schema_->Column(i).Type() == DataType::DE_FLOAT64);
  if (!is_array && judge_float) {
    return LoadFloatTensor(column_value, i, row);
  }
  // load float array
  if (is_array && judge_float) {
    return LoadFloatArrayTensor(column_value, i, row);
  }
  // int value
  bool judge_int =
    (data_schema_->Column(i).Type() == DataType::DE_INT64) || (data_schema_->Column(i).Type() == DataType::DE_INT32);
  if (!is_array && judge_int) {
    return LoadIntTensor(column_value, i, row);
  }
  // int array
  if (is_array && judge_int) {
    return LoadIntArrayTensor(column_value, i, row);
  } else {
    MS_LOG(WARNING) << "Value type for column: " << data_schema_->Column(i).Name() << " is not supported.";
    return Status::OK();
  }
}

void AlbumOp::Print(std::ostream &out, bool show_all) const {
  constexpr int64_t field_width = 2;
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(field_width) << operator_id_ << ") <AlbumOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nAlbum directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status AlbumOp::ComputeColMap() {
  // Set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(INFO) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
