/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "album_op_android.h"  //NOLINT
#include <fstream>
#include <iomanip>
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#include "minddata/dataset/kernels/image/exif_utils.h"

namespace mindspore {
namespace dataset {

AlbumOp::AlbumOp(const std::string &file_dir, bool do_decode, const std::string &schema_file,
                 const std::vector<std::string> &column_names, const std::set<std::string> &exts)
    : folder_path_(file_dir),
      decode_(do_decode),
      extensions_(exts),
      schema_file_(schema_file),
      row_cnt_(0),
      buf_cnt_(0),
      current_cnt_(0),
      dirname_offset_(0),
      sampler_(false),
      sampler_index_(0),
      rotate_(true),
      column_names_(column_names) {
  PrescanEntry();
}

AlbumOp::AlbumOp(const std::string &file_dir, bool do_decode, const std::string &schema_file,
                 const std::vector<std::string> &column_names, const std::set<std::string> &exts, uint32_t index)
    : folder_path_(file_dir),
      decode_(do_decode),
      extensions_(exts),
      schema_file_(schema_file),
      row_cnt_(0),
      buf_cnt_(0),
      current_cnt_(0),
      dirname_offset_(0),
      sampler_(true),
      sampler_index_(index),
      rotate_(true),
      column_names_(column_names) {
  PrescanEntry();
}

// Helper function for string comparison
// album sorts the files via numerical values, so this is not a simple string comparison
bool StrComp(const std::string &a, const std::string &b) {
  // returns 1 if string "a" represent a numeric value less than string "b"
  // the following will always return name, provided there is only one "." character in name
  // "." character is guaranteed to exist since the extension is checked before this function call.
  int64_t value_a = std::atoi(a.substr(1, a.find(".")).c_str());
  int64_t value_b = std::atoi(b.substr(1, b.find(".")).c_str());
  return value_a < value_b;
}

// Single thread to go through the folder directory and gets all file names
// calculate numRows then return
Status AlbumOp::PrescanEntry() {
  data_schema_ = std::make_unique<DataSchema>();
  Path schema_file(schema_file_);
  if (schema_file_ == "" || !schema_file.Exists()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, schema_file is invalid or not set: " + schema_file_);
  } else {
    MS_LOG(INFO) << "Schema file provided: " << schema_file_ << ".";
    data_schema_->LoadSchemaFile(schema_file_, columns_to_load_);
  }

  Path folder(folder_path_);
  dirname_offset_ = folder_path_.length();
  std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&folder);
  if (folder.Exists() == false || dirItr == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open folder: " + folder_path_);
  }
  MS_LOG(INFO) << "Album folder Path found: " << folder_path_ << ".";

  while (dirItr->hasNext()) {
    Path file = dirItr->next();
    if (extensions_.empty() || extensions_.find(file.Extension()) != extensions_.end()) {
      (void)image_rows_.push_back(file.toString().substr(dirname_offset_));
    } else {
      MS_LOG(WARNING) << "Album operator unsupported file found: " << file.toString()
                      << ", extension: " << file.Extension() << ".";
    }
  }

  std::sort(image_rows_.begin(), image_rows_.end(), StrComp);

  if (image_rows_.size() == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API AlbumDataset. Please check file path or dataset API.");
  }

  if (sampler_) {
    if (sampler_index_ < 0 || sampler_index_ >= image_rows_.size()) {
      RETURN_STATUS_UNEXPECTED("the sampler index was out of range");
    }
    std::vector<std::string> tmp;
    tmp.emplace_back(image_rows_[sampler_index_]);
    image_rows_.clear();
    image_rows_ = tmp;
  }

  return Status::OK();
}

// contains the main logic of pulling a IOBlock from IOBlockQueue, load a buffer and push the buffer to out_connector_
// IMPORTANT: 1 IOBlock produces 1 DataBuffer
bool AlbumOp::GetNextRow(std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row) {
  if (map_row == nullptr) {
    MS_LOG(ERROR) << "GetNextRow in AlbumOp: the point of map_row is nullptr";
    return false;
  }

  if (current_cnt_ == image_rows_.size()) {
    return false;
  }

  Status ret = LoadTensorRow(current_cnt_, image_rows_[current_cnt_], map_row);
  if (ret.IsError()) {
    MS_LOG(ERROR) << "GetNextRow in AlbumOp: " << ret.ToString() << "\n";
    return false;
  }
  current_cnt_++;
  return true;
}

// Only support JPEG/PNG/GIF/BMP
// Optimization: Could take in a tensor
// This function does not return status because we want to just skip bad input, not crash
bool AlbumOp::CheckImageType(const std::string &file_name, bool *valid) {
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

bool AlbumOp::IsReadColumn(const std::string &column_name) {
  for (uint32_t i = 0; i < this->column_names_.size(); i++) {
    if (this->column_names_[i] == column_name) {
      return true;
    }
  }
  return false;
}

Status AlbumOp::LoadImageTensor(const std::string &image_file_path, uint32_t col_num, TensorPtr *tensor) {
  TensorPtr image;
  TensorPtr rotate_tensor;
  std::ifstream fs;
  fs.open(image_file_path, std::ios::binary | std::ios::in);
  if (fs.fail()) {
    MS_LOG(WARNING) << "File not found:" << image_file_path << ".";
    // If file doesn't exist, we don't flag this as error in input check, simply push back empty tensor
    RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, tensor));
    return Status::OK();
  }
  // Hack logic to replace png images with empty tensor
  Path file(image_file_path);
  std::set<std::string> png_ext = {".png", ".PNG"};
  if (png_ext.find(file.Extension()) != png_ext.end()) {
    // load empty tensor since image is not jpg
    MS_LOG(INFO) << "load empty tensor since image is PNG" << image_file_path << ".";
    RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, tensor));
    return Status::OK();
  }
  // treat bin files separately
  std::set<std::string> bin_ext = {".bin", ".BIN"};
  if (bin_ext.find(file.Extension()) != bin_ext.end()) {
    // load empty tensor since image is not jpg
    MS_LOG(INFO) << "Bin file found" << image_file_path << ".";
    RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_file_path, tensor));
    return Status::OK();
  }

  // check that the file is an image before decoding
  bool valid = false;
  bool check_success = CheckImageType(image_file_path, &valid);
  if (!check_success || !valid) {
    RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, tensor));
    return Status::OK();
  }
  // if it is a jpeg image, load and try to decode
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_file_path, &image));
  Status rc;
  if (decode_ && valid) {
    rc = Decode(image, tensor);
    if (rc.IsError()) {
      RETURN_IF_NOT_OK(LoadEmptyTensor(col_num, tensor));
      return Status::OK();
    }
  }
  return Status::OK();
}

// get orientation from EXIF file
int AlbumOp::GetOrientation(const std::string &folder_path) {
  FILE *fp = fopen(folder_path.c_str(), "rb");
  if (fp == nullptr) {
    MS_LOG(ERROR) << "Can't read file for EXIF:  file = " << folder_path;
    return 0;
  }
  fseek(fp, 0, SEEK_END);
  int64_t fsize = ftell(fp);
  rewind(fp);
  if (fsize > INT_MAX) {
    fclose(fp);
    return 0;
  }
  unsigned char *buf = new unsigned char[fsize];
  if (fread(buf, 1, fsize, fp) != fsize) {
    MS_LOG(ERROR) << "read file size error for EXIF:  file = " << folder_path;
    delete[] buf;
    fclose(fp);
    return 0;
  }
  fclose(fp);

  // Parse EXIF
  mindspore::dataset::ExifInfo result;
  int code = result.parseOrientation(buf, fsize);
  delete[] buf;
  MS_LOG(INFO) << "AlbumOp::GetOrientation:  orientation= " << code << ".";
  return code;
}

Status AlbumOp::LoadStringArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor) {
  std::vector<std::string> data = json_obj.get<std::vector<std::string>>();

  MS_LOG(INFO) << "String array label found: " << data << ".";
  //  TensorPtr label;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, tensor));
  return Status::OK();
}

Status AlbumOp::LoadStringTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor) {
  std::string data = json_obj;
  // now we iterate over the elements in json

  MS_LOG(INFO) << "String label found: " << data << ".";
  TensorPtr label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar<std::string>(data, tensor));
  return Status::OK();
}

Status AlbumOp::LoadIntArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor) {
  //  TensorPtr label;
  // consider templating this function to handle all ints
  if (data_schema_->column(col_num).type() == DataType::DE_INT64) {
    std::vector<int64_t> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, tensor));
  } else if (data_schema_->column(col_num).type() == DataType::DE_INT32) {
    std::vector<int32_t> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, tensor));
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid data, column type is neither int32 nor int64, it is " +
                             data_schema_->column(col_num).type().ToString());
  }
  return Status::OK();
}

Status AlbumOp::LoadFloatArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor) {
  //  TensorPtr float_array;
  // consider templating this function to handle all ints
  if (data_schema_->column(col_num).type() == DataType::DE_FLOAT64) {
    std::vector<double> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, tensor));
  } else if (data_schema_->column(col_num).type() == DataType::DE_FLOAT32) {
    std::vector<float> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, tensor));
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid data, column type is neither float32 nor float64, it is " +
                             data_schema_->column(col_num).type().ToString());
  }
  return Status::OK();
}

Status AlbumOp::LoadIDTensor(const std::string &file, uint32_t col_num, TensorPtr *tensor) {
  if (data_schema_->column(col_num).type() == DataType::DE_STRING) {
    //    TensorPtr id;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<std::string>(file, tensor));
    return Status::OK();
  }
  // hack to get the file name without extension, the 1 is to get rid of the backslash character
  int64_t image_id = std::atoi(file.substr(1, file.find(".")).c_str());
  //  TensorPtr id;
  RETURN_IF_NOT_OK(Tensor::CreateScalar<int64_t>(image_id, tensor));
  MS_LOG(INFO) << "File ID " << image_id << ".";
  return Status::OK();
}

Status AlbumOp::LoadEmptyTensor(uint32_t col_num, TensorPtr *tensor) {
  // hack to get the file name without extension, the 1 is to get rid of the backslash character
  //  TensorPtr empty_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({0}), data_schema_->column(col_num).type(), tensor));
  return Status::OK();
}

// Loads a tensor with float value, issue with float64, we don't have reverse look up to the type
// So we actually have to check what type we want to fill the tensor with.
// Float64 doesn't work with reinterpret cast here. Otherwise we limit the float in the schema to
// only be float32, seems like a weird limitation to impose
Status AlbumOp::LoadFloatTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor) {
  //  TensorPtr float_tensor;
  if (data_schema_->column(col_num).type() == DataType::DE_FLOAT64) {
    double data = json_obj;
    MS_LOG(INFO) << "double found: " << json_obj << ".";
    RETURN_IF_NOT_OK(Tensor::CreateScalar<double>(data, tensor));
  } else if (data_schema_->column(col_num).type() == DataType::DE_FLOAT32) {
    float data = json_obj;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<float>(data, tensor));
    MS_LOG(INFO) << "float found: " << json_obj << ".";
  }
  return Status::OK();
}

// Loads a tensor with int value, we have to cast the value to type specified in the schema.
Status AlbumOp::LoadIntTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorPtr *tensor) {
  //  TensorPtr int_tensor;
  if (data_schema_->column(col_num).type() == DataType::DE_INT64) {
    int64_t data = json_obj;
    MS_LOG(INFO) << "int64 found: " << json_obj << ".";
    RETURN_IF_NOT_OK(Tensor::CreateScalar<int64_t>(data, tensor));
  } else if (data_schema_->column(col_num).type() == DataType::DE_INT32) {
    int32_t data = json_obj;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<int32_t>(data, tensor));
    MS_LOG(INFO) << "int32 found: " << json_obj << ".";
  }
  return Status::OK();
}

Status AlbumOp::LoadIntTensorRowByIndex(int index, bool is_array, const nlohmann::json &column_value,
                                        std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row) {
  int i = index;
  // int value
  if (!is_array &&
      (data_schema_->column(i).type() == DataType::DE_INT64 || data_schema_->column(i).type() == DataType::DE_INT32)) {
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadIntTensor(column_value, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }
  // int array
  if (is_array &&
      (data_schema_->column(i).type() == DataType::DE_INT64 || data_schema_->column(i).type() == DataType::DE_INT32)) {
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadIntArrayTensor(column_value, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }
  return Status::OK();
}

Status AlbumOp::LoadTensorRowByIndex(int index, const std::string &file, const nlohmann::json &js,
                                     std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row) {
  int i = index;
  // special case to handle
  if (data_schema_->column(i).name() == "id") {
    // id is internal, special case to load from file
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadIDTensor(file, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }
  // find if key does not exist, insert placeholder nullptr if not found
  if (js.find(data_schema_->column(i).name()) == js.end()) {
    // iterator not found, push nullptr as placeholder
    MS_LOG(INFO) << "Pushing empty tensor for column: " << data_schema_->column(i).name() << ".";
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadEmptyTensor(i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }
  nlohmann::json column_value = js.at(data_schema_->column(i).name());
  MS_LOG(INFO) << "This column is: " << data_schema_->column(i).name() << ".";
  bool is_array = column_value.is_array();
  // load single string
  if (column_value.is_string() && data_schema_->column(i).type() == DataType::DE_STRING) {
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadStringTensor(column_value, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }
  // load string array
  if (is_array && data_schema_->column(i).type() == DataType::DE_STRING) {
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadStringArrayTensor(column_value, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }
  // load image file
  if (column_value.is_string() && data_schema_->column(i).type() != DataType::DE_STRING) {
    std::string image_file_path = column_value;
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadImageTensor(image_file_path, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
    uint32_t orientation = GetOrientation(image_file_path);
    TensorPtr scalar_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<uint32_t>(orientation, &scalar_tensor));
    (*map_row)["orientation"] = scalar_tensor;
  }
  // load float value
  if (!is_array && (data_schema_->column(i).type() == DataType::DE_FLOAT32 ||
                    data_schema_->column(i).type() == DataType::DE_FLOAT64)) {
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadFloatTensor(column_value, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }
  // load float array
  if (is_array && (data_schema_->column(i).type() == DataType::DE_FLOAT32 ||
                   data_schema_->column(i).type() == DataType::DE_FLOAT64)) {
    TensorPtr tensor;
    RETURN_IF_NOT_OK(LoadFloatArrayTensor(column_value, i, &tensor));
    (*map_row)[data_schema_->column(i).name()] = tensor;
  }

  RETURN_IF_NOT_OK(LoadIntTensorRowByIndex(i, is_array, column_value, map_row));
  return Status::OK();
}

// Load 1 TensorRow (image,label) using 1 ImageColumns. 1 function call produces 1 TensorRow in a DataBuffer
// possible optimization: the helper functions of LoadTensorRow should be optimized
// to take a reference to a column descriptor?
// the design of this class is to make the code more readable, forgoing minor performance gain like
// getting rid of duplicated checks
Status AlbumOp::LoadTensorRow(row_id_type row_id, const std::string &file,
                              std::unordered_map<std::string, std::shared_ptr<Tensor>> *map_row) {
  // testing here is to just print out file path
  MS_LOG(INFO) << "Image row file: " << file << ".";

  std::ifstream file_handle(folder_path_ + file);
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open json file: " + folder_path_ + file);
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
        if (!IsReadColumn(data_schema_->column(i).name())) {
          continue;
        }
        RETURN_IF_NOT_OK(LoadTensorRowByIndex(i, file, js, map_row));
      }
    } catch (const std::exception &err) {
      file_handle.close();
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to parse json file: " + folder_path_ + file);
    }
  }
  file_handle.close();
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
