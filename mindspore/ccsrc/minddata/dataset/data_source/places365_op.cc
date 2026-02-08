/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/data_source/places365_op.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>

#include "include/utils/common.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/data_source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr char kCategoriesMeta[] = "categories_places365.txt";
const std::map<std::string, std::string> kFileListMeta = {{"train-standard", "places365_train_standard.txt"},
                                                          {"train-challenge", "places365_train_challenge.txt"},
                                                          {"val", "places365_val.txt"}};
const std::map<std::pair<std::string, bool>, std::string> kImagesMeta = {
  {std::pair<std::string, bool>("train-standard", false), "data_large_standard"},
  {std::pair<std::string, bool>("train-challenge", false), "data_large_challenge"},
  {std::pair<std::string, bool>("val", false), "val_large"},
  {std::pair<std::string, bool>("train-standard", true), "data_256_standard"},
  {std::pair<std::string, bool>("train-challenge", true), "data_256_challenge"},
  {std::pair<std::string, bool>("val", true), "val_256"},
};

Places365Op::Places365Op(const std::string &root, const std::string &usage, bool small, bool decode,
                         int32_t num_workers, int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
                         std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      root_(root),
      usage_(usage),
      small_(small),
      decode_(decode),
      buf_cnt_(0),
      categorie2id_({}),
      image_path_label_pairs_({}),
      data_schema_(std::move(data_schema)) {}

Status Places365Op::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::shared_ptr<Tensor> image, label;
  // make a copy of cached tensor.
  RETURN_IF_NOT_OK(GetPlaces365DataTensor(row_id, &image));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(image_path_label_pairs_[row_id].second, &label));

  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  return Status::OK();
}

void Places365Op::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nPlaces365 directory: " << root_ << "\nUsage: " << usage_
        << "\nSmall: " << (small_ ? "yes" : "no") << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

// Derived from RandomAccessOp.
Status Places365Op::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_path_label_pairs_.empty()) {
    if (image_path_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("No image found in dataset. Check if image was read successfully.");
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR] Map for containing image-index pair is nullptr or has been set in other place,"
        "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < image_path_label_pairs_.size(); ++i) {
    (*cls_ids)[image_path_label_pairs_[i].second].push_back(i);
  }
  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

Status Places365Op::GetFileContent(const std::string &info_file, std::string *ans) {
  RETURN_UNEXPECTED_IF_NULL(ans);
  std::ifstream reader;
  reader.open(info_file, std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(
    !reader.fail(), "Invalid file, failed to open " + info_file + ": Places365 file is damaged or permission denied.");
  reader.seekg(0, std::ios::end);
  std::size_t size = reader.tellg();
  reader.seekg(0, std::ios::beg);

  CHECK_FAIL_RETURN_UNEXPECTED(size > 0, "Invalid file, the file size of " + info_file + " is unexpected, got size 0.");
  std::string buffer(size, ' ');
  reader.read(&buffer[0], size);
  reader.close();

  // remove \n character in the buffer.
  std::regex pattern("([\\s\\n]+)");
  std::string fmt = " ";
  std::string s = std::regex_replace(buffer, pattern, fmt);

  // remove the head and tail whiteblanks of the s.
  s.erase(0, s.find_first_not_of(" "));
  s.erase(s.find_last_not_of(" ") + 1);
  // append one whiteblanks to the end of s.
  s += " ";
  *ans = s;
  return Status::OK();
}

Status Places365Op::LoadCategories(const std::string &category_meta_name) {
  categorie2id_.clear();
  std::string s;
  RETURN_IF_NOT_OK(GetFileContent(category_meta_name, &s));
  auto get_splited_str = [&s, &category_meta_name](std::size_t pos) {
    std::string item = s.substr(0, pos);
    // If pos+1 is equal to the string length, the function returns an empty string.
    s = s.substr(pos + 1);
    return item;
  };

  std::string category;
  uint32_t label;
  // Category meta info is read into string s in the format: "category1 label1 category2 label2 category3 label3 ...".
  // Use blank space delimiter to split the string and process each substring.
  // Like state matching, the type of each substring needs to be switched.
  enum ColType { CATEGORY, LABEL };
  std::size_t pos = 0;
  ColType col_idx = CATEGORY;
  while ((pos = s.find(" ")) != std::string::npos) {
    switch (col_idx) {
      case CATEGORY: {
        CHECK_FAIL_RETURN_UNEXPECTED(pos + 1 <= s.size(), "Invalid data, Reading places365 category file failed: " +
                                                            category_meta_name + ", space characters not found.");
        category = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!category.empty(), "Invalid data, Reading places365 category file failed: " +
                                                          category_meta_name + ", space characters not found.");
        // switch the type of substring.
        col_idx = LABEL;
        break;
      }
      case LABEL: {
        CHECK_FAIL_RETURN_UNEXPECTED(pos + 1 <= s.size(), "Invalid data, Reading places365 category file failed: " +
                                                            category_meta_name + ", space characters not found.");
        std::string label_item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!label_item.empty(), "Invalid data, Reading places365 category file failed: " +
                                                            category_meta_name + ", space characters not found.");
        label = std::atoi(label_item.c_str());
        // switch the type of substring.
        col_idx = CATEGORY;
        categorie2id_.insert({category, label});
        break;
      }
      default: {
        break;
      }
    }
  }
  return Status::OK();
}

Status Places365Op::LoadFileLists(const std::string &filelists_meta_name) {
  std::string folder_path = (Path(root_) / Path(kImagesMeta.at(std::make_pair(usage_, small_)))).ToString();
  image_path_label_pairs_.clear();

  std::string s;
  RETURN_IF_NOT_OK(GetFileContent(filelists_meta_name, &s));
  auto get_splited_str = [&s, &filelists_meta_name](std::size_t pos) {
    std::string item = s.substr(0, pos);
    s = s.substr(pos + 1);
    return item;
  };
  std::string path;
  uint32_t label;
  // Category meta info is read into string s in the format: "path1 label1 path2 label2 path2 label3 ...".
  // Use blank space delimiter to split the string and process each substring.
  // Like state matching, the type of each substring needs to be switched.
  enum ColType { PATH, LABEL };
  std::size_t pos = 0;
  ColType col_idx = PATH;
  while ((pos = s.find(" ")) != std::string::npos) {
    switch (col_idx) {
      case PATH: {
        CHECK_FAIL_RETURN_UNEXPECTED(pos + 1 <= s.size(), "Invalid data, Reading places365 category file failed: " +
                                                            filelists_meta_name + ", space characters not found.");
        path = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!path.empty(), "Invalid data, Reading places365 filelist file failed: " +
                                                      filelists_meta_name + ", space characters not found.");
        // switch the type of substring.
        col_idx = LABEL;
        break;
      }
      case LABEL: {
        CHECK_FAIL_RETURN_UNEXPECTED(pos + 1 <= s.size(), "Invalid data, Reading places365 category file failed: " +
                                                            filelists_meta_name + ", space characters not found.");
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(), "Invalid data, Reading places365 filelist file failed: " +
                                                      filelists_meta_name + ", space characters not found.");
        label = std::atoi(item.c_str());
        // switch the type of substring.
        col_idx = PATH;
        image_path_label_pairs_.push_back({(Path(folder_path) / Path(path)).ToString(), label});
        break;
      }
      default: {
        break;
      }
    }
  }
  return Status::OK();
}

Status Places365Op::GetPlaces365DataTensor(uint32_t index, std::shared_ptr<Tensor> *image_tensor) {
  std::string file_path = image_path_label_pairs_[index].first;
  CHECK_FAIL_RETURN_UNEXPECTED(Path(file_path).Exists(),
                               "Invalid file path, Places365 image: " + file_path + " does not exists.");
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(file_path, image_tensor));
  if (decode_) {
    Status rc = Decode(*image_tensor, image_tensor);
    if (rc.IsError()) {
      *image_tensor = nullptr;
      std::string err_msg =
        "Invalid image, failed to decode " + file_path + ": the image is damaged or permission denied.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  return Status::OK();
}

Status Places365Op::PrepareData() {
  auto real_folder_path = FileUtils::GetRealPath(root_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(real_folder_path.has_value(), "Invalid file path, " + root_ + " does not exist.");

  RETURN_IF_NOT_OK(LoadCategories((Path(real_folder_path.value()) / Path(kCategoriesMeta)).ToString()));
  RETURN_IF_NOT_OK(LoadFileLists((Path(real_folder_path.value()) / Path(kFileListMeta.at(usage_))).ToString()));
  num_rows_ = image_path_label_pairs_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0,
    "Invalid data, no valid data matching the dataset API Places365Dataset. Please check dataset API or file path: " +
      root_ + ".");
  return Status::OK();
}

Status Places365Op::CountTotalRows(const std::string &dir, const std::string &usage, const bool small,
                                   const bool decode, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op = std::make_shared<Places365Op>(dir, usage, small, decode, num_workers, op_connect_size, std::move(schema),
                                          std::move(sampler));
  RETURN_IF_NOT_OK(op->PrepareData());

  for (size_t i = 0; i < op->image_path_label_pairs_.size(); ++i) {
    CHECK_FAIL_RETURN_UNEXPECTED(Path(op->image_path_label_pairs_[i].first).Exists(),
                                 "Invalid file path, " + op->image_path_label_pairs_[i].first + " does not exists.");
  }
  *count = op->image_path_label_pairs_.size();
  return Status::OK();
}

Status Places365Op::ComputeColMap() {
  // set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
