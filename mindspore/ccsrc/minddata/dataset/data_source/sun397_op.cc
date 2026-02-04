/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/data_source/sun397_op.h"

#include <fstream>
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
constexpr char kCategoriesMeta[] = "ClassName.txt";

SUN397Op::SUN397Op(const std::string &file_dir, bool decode, int32_t num_workers, int32_t queue_size,
                   std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(file_dir),
      decode_(decode),
      buf_cnt_(0),
      categorie2id_({}),
      image_path_label_pairs_({}),
      data_schema_(std::move(data_schema)) {}

Status SUN397Op::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  auto file_path = image_path_label_pairs_[row_id].first;
  auto label_num = image_path_label_pairs_[row_id].second;

  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(label_num, &label));
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(file_path, &image));

  if (decode_) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Invalid image, " + file_path + " decode failed, the image is broken or permission denied.";
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({file_path, std::string("")});
  return Status::OK();
}

void SUN397Op::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nSUN397 directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

// Derived from RandomAccessOp.
Status SUN397Op::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
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

Status SUN397Op::GetFileContent(const std::string &info_file, std::string *ans) {
  RETURN_UNEXPECTED_IF_NULL(ans);
  std::ifstream reader;
  reader.open(info_file, std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(
    !reader.fail(), "Invalid file, failed to open " + info_file + ": SUN397 file is damaged or permission denied.");
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

Status SUN397Op::LoadCategories(const std::string &category_meta_name) {
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
  uint32_t label = 0;
  std::size_t pos = 0;
  while ((pos = s.find(" ")) != std::string::npos) {
    CHECK_FAIL_RETURN_UNEXPECTED(pos + 1 <= s.size(), "Invalid data, Reading SUN397 category file failed: " +
                                                        category_meta_name + ", space characters not found.");
    category = get_splited_str(pos);
    CHECK_FAIL_RETURN_UNEXPECTED(!category.empty(), "Invalid data, Reading SUN397 category file failed: " +
                                                      category_meta_name + ", space characters not found.");
    categorie2id_.insert({category, label});
    label++;
  }
  return Status::OK();
}

Status SUN397Op::PrepareData() {
  auto real_folder_path = FileUtils::GetRealPath(folder_path_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(real_folder_path.has_value(), "Invalid file path, " + folder_path_ + " does not exist.");

  RETURN_IF_NOT_OK(LoadCategories((Path(real_folder_path.value()) / Path(kCategoriesMeta)).ToString()));
  image_path_label_pairs_.clear();
  for (auto c2i : categorie2id_) {
    std::string folder_name = c2i.first;
    uint32_t label = c2i.second;

    Path folder(folder_path_ + folder_name);
    std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&folder);
    if (!folder.Exists() || dirItr == nullptr) {
      RETURN_STATUS_UNEXPECTED("Invalid path, " + folder_name + " does not exist or permission denied.");
    }
    std::set<std::string> imgs;  // use this for ordering
    auto dirname_offset = folder.ToString().size();
    while (dirItr->HasNext()) {
      Path file = dirItr->Next();
      if (file.Extension() == ".jpg") {
        auto file_str = file.ToString();
        if (file_str.substr(dirname_offset + 1).find("sun_") == 0) {
          (void)imgs.insert(file_str);
        }
      } else {
        MS_LOG(WARNING) << "SUN397Dataset unsupported file found: " << file.ToString()
                        << ", extension: " << file.Extension() << ".";
      }
    }
    for (const std::string &img : imgs) {
      image_path_label_pairs_.push_back({img, label});
    }
  }
  num_rows_ = image_path_label_pairs_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0,
    "Invalid data, no valid data matching the dataset API SUN397Dataset. Please check dataset API or file path: " +
      folder_path_ + ".");
  return Status::OK();
}

Status SUN397Op::CountTotalRows(const std::string &dir, bool decode, int64_t *count) {
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
  auto op =
    std::make_shared<SUN397Op>(dir, decode, num_workers, op_connect_size, std::move(schema), std::move(sampler));
  RETURN_IF_NOT_OK(op->PrepareData());

  *count = op->image_path_label_pairs_.size();
  return Status::OK();
}

Status SUN397Op::ComputeColMap() {
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
