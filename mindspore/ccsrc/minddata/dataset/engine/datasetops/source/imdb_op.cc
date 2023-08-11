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

#include "minddata/dataset/engine/datasetops/source/imdb_op.h"

#include <fstream>
#include <unordered_set>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr int32_t kNumClasses = 2;

IMDBOp::IMDBOp(int32_t num_workers, const std::string &file_dir, int32_t queue_size, const std::string &usage,
               std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(std::move(file_dir)),
      usage_(usage),
      data_schema_(std::move(data_schema)),
      sampler_ind_(0) {}

Status IMDBOp::PrepareData() {
  std::vector<std::string> usage_list;
  if (usage_ == "all") {
    usage_list.push_back("train");
    usage_list.push_back("test");
  } else {
    usage_list.push_back(usage_);
  }
  std::vector<std::string> label_list = {"pos", "neg"};
  // get abs path for folder_path_
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, imdb dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, imdb dataset dir: " + folder_path_ + " does not exist.");
  }
  Path base_dir(realpath.value());
  for (auto usage : usage_list) {
    for (auto label : label_list) {
      Path dir = base_dir / usage / label;
      RETURN_IF_NOT_OK(GetDataByUsage(dir.ToString(), label));
    }
  }
  text_label_pairs_.shrink_to_fit();
  num_rows_ = text_label_pairs_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED("Invalid data, " + DatasetName(true) +
                             "Dataset API can't read the data file (interface mismatch or no data found). Check " +
                             DatasetName() + " file path: " + folder_path_);
  }
  return Status::OK();
}

// Load 1 TensorRow (text, label) using 1 std::pair<std::string, int32_t>. 1 function call produces 1 TensorTow
Status IMDBOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::pair<std::string, int32_t> pair_ptr = text_label_pairs_[row_id];
  std::shared_ptr<Tensor> text, label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(pair_ptr.second, &label));
  RETURN_IF_NOT_OK(LoadFile(pair_ptr.first, &text));

  (*trow) = TensorRow(row_id, {std::move(text), std::move(label)});
  trow->setPath({pair_ptr.first, std::string("")});
  return Status::OK();
}

void IMDBOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\n"
        << DatasetName(true) << " directory: " << folder_path_ << "\nUsage: " << usage_ << "\n\n";
  }
}

// Derived from RandomAccessOp
Status IMDBOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || text_label_pairs_.empty()) {
    if (text_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("Invalid dataset dir, " + DatasetName(true) +
                               "Dataset API can't read the data file (interface mismatch or no data found). Check " +
                               DatasetName() + " file path: " + folder_path_);
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR], Map containing text-index pair is nullptr or has been set in other place, "
        "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < text_label_pairs_.size(); ++i) {
    (*cls_ids)[text_label_pairs_[i].second].push_back(i);
  }
  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

Status IMDBOp::GetDataByUsage(const std::string &folder, const std::string &label) {
  Path dir_usage_label(folder);
  if (!dir_usage_label.Exists() || !dir_usage_label.IsDirectory()) {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, dataset dir may not exist or is not a directory: " + folder);
  }
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&dir_usage_label);
  CHECK_FAIL_RETURN_UNEXPECTED(dir_itr != nullptr,
                               "Invalid path, failed to open imdb dir: " + folder + ", permission denied.");
  std::map<std::string, int32_t> text_label_map;
  while (dir_itr->HasNext()) {
    Path file = dir_itr->Next();
    text_label_map[file.ToString()] = (label == "pos") ? 1 : 0;
  }
  for (auto item : text_label_map) {
    text_label_pairs_.emplace_back(std::make_pair(item.first, item.second));
  }
  return Status::OK();
}

Status IMDBOp::CountRows(const std::string &path, const std::string &usage, int64_t *num_rows) {
  RETURN_UNEXPECTED_IF_NULL(num_rows);
  // get abs path for folder_path_
  auto abs_path = FileUtils::GetRealPath(path.c_str());
  if (!abs_path.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, imdb dataset dir: " << path << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, imdb dataset dir: " + path + " does not exist.");
  }
  Path data_dir(abs_path.value());
  std::vector<std::string> all_dirs_list = {"pos", "neg"};
  std::vector<std::string> usage_list;
  if (usage == "all") {
    usage_list.push_back("train");
    usage_list.push_back("test");
  } else {
    usage_list.push_back(usage);
  }
  int64_t row_cnt = 0;
  for (int32_t ind = 0; ind < usage_list.size(); ++ind) {
    Path texts_dir_usage_path = data_dir / usage_list[ind];
    CHECK_FAIL_RETURN_UNEXPECTED(
      texts_dir_usage_path.Exists() && texts_dir_usage_path.IsDirectory(),
      "Invalid path, dataset path may not exist or is not a directory: " + texts_dir_usage_path.ToString());

    for (auto dir : all_dirs_list) {
      Path texts_dir_usage_dir_path((texts_dir_usage_path / dir).ToString());
      std::shared_ptr<Path::DirIterator> dir_iter = Path::DirIterator::OpenDirectory(&texts_dir_usage_dir_path);
      CHECK_FAIL_RETURN_UNEXPECTED(dir_iter != nullptr,
                                   "Invalid path, failed to open imdb dir: " + path + ", permission denied.");
      RETURN_UNEXPECTED_IF_NULL(dir_iter);
      while (dir_iter->HasNext()) {
        row_cnt++;
      }
    }
  }
  (*num_rows) = row_cnt;
  return Status::OK();
}

Status IMDBOp::ComputeColMap() {
  // Set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

// Get number of classes
Status IMDBOp::GetNumClasses(int64_t *num_classes) {
  RETURN_UNEXPECTED_IF_NULL(num_classes);
  *num_classes = kNumClasses;
  return Status::OK();
}

Status IMDBOp::LoadFile(const std::string &file, std::shared_ptr<Tensor> *out_row) {
  RETURN_UNEXPECTED_IF_NULL(out_row);

  std::ifstream handle(file, std::ios::in);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file);
  }

  std::string line;
  // IMDB just have a line for every txt.
  while (getline(handle, line)) {
    if (line.empty()) {
      continue;
    }
    auto rc = LoadTensor(line, out_row);
    if (rc.IsError()) {
      handle.close();
      return rc;
    }
  }
  handle.close();
  return Status::OK();
}

Status IMDBOp::LoadTensor(const std::string &line, std::shared_ptr<Tensor> *out_row) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(line, &tensor));
  *out_row = std::move(tensor);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
