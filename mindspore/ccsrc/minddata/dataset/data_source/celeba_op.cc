/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/data_source/celeba_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/data_source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/path.h"
#include "utils/file_utils.h"
#include "minddata/dataset/vision/kernels/image_utils.h"

namespace mindspore {
namespace dataset {
#ifdef ENABLE_PYTHON
CelebAOp::CelebAOp(int32_t num_workers, const std::string &dir, int32_t queue_size, bool decode,
                   const std::string &usage, const std::set<std::string> &exts, std::unique_ptr<DataSchema> schema,
                   std::shared_ptr<SamplerRT> sampler, const py::function &decrypt)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(dir),
      decode_(decode),
      extensions_(exts),
      data_schema_(std::move(schema)),
      num_rows_in_attr_file_(0),
      attr_file_(""),
      usage_(usage) {
  if (Py_IsInitialized() != 0) {
    py::gil_scoped_acquire gil_acquire;
    decrypt_ = decrypt;
  }
  attr_info_queue_ = std::make_unique<Queue<std::vector<std::string>>>(queue_size);
}

CelebAOp::~CelebAOp() {
  if (Py_IsInitialized() != 0) {
    py::gil_scoped_acquire gil_acquire;
    decrypt_ = py::object();
  }
}
#else
CelebAOp::CelebAOp(int32_t num_workers, const std::string &dir, int32_t queue_size, bool decode,
                   const std::string &usage, const std::set<std::string> &exts, std::unique_ptr<DataSchema> schema,
                   std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(dir),
      decode_(decode),
      extensions_(exts),
      data_schema_(std::move(schema)),
      num_rows_in_attr_file_(0),
      attr_file_(""),
      usage_(usage) {
  attr_info_queue_ = std::make_unique<Queue<std::vector<std::string>>>(queue_size);
}

CelebAOp::~CelebAOp() = default;
#endif

Status CelebAOp::RegisterAndLaunchThreads() {
  RETURN_IF_NOT_OK(ParallelOp::RegisterAndLaunchThreads());
  RETURN_IF_NOT_OK(attr_info_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->AllTasks()->CreateAsyncTask("Walking attr file", std::bind(&CelebAOp::ParseAttrFile, this), nullptr, id()));
  return Status::OK();
}

Status CelebAOp::ParseAttrFile() {
  TaskManager::FindMe()->Post();
  Path folder_path(folder_path_);

  auto realpath = FileUtils::GetRealPath((folder_path / "list_attr_celeba.txt").ToString().c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << (folder_path / "list_attr_celeba.txt").ToString() << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + (folder_path / "list_attr_celeba.txt").ToString() +
                             " does not exist.");
  }

  std::ifstream attr_file(realpath.value(), std::ios::in);
  if (!attr_file.is_open()) {
    std::string attr_file_name = (folder_path / "list_attr_celeba.txt").ToString();
    RETURN_STATUS_ERROR(StatusCode::kMDFileNotExist,
                        "Invalid attr file, failed to open: " + attr_file_name + ", permission denied.");
  }

  attr_file_ = (folder_path / "list_attr_celeba.txt").ToString();
  const auto PushBackToQueue = [this](std::vector<std::string> &vec, std::ifstream &attr_file,
                                      std::ifstream &partition_file) {
    Status s = attr_info_queue_->EmplaceBack(vec);
    if (s.IsError()) {
      CLOSE_FILE(attr_file, partition_file);
      return s;
    }
    return Status::OK();
  };

  std::string rows_num;
  std::string attr_name;
  (void)getline(attr_file, rows_num);
  try {
    num_rows_in_attr_file_ = static_cast<int64_t>(std::stoul(rows_num));  // First line is rows number in attr file
  } catch (std::invalid_argument &e) {
    CLOSE_FILE(attr_file, partition_file_);
    RETURN_STATUS_UNEXPECTED("Invalid rows_num, failed to convert rows_num: " + rows_num + " to unsigned long in " +
                             attr_file_ + ".");
  } catch (std::out_of_range &e) {
    CLOSE_FILE(attr_file, partition_file_);
    RETURN_STATUS_UNEXPECTED("Invalid rows_num, rows_num in " + attr_file_ + " is out of range, rows_num is " +
                             rows_num + ".");
  }

  (void)getline(attr_file, attr_name);  // Second line is attribute name,ignore it
  std::string image_info;
  std::vector<std::string> image_infos;
  image_infos.reserve(oc_queue_size_);
  while (getline(attr_file, image_info)) {
    if ((image_info.empty()) || (usage_ != "all" && !CheckDatasetTypeValid())) {
      continue;
    }
    image_infos.push_back(image_info);
    if (image_info.size() % oc_queue_size_ == 0) {
      RETURN_IF_NOT_OK(PushBackToQueue(image_infos, attr_file, partition_file_));
      image_infos.clear();
    }
  }
  if (!image_infos.empty()) {
    RETURN_IF_NOT_OK(PushBackToQueue(image_infos, attr_file, partition_file_));
  }
  std::vector<std::string> end_indicator = std::vector<std::string>(0);
  RETURN_IF_NOT_OK(PushBackToQueue(end_indicator, attr_file, partition_file_));  // end indicator
  CLOSE_FILE(attr_file, partition_file_);
  return Status::OK();
}

bool CelebAOp::CheckDatasetTypeValid() {
  if (!partition_file_.is_open()) {
    Path folder_path(folder_path_);
    partition_file_.open((folder_path / "list_eval_partition.txt").ToString());
    if (!partition_file_.is_open()) {
      MS_LOG(ERROR) << "Invalid eval partition file, failed to open eval partition file: "
                    << (folder_path / "list_eval_partition.txt").ToString() << " does not exist or permission denied.";
      return false;
    }
  }
  std::string line;
  (void)getline(partition_file_, line);
  std::vector<std::string> vec = Split(line);
  if (vec.size() != 2) {
    return false;
  }
  int32_t type;
  try {
    type = std::stoi(vec[1]);
  } catch (std::invalid_argument &e) {
    MS_LOG(WARNING) << "Invalid number, the second word in list_eval_partition.txt should be numeric, but got: "
                    << vec[1] << ".";
    return false;
  } catch (std::out_of_range &e) {
    MS_LOG(WARNING) << "Invalid number, the second word in list_eval_partition.txt is out of range, word is: " << vec[1]
                    << ".";
    return false;
  }
  // train:0, valid=1, test=2
  constexpr int32_t train_type = 0;
  constexpr int32_t valid_type = 1;
  constexpr int32_t test_type = 2;

  if (usage_ == "train" && (type == train_type)) {
    return true;
  } else if (usage_ == "valid" && (type == valid_type)) {
    return true;
  } else if (usage_ == "test" && (type == test_type)) {
    return true;
  }

  return false;
}

Status CelebAOp::PrepareData() {
  std::vector<std::string> image_infos;
  bool need_more_data = true;
  RETURN_IF_NOT_OK(attr_info_queue_->PopFront(&image_infos));
  while (!image_infos.empty() && need_more_data) {
    for (uint32_t index = 0; index < image_infos.size(); index++) {
      std::string image_info = image_infos[index];
      std::vector<std::string> split = Split(image_info);
      std::pair<std::string, std::vector<int32_t>> image_labels;

      Path path(folder_path_);
      Path file_path = path / split[0];
      if (!extensions_.empty() && extensions_.find(file_path.Extension()) == extensions_.end()) {
        MS_LOG(WARNING) << "Unsupported file found at " << file_path.ToString().c_str() << ", its extension is "
                        << file_path.Extension().c_str() << ".";
        continue;
      }
      image_labels.first = split[0];
      for (uint32_t label_index = 1; label_index < split.size(); label_index++) {
        int32_t value;
        try {
          value = std::stoi(split[label_index]);
        } catch (std::invalid_argument &e) {
          RETURN_STATUS_UNEXPECTED("Invalid label index, the label index in " + file_path.ToString() +
                                   " should be numeric, but got: " + split[label_index] + ".");
        } catch (std::out_of_range &e) {
          RETURN_STATUS_UNEXPECTED("Invalid label index, the label index in " + file_path.ToString() +
                                   " is out of range, index is " + split[label_index] + ".");
        }
        image_labels.second.push_back(value);
      }

      image_labels_vec_.push_back(image_labels);
    }

    RETURN_IF_NOT_OK(attr_info_queue_->PopFront(&image_infos));
  }

  num_rows_ = image_labels_vec_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, CelebADataset API can't read the data file (interface mismatch or no data found). "
      "Check file path: " +
      folder_path_);
  }
  MS_LOG(DEBUG) << "Celeba dataset rows number is " << num_rows_ << ".";
  return Status::OK();
}

std::vector<std::string> CelebAOp::Split(const std::string &line) {
  std::string str = line;
  std::string::size_type pos;
  std::vector<std::string> split;
  str += " ";
  int size = str.size();
  for (uint32_t index = 0; index < size;) {
    pos = str.find(" ", index);
    if (pos != index) {  // skip space
      std::string s = str.substr(index, pos - index);
      split.push_back(s);
    }
    index = pos + 1;
  }

  return split;
}

Status CelebAOp::LoadTensorRow(row_id_type row_id, TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  std::pair<std::string, std::vector<int32_t>> &image_label = image_labels_vec_[row_id];
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> label;

  Path path(folder_path_);
  Path image_path = path / image_label.first;

#ifdef ENABLE_PYTHON
  RETURN_IF_NOT_OK(MappableLeafOp::ImageDecrypt(image_path.ToString(), &image, decrypt_));
#else
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_path.ToString(), &image));
#endif
  if (decode_ == true) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      image = nullptr;
      std::string err_msg =
        "Invalid image, " + image_path.ToString() + " decode failed, the image is broken or permission denied.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  RETURN_IF_NOT_OK(
    Tensor::CreateEmpty(TensorShape({1, (uint32_t)image_label.second.size()}), data_schema_->Column(1).Type(), &label));
  RETURN_IF_NOT_OK(label->Zero());
  for (uint32_t index = 0; index < image_label.second.size(); index++) {
    if (image_label.second[index] == 1) {
      RETURN_IF_NOT_OK(label->SetItemAt<uint32_t>({0, static_cast<dsize_t>(index)}, 1));
    } else {
      RETURN_IF_NOT_OK(label->SetItemAt<uint32_t>({0, static_cast<dsize_t>(index)}, 0));
    }
  }
  label->Squeeze();

  (*row) = TensorRow(row_id, {std::move(image), std::move(label)});
  // Add file path info
  row->setPath({image_path.ToString(), attr_file_});
  return Status::OK();
}

void CelebAOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nceleba dir: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status CelebAOp::ComputeColMap() {
  // Set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t index = 0; index < data_schema_->NumColumns(); index++) {
      column_name_id_map_[data_schema_->Column(index).Name()] = index;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status CelebAOp::InitPullMode() {
  if (!image_labels_vec_.empty()) {
    return Status::OK();
  }
  if (attr_info_queue_->empty()) {
    RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask("Walking attr file", std::bind(&CelebAOp::ParseAttrFile, this),
                                                        nullptr, id()));
  }
  return PrepareData();
}

}  // namespace dataset
}  // namespace mindspore
