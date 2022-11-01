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

#include "minddata/dataset/engine/datasetops/source/lfw_op.h"

#include <fstream>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
const char kImageFolder[] = "lfw";
const char kImageFolderFunneled[] = "lfw_funneled";
const char kImageFolderDeepFunneled[] = "lfw-deepfunneled";
const char kImageExtension[] = ".jpg";
const char kAnnotationDevTest[] = "DevTest";
const char kAnnotationDevTrain[] = "DevTrain";
const char kAnnotationNames[] = "lfw-names";
const char kAnnotationExtension[] = ".txt";
const int32_t kLFWFileNameLength = 4;

LFWOp::LFWOp(int32_t num_workers, const std::string &dataset_dir, const std::string &task, const std::string &usage,
             const std::string &image_set, int32_t queue_size, bool decode, std::unique_ptr<DataSchema> data_schema,
             const std::shared_ptr<SamplerRT> &sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(dataset_dir),
      task_(task),
      usage_(usage),
      image_set_(image_set),
      decode_(decode),
      data_schema_(std::move(data_schema)) {}

void LFWOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nLFW directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status LFWOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  Path path(real_folder_path_);
  std::string image_type_path = "";
  if (image_set_ == "original") {
    image_type_path = kImageFolder;
  } else if (image_set_ == "funneled") {
    image_type_path = kImageFolderFunneled;
  } else if (image_set_ == "deepfunneled") {
    image_type_path = kImageFolderDeepFunneled;
  }
  if (task_ == "people") {
    std::vector<std::string> image_id = image_label_people_[row_id];
    std::shared_ptr<Tensor> image, label;
    Path kImageFile = path / image_type_path / image_id[1] / (image_id[0] + kImageExtension);
    std::string name = image_id[1];
    RETURN_IF_NOT_OK(Tensor::CreateScalar(class_index_[name], &label));
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile.ToString(), data_schema_->Column(0), &image));
    (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
    trow->setPath({kImageFile.ToString(), std::string("")});
  } else if (task_ == "pairs") {
    std::vector<std::string> image_id = image_label_pair_[row_id];
    std::shared_ptr<Tensor> image1, image2, label;
    Path kImageFile1 = path / image_type_path / image_id[1] / (image_id[0] + kImageExtension);
    Path kImageFile2 = path / image_type_path / image_id[3] / (image_id[2] + kImageExtension);
    uint32_t match = std::atoi(image_id[4].c_str());
    RETURN_IF_NOT_OK(Tensor::CreateScalar(match, &label));
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile1.ToString(), data_schema_->Column(0), &image1));
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile2.ToString(), data_schema_->Column(1), &image2));
    (*trow) = TensorRow(row_id, {std::move(image1), std::move(image2), std::move(label)});
    trow->setPath({kImageFile1.ToString(), kImageFile2.ToString(), std::string("")});
  }
  return Status::OK();
}

Status LFWOp::GetClassIndexing() {
  Path path(real_folder_path_);
  Path lfw_names_file = path / (std::string(kAnnotationNames) + std::string(kAnnotationExtension));
  std::ifstream in_file;
  in_file.open(lfw_names_file.ToString());
  if (in_file.fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + path.ToString());
  }
  std::string lfw_names;
  uint32_t line = 0;
  while (getline(in_file, lfw_names)) {
    std::stringstream line_string(lfw_names);
    std::string line_result;
    std::vector<std::string> vector_string;
    while (line_string >> line_result) {
      vector_string.push_back(line_result);
    }
    class_index_[vector_string[0]] = line;
    line += 1;
  }
  in_file.close();
  return Status::OK();
}

Status LFWOp::ParsePeopleImageIds(const std::vector<std::vector<std::string>> &annotation_vector_string) {
  if (annotation_vector_string.empty()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, LFW annotation file is empty. Check file path: " + folder_path_);
  }
  uint32_t fold_nums, begin;
  if (usage_ == "10fold") {
    fold_nums = std::atoi(annotation_vector_string[0][0].c_str());
    begin = 1;
  } else {
    fold_nums = 1;
    begin = 0;
  }
  for (uint32_t i = 0; i < fold_nums; i++) {
    uint32_t n_lines = std::atoi(annotation_vector_string[begin][0].c_str());
    for (uint32_t j = begin + 1; j < begin + n_lines + 1; j++) {
      std::vector<std::string> vector_string = annotation_vector_string[j];
      std::string name = vector_string[0];
      uint32_t num = std::atoi(vector_string[1].c_str());
      std::string format_id;
      for (uint32_t k = 1; k < num + 1; k++) {
        format_id = "";
        std::string id = std::to_string(k);
        for (int32_t l = 0; l < kLFWFileNameLength - id.size(); ++l) {
          format_id = format_id + std::string("0");
        }
        std::string file_name = name + "_" + (format_id + id);
        std::vector<std::string> image_name;
        image_name.emplace_back(file_name);
        image_name.emplace_back(name);
        image_label_people_.emplace_back(image_name);
      }
    }
    begin += n_lines + 1;
  }
  num_rows_ = image_label_people_.size();
  return Status::OK();
}

Status LFWOp::ParsePairsImageIds(const std::vector<std::vector<std::string>> &annotation_vector_string) {
  if (annotation_vector_string.empty()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, LFW annotation file is empty. Check file path: " + folder_path_);
  }
  uint32_t fold_nums, pairs_nums, begin;
  if (usage_ == "10fold") {
    fold_nums = std::atoi(annotation_vector_string[0][0].c_str());
    pairs_nums = std::atoi(annotation_vector_string[0][1].c_str());
  } else {
    fold_nums = 1;
    pairs_nums = std::atoi(annotation_vector_string[0][0].c_str());
  }
  begin = 1;
  for (uint32_t i = 0; i < fold_nums; i++) {
    std::string first_format_id;
    std::string second_format_id;
    for (uint32_t j = begin; j < begin + pairs_nums; j++) {
      std::vector<std::string> vector_string = annotation_vector_string[j];
      std::string name = vector_string[0];
      std::string first_extension = vector_string[1];
      std::string second_extension = vector_string[2];
      first_format_id = "";
      second_format_id = "";
      for (int32_t l = 0; l < kLFWFileNameLength - first_extension.size(); ++l) {
        first_format_id = first_format_id + std::string("0");
      }
      for (int32_t l = 0; l < kLFWFileNameLength - second_extension.size(); ++l) {
        second_format_id = second_format_id + std::string("0");
      }
      std::string first_name = name + "_" + (first_format_id + first_extension);
      std::string second_name = name + "_" + (second_format_id + second_extension);
      std::vector<std::string> image_pair;
      image_pair.emplace_back(first_name);
      image_pair.emplace_back(name);
      image_pair.emplace_back(second_name);
      image_pair.emplace_back(name);
      image_pair.emplace_back("1");
      image_label_pair_.push_back(image_pair);
    }
    int multiple = 2;
    for (uint32_t k = begin + pairs_nums; k < begin + (multiple * pairs_nums); k++) {
      std::vector<std::string> vector_string = annotation_vector_string[k];
      std::string first = vector_string[0];
      std::string first_extension = vector_string[1];
      std::string second = vector_string[2];
      std::string second_extension = vector_string[3];
      first_format_id = "";
      second_format_id = "";
      for (int32_t l = 0; l < kLFWFileNameLength - first_extension.size(); ++l) {
        first_format_id = first_format_id + std::string("0");
      }
      for (int32_t l = 0; l < kLFWFileNameLength - second_extension.size(); ++l) {
        second_format_id = second_format_id + std::string("0");
      }
      std::string first_name = first + "_" + (first_format_id + first_extension);
      std::string second_name = second + "_" + (second_format_id + second_extension);
      std::vector<std::string> image_pair;
      image_pair.emplace_back(first_name);
      image_pair.emplace_back(first);
      image_pair.emplace_back(second_name);
      image_pair.emplace_back(second);
      image_pair.emplace_back("0");
      image_label_pair_.push_back(image_pair);
    }
    begin += (multiple * pairs_nums);
  }
  num_rows_ = image_label_pair_.size();
  return Status::OK();
}

std::vector<std::vector<std::string>> LFWOp::ReadFile(const std::string &annotation_file_path) const {
  std::ifstream in_file;
  in_file.open(annotation_file_path);
  if (in_file.fail()) {
    MS_LOG(ERROR) << "Invalid file, failed to open file: " + annotation_file_path;
  }
  std::string line;
  std::vector<std::vector<std::string>> annotation_vector_string;
  while (getline(in_file, line)) {
    std::stringstream line_string(line);
    std::string line_result;
    std::vector<std::string> vector_string;
    while (line_string >> line_result) {
      vector_string.push_back(line_result);
    }
    annotation_vector_string.emplace_back(vector_string);
  }
  in_file.close();
  return annotation_vector_string;
}

Status LFWOp::ParseImageIds() {
  Path path(real_folder_path_);
  std::vector<std::string> file_list;
  if (usage_ == "10fold") {
    file_list.push_back((path / (task_ + kAnnotationExtension)).ToString());
  }
  if (usage_ == "train" || usage_ == "all") {
    file_list.push_back(
      (path / (task_ + std::string(kAnnotationDevTrain) + std::string(kAnnotationExtension))).ToString());
  }
  if (usage_ == "test" || usage_ == "all") {
    file_list.push_back(
      (path / (task_ + std::string(kAnnotationDevTest) + std::string(kAnnotationExtension))).ToString());
  }
  for (auto file : file_list) {
    std::vector<std::vector<std::string>> annotation_vector_string = ReadFile(file);
    if (task_ == "people") {
      RETURN_IF_NOT_OK(ParsePeopleImageIds(annotation_vector_string));
    } else if (task_ == "pairs") {
      RETURN_IF_NOT_OK(ParsePairsImageIds(annotation_vector_string));
    } else {
      RETURN_STATUS_UNEXPECTED("Invalid parameter, task should be \"people\" or \"pairs\", got " + task_);
    }
  }
  return Status::OK();
}

Status LFWOp::PrepareData() {
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, LFW Dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, LFW Dataset dir: " + folder_path_ + " does not exist.");
  }
  real_folder_path_ = realpath.value();
  RETURN_IF_NOT_OK(this->ParseImageIds());
  RETURN_IF_NOT_OK(this->GetClassIndexing());
  return Status::OK();
}

Status LFWOp::ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(path, tensor));
  if (decode_) {
    Status rc = Decode(*tensor, tensor);
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("Invalid image, failed to decode " + path +
                               ": the image is damaged or permission denied.");
    }
  }
  return Status::OK();
}

Status LFWOp::CountTotalRows(int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  RETURN_IF_NOT_OK(PrepareData());
  if (task_ == "people") {
    *count = static_cast<int64_t>(image_label_people_.size());
  } else if (task_ == "pairs") {
    *count = static_cast<int64_t>(image_label_pair_.size());
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, task should be \"people\" or \"pairs\", got " + task_);
  }
  return Status::OK();
}

Status LFWOp::ComputeColMap() {
  // Set the column name map (base class field).
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
