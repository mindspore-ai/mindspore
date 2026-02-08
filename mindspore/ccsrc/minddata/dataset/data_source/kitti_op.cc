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

#include "minddata/dataset/data_source/kitti_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>

#include "include/utils/common.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/data_source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr int kLabelNameIndex = 0;
constexpr int kTruncatedIndex = 1;
constexpr int kOccludedIndex = 2;
constexpr int kAlphaIndex = 3;
constexpr int kXMinIndex = 4;
constexpr int kYMinIndex = 5;
constexpr int kXMaxIndex = 6;
constexpr int kYMaxIndex = 7;
constexpr int kFirstDimensionIndex = 8;
constexpr int kSecondDimensionIndex = 9;
constexpr int kThirdDimensionIndex = 10;
constexpr int kFirstLocationIndex = 11;
constexpr int kSecondLocationIndex = 12;
constexpr int kThirdLocationIndex = 13;
constexpr int kRotationYIndex = 14;
constexpr int kTotalParamNums = 14;
const char kImagesFolder[] = "data_object_image_2";
const char kAnnotationsFolder[] = "data_object_label_2";
const char kImageExtension[] = ".png";
const char kAnnotationExtension[] = ".txt";
const int32_t kKittiFileNameLength = 6;

KITTIOp::KITTIOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers, int32_t queue_size,
                 bool decode, std::unique_ptr<DataSchema> data_schema, const std::shared_ptr<SamplerRT> &sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      decode_(decode),
      usage_(usage),
      folder_path_(dataset_dir),
      data_schema_(std::move(data_schema)) {}

void KITTIOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nKITTI directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status KITTIOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::string image_id = image_ids_[row_id];
  std::shared_ptr<Tensor> image;
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, KITTI Dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, KITTI Dataset dir: " + folder_path_ + " does not exist.");
  }
  Path path(realpath.value());
  if (usage_ == "train") {
    TensorRow annotation;
    Path kImageFile = path / kImagesFolder / "training" / "image_2" / (image_id + kImageExtension);
    Path kAnnotationFile = path / kAnnotationsFolder / "training" / "label_2" / (image_id + kAnnotationExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile.ToString(), data_schema_->Column(0), &image));
    RETURN_IF_NOT_OK(ReadAnnotationToTensor(kAnnotationFile.ToString(), &annotation));
    trow->setId(row_id);
    trow->setPath({kImageFile.ToString(), kAnnotationFile.ToString(), kAnnotationFile.ToString(),
                   kAnnotationFile.ToString(), kAnnotationFile.ToString(), kAnnotationFile.ToString(),
                   kAnnotationFile.ToString(), kAnnotationFile.ToString(), kAnnotationFile.ToString()});
    trow->push_back(std::move(image));
    trow->insert(trow->end(), annotation.begin(), annotation.end());
  } else if (usage_ == "test") {
    Path kImageFile = path / kImagesFolder / "testing" / "image_2" / (image_id + kImageExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile.ToString(), data_schema_->Column(0), &image));
    trow->setId(row_id);
    trow->setPath({kImageFile.ToString()});
    trow->push_back(std::move(image));
  }
  return Status::OK();
}

Status KITTIOp::ParseImageIds() {
  if (!image_ids_.empty()) {
    return Status::OK();
  }
  auto folder_realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!folder_realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, KITTI Dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, KITTI Dataset dir: " + folder_path_ + " does not exist.");
  }
  Path path(folder_realpath.value());
  Path image_sets_file("");
  if (usage_ == "train") {
    image_sets_file = path / kImagesFolder / "training" / "image_2";
  } else if (usage_ == "test") {
    image_sets_file = path / kImagesFolder / "testing" / "image_2";
  }
  std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&image_sets_file);
  CHECK_FAIL_RETURN_UNEXPECTED(dirItr != nullptr, "Invalid path, failed to open KITTI image dir: " +
                                                    image_sets_file.ToString() + ", permission denied.");
  int32_t total_image_size = 0;
  while (dirItr->HasNext()) {
    total_image_size++;
  }
  std::string format_id;
  for (int32_t i = 0; i < total_image_size; ++i) {
    format_id = "";
    std::string id = std::to_string(i);
    for (int32_t j = 0; j < kKittiFileNameLength - id.size(); ++j) {
      format_id = format_id + std::string("0");
    }
    image_ids_.push_back(format_id + id);
  }
  image_ids_.shrink_to_fit();
  num_rows_ = image_ids_.size();
  return Status::OK();
}

Status KITTIOp::ParseAnnotationIds() {
  std::vector<std::string> new_image_ids;
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, KITTI Dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, KITTI Dataset dir: " + folder_path_ + " does not exist.");
  }
  Path path(realpath.value());
  for (auto id : image_ids_) {
    Path kAnnotationName = path / kAnnotationsFolder / "training" / "label_2" / (id + kAnnotationExtension);
    RETURN_IF_NOT_OK(ParseAnnotationBbox(kAnnotationName.ToString()));
    if (annotation_map_.find(kAnnotationName.ToString()) != annotation_map_.end()) {
      new_image_ids.push_back(id);
    }
  }
  if (image_ids_.size() != new_image_ids.size()) {
    image_ids_.clear();
    image_ids_.insert(image_ids_.end(), new_image_ids.begin(), new_image_ids.end());
  }
  uint32_t count = 0;
  for (auto &label : label_index_) {
    label.second = count++;
  }
  num_rows_ = image_ids_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API KITTIDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

Status KITTIOp::ParseAnnotationBbox(const std::string &path) {
  CHECK_FAIL_RETURN_UNEXPECTED(Path(path).Exists(), "Invalid path, " + path + " does not exist.");
  Annotation annotation;
  std::ifstream in_file;
  in_file.open(path, std::ifstream::in);
  if (in_file.fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + path);
  }
  std::string anno;
  while (getline(in_file, anno)) {
    std::string label_name;
    std::string line_result;
    std::vector<std::string> vector_string;
    float truncated = 0.0, occluded = 0.0, alpha = 0.0, xmin = 0.0, ymin = 0.0, xmax = 0.0, ymax = 0.0,
          first_dimension = 0.0, second_dimension = 0.0, third_dimension = 0.0, first_location = 0.0,
          second_location = 0.0, third_location = 0.0, rotation_y = 0.0;
    std::stringstream line(anno);
    while (line >> line_result) {
      vector_string.push_back(line_result);
    }
    label_name = vector_string[kLabelNameIndex];
    truncated = std::atof(vector_string[kTruncatedIndex].c_str());
    occluded = std::atof(vector_string[kOccludedIndex].c_str());
    alpha = std::atof(vector_string[kAlphaIndex].c_str());
    xmin = std::atof(vector_string[kXMinIndex].c_str());
    ymin = std::atof(vector_string[kYMinIndex].c_str());
    xmax = std::atof(vector_string[kXMaxIndex].c_str());
    ymax = std::atof(vector_string[kYMaxIndex].c_str());
    first_dimension = std::atof(vector_string[kFirstDimensionIndex].c_str());
    second_dimension = std::atof(vector_string[kSecondDimensionIndex].c_str());
    third_dimension = std::atof(vector_string[kThirdDimensionIndex].c_str());
    first_location = std::atof(vector_string[kFirstLocationIndex].c_str());
    second_location = std::atof(vector_string[kSecondLocationIndex].c_str());
    third_location = std::atof(vector_string[kThirdLocationIndex].c_str());
    rotation_y = std::atof(vector_string[kRotationYIndex].c_str());
    if (label_name != "" || (xmin > 0 && ymin > 0 && xmax > xmin && ymax > ymin)) {
      std::vector<float> bbox_list = {truncated,
                                      occluded,
                                      alpha,
                                      xmin,
                                      ymin,
                                      xmax,
                                      ymax,
                                      first_dimension,
                                      second_dimension,
                                      third_dimension,
                                      first_location,
                                      second_location,
                                      third_location,
                                      rotation_y};
      annotation.emplace_back(std::make_pair(label_name, bbox_list));
      label_index_[label_name] = 0;
    }
  }
  in_file.close();
  if (annotation.size() > 0) {
    annotation_map_[path] = annotation;
  }
  return Status::OK();
}

Status KITTIOp::PrepareData() {
  RETURN_IF_NOT_OK(this->ParseImageIds());
  if (usage_ == "train") {
    RETURN_IF_NOT_OK(this->ParseAnnotationIds());
  }
  return Status::OK();
}

Status KITTIOp::ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(path, tensor));
  if (decode_) {
    Status rc = Decode(*tensor, tensor);
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("Invalid data, failed to decode image: " + path);
    }
  }
  return Status::OK();
}

Status KITTIOp::ReadAnnotationToTensor(const std::string &path, TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  Annotation annotation = annotation_map_[path];
  std::shared_ptr<Tensor> bbox, alpha, dimensions, location, occuluded, rotation_y, truncated, label;
  std::vector<float> bbox_data, alpha_data, dimensions_data, location_data, rotation_y_data, truncated_data;
  std::vector<uint32_t> occuluded_data;
  std::vector<uint32_t> label_data;
  dsize_t bbox_num = 0;
  for (auto item : annotation) {
    if (label_index_.find(item.first) != label_index_.end()) {
      label_data.push_back(static_cast<uint32_t>(label_index_[item.first]));
      CHECK_FAIL_RETURN_UNEXPECTED(item.second.size() == kTotalParamNums,
                                   "Invalid file, the format of the annotation file is not as expected, got " +
                                     std::to_string(item.second.size()) + " parameters.");

      std::vector<float> tmp_bbox = {(item.second)[3], (item.second)[4], (item.second)[5], (item.second)[6]};
      std::vector<float> tmp_dimensions = {(item.second)[7], (item.second)[8], (item.second)[9]};
      std::vector<float> tmp_location = {(item.second)[10], (item.second)[11], (item.second)[12]};
      bbox_data.insert(bbox_data.end(), tmp_bbox.begin(), tmp_bbox.end());
      dimensions_data.insert(dimensions_data.end(), tmp_dimensions.begin(), tmp_dimensions.end());
      location_data.insert(location_data.end(), tmp_location.begin(), tmp_location.end());
      truncated_data.push_back(static_cast<float>((item.second)[0]));
      occuluded_data.push_back(static_cast<uint32_t>(int64_t((item.second)[1])));
      alpha_data.push_back(static_cast<float>((item.second)[2]));
      rotation_y_data.push_back(static_cast<float>((item.second)[13]));
      bbox_num++;
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(label_data, TensorShape({bbox_num, 1}), &label));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(truncated_data, TensorShape({bbox_num, 1}), &truncated));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(occuluded_data, TensorShape({bbox_num, 1}), &occuluded));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(alpha_data, TensorShape({bbox_num, 1}), &alpha));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(bbox_data, TensorShape({bbox_num, 4}), &bbox));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(dimensions_data, TensorShape({bbox_num, 3}), &dimensions));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(location_data, TensorShape({bbox_num, 3}), &location));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(rotation_y_data, TensorShape({bbox_num, 1}), &rotation_y));
  (*row) = TensorRow({std::move(label), std::move(truncated), std::move(occuluded), std::move(alpha), std::move(bbox),
                      std::move(dimensions), std::move(location), std::move(rotation_y)});
  return Status::OK();
}

Status KITTIOp::CountTotalRows(int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  RETURN_IF_NOT_OK(PrepareData());
  *count = static_cast<int64_t>(image_ids_.size());
  return Status::OK();
}

Status KITTIOp::ComputeColMap() {
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
