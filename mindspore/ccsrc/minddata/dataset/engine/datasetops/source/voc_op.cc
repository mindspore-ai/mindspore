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
#include "minddata/dataset/engine/datasetops/source/voc_op.h"

#include <algorithm>
#include <fstream>

#include "utils/file_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
const char kColumnImage[] = "image";
const char kColumnTarget[] = "target";
const char kColumnBbox[] = "bbox";
const char kColumnLabel[] = "label";
const char kColumnDifficult[] = "difficult";
const char kColumnTruncate[] = "truncate";
const char kJPEGImagesFolder[] = "/JPEGImages/";
const char kSegmentationClassFolder[] = "/SegmentationClass/";
const char kAnnotationsFolder[] = "/Annotations/";
const char kImageSetsSegmentation[] = "/ImageSets/Segmentation/";
const char kImageSetsMain[] = "/ImageSets/Main/";
const char kImageExtension[] = ".jpg";
const char kSegmentationExtension[] = ".png";
const char kAnnotationExtension[] = ".xml";
const char kImageSetsExtension[] = ".txt";

VOCOp::VOCOp(const TaskType &task_type, const std::string &task_mode, const std::string &folder_path,
             const std::map<std::string, int32_t> &class_index, int32_t num_workers, int32_t queue_size, bool decode,
             std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler, bool extra_metadata)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      decode_(decode),
      row_cnt_(0),
      task_type_(task_type),
      usage_(task_mode),
      folder_path_(folder_path),
      class_index_(class_index),
      data_schema_(std::move(data_schema)),
      extra_metadata_(extra_metadata) {
  io_block_queues_.Init(num_workers_, queue_size);
}

void VOCOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nVOC Directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status VOCOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  std::string image_id = image_ids_[row_id];
  std::vector<std::string> path_list;
  const std::string kImageFile =
    folder_path_ + std::string(kJPEGImagesFolder) + image_id + std::string(kImageExtension);
  if (task_type_ == TaskType::Segmentation) {
    std::shared_ptr<Tensor> image, target;
    const std::string kTargetFile =
      folder_path_ + std::string(kSegmentationClassFolder) + image_id + std::string(kSegmentationExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile, data_schema_->Column(0), &image));
    RETURN_IF_NOT_OK(ReadImageToTensor(kTargetFile, data_schema_->Column(1), &target));
    (*trow) = TensorRow(row_id, {std::move(image), std::move(target)});
    path_list = {kImageFile, kTargetFile};
  } else if (task_type_ == TaskType::Detection) {
    std::shared_ptr<Tensor> image;
    TensorRow annotation;
    const std::string kAnnotationFile =
      folder_path_ + std::string(kAnnotationsFolder) + image_id + std::string(kAnnotationExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile, data_schema_->Column(0), &image));
    RETURN_IF_NOT_OK(ReadAnnotationToTensor(kAnnotationFile, &annotation));
    trow->setId(row_id);
    trow->push_back(std::move(image));
    trow->insert(trow->end(), annotation.begin(), annotation.end());
    path_list = {kImageFile, kAnnotationFile, kAnnotationFile, kAnnotationFile, kAnnotationFile};
  }
  if (extra_metadata_) {
    // Now VOCDataset add a new column named "_meta-filename".
    std::shared_ptr<Tensor> filename;
    RETURN_IF_NOT_OK(Tensor::CreateScalar(image_id, &filename));
    trow->push_back(std::move(filename));
    path_list.push_back(kImageFile);
  }
  trow->setPath(path_list);
  return Status::OK();
}

Status VOCOp::ParseImageIds() {
  if (!image_ids_.empty()) return Status::OK();
  std::string image_sets_file;
  if (task_type_ == TaskType::Segmentation) {
    image_sets_file = folder_path_ + std::string(kImageSetsSegmentation) + usage_ + std::string(kImageSetsExtension);
  } else if (task_type_ == TaskType::Detection) {
    image_sets_file = folder_path_ + std::string(kImageSetsMain) + usage_ + std::string(kImageSetsExtension);
  }

  auto realpath = FileUtils::GetRealPath(image_sets_file.data());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file, get real path failed, path=" << image_sets_file;
    RETURN_STATUS_UNEXPECTED("Invalid file, get real path failed, path=" + image_sets_file);
  }

  std::ifstream in_file;
  in_file.open(realpath.value());
  if (in_file.fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + image_sets_file);
  }
  std::string id;
  while (getline(in_file, id)) {
    if (id.size() > 0 && id[id.size() - 1] == '\r') {
      image_ids_.push_back(id.substr(0, id.size() - 1));
    } else {
      image_ids_.push_back(id);
    }
  }
  in_file.close();
  image_ids_.shrink_to_fit();
  num_rows_ = image_ids_.size();
  return Status::OK();
}

Status VOCOp::ParseAnnotationIds() {
  std::vector<std::string> new_image_ids;
  for (auto id : image_ids_) {
    const std::string annotation_name =
      folder_path_ + std::string(kAnnotationsFolder) + id + std::string(kAnnotationExtension);
    RETURN_IF_NOT_OK(ParseAnnotationBbox(annotation_name));
    if (annotation_map_.find(annotation_name) != annotation_map_.end()) {
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
      "Invalid data, VOCDataset API can't read the data file (interface mismatch or no data found). "
      "Check file in directory:" +
      folder_path_);
  }
  return Status::OK();
}

void VOCOp::ParseNodeValue(XMLElement *bbox_node, const char *name, float *value) {
  *value = 0.0;
  if (bbox_node != nullptr) {
    XMLElement *node = bbox_node->FirstChildElement(name);
    if (node != nullptr) {
      *value = node->FloatText();
    }
  }
}

Status VOCOp::CheckIfBboxValid(const float &xmin, const float &ymin, const float &xmax, const float &ymax,
                               const std::string &path) {
  if (!(xmin > 0 && ymin > 0 && xmax > xmin && ymax > ymin)) {
    std::string invalid_bbox = "{" + std::to_string(static_cast<int>(xmin)) + ", " +
                               std::to_string(static_cast<int>(ymin)) + ", " + std::to_string(static_cast<int>(xmax)) +
                               ", " + std::to_string(static_cast<int>(ymax)) + "}";
    RETURN_STATUS_UNEXPECTED("Invalid bndbox: " + invalid_bbox + " found in " + path);
  }
  return Status::OK();
}

Status VOCOp::ParseAnnotationBbox(const std::string &path) {
  if (!Path(path).Exists()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + path);
  }
  Annotation annotation;
  XMLDocument doc;
  XMLError e = doc.LoadFile(common::SafeCStr(path));
  if (e != XMLError::XML_SUCCESS) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to load xml file: " + path);
  }
  XMLElement *root = doc.RootElement();
  if (root == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid data, failed to load root element for xml file.");
  }
  XMLElement *object = root->FirstChildElement("object");
  if (object == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid data, no object found in " + path);
  }
  while (object != nullptr) {
    std::string label_name;
    float xmin = 0.0, ymin = 0.0, xmax = 0.0, ymax = 0.0, truncated = 0.0, difficult = 0.0;
    XMLElement *name_node = object->FirstChildElement("name");
    if (name_node != nullptr && name_node->GetText() != 0) label_name = name_node->GetText();
    ParseNodeValue(object, "difficult", &difficult);
    ParseNodeValue(object, "truncated", &truncated);

    XMLElement *bbox_node = object->FirstChildElement("bndbox");
    if (bbox_node != nullptr) {
      ParseNodeValue(bbox_node, "xmin", &xmin);
      ParseNodeValue(bbox_node, "xmax", &xmax);
      ParseNodeValue(bbox_node, "ymin", &ymin);
      ParseNodeValue(bbox_node, "ymax", &ymax);
      RETURN_IF_NOT_OK(CheckIfBboxValid(xmin, ymin, xmax, ymax, path));
    } else {
      RETURN_STATUS_UNEXPECTED("Invalid data, bndbox dismatch in " + path);
    }

    if (label_name != "" && (class_index_.empty() || class_index_.find(label_name) != class_index_.end()) && xmin > 0 &&
        ymin > 0 && xmax > xmin && ymax > ymin) {
      std::vector<float> bbox_list = {xmin, ymin, xmax - xmin, ymax - ymin, difficult, truncated};
      annotation.emplace_back(std::make_pair(label_name, bbox_list));
      label_index_[label_name] = 0;
    }
    object = object->NextSiblingElement("object");
  }
  if (annotation.size() > 0) {
    annotation_map_[path] = annotation;
  }
  return Status::OK();
}

Status VOCOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&VOCOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->ParseImageIds());
  if (task_type_ == TaskType::Detection) {
    RETURN_IF_NOT_OK(this->ParseAnnotationIds());
  }
  RETURN_IF_NOT_OK(this->InitSampler());
  return Status::OK();
}

Status VOCOp::ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor) {
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(path, tensor));
  if (decode_ == true) {
    Status rc = Decode(*tensor, tensor);
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("Invalid data, failed to decode image: " + path);
    }
  }
  return Status::OK();
}

// When task is Detection, user can get bbox data with four columns:
// column ["bbox"] with datatype=float32
// column ["label"] with datatype=uint32
// column ["difficult"] with datatype=uint32
// column ["truncate"] with datatype=uint32
Status VOCOp::ReadAnnotationToTensor(const std::string &path, TensorRow *row) {
  Annotation annotation = annotation_map_[path];
  std::shared_ptr<Tensor> bbox, label, difficult, truncate;
  std::vector<float> bbox_data;
  std::vector<uint32_t> label_data, difficult_data, truncate_data;
  dsize_t bbox_num = 0;
  for (auto item : annotation) {
    if (label_index_.find(item.first) != label_index_.end()) {
      if (class_index_.find(item.first) != class_index_.end()) {
        label_data.push_back(static_cast<uint32_t>(class_index_[item.first]));
      } else {
        label_data.push_back(static_cast<uint32_t>(label_index_[item.first]));
      }
      CHECK_FAIL_RETURN_UNEXPECTED(
        item.second.size() == 6,
        "Invalid parameter, annotation only support 6 parameters, but got " + std::to_string(item.second.size()));

      std::vector<float> tmp_bbox = {(item.second)[0], (item.second)[1], (item.second)[2], (item.second)[3]};
      bbox_data.insert(bbox_data.end(), tmp_bbox.begin(), tmp_bbox.end());
      difficult_data.push_back(static_cast<uint32_t>((item.second)[4]));
      truncate_data.push_back(static_cast<uint32_t>((item.second)[5]));
      bbox_num++;
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(bbox_data, TensorShape({bbox_num, 4}), &bbox));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(label_data, TensorShape({bbox_num, 1}), &label));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(difficult_data, TensorShape({bbox_num, 1}), &difficult));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(truncate_data, TensorShape({bbox_num, 1}), &truncate));
  (*row) = TensorRow({std::move(bbox), std::move(label), std::move(difficult), std::move(truncate)});
  return Status::OK();
}

Status VOCOp::CountTotalRows(int64_t *count) {
  switch (task_type_) {
    case TaskType::Detection:
      RETURN_IF_NOT_OK(ParseImageIds());
      RETURN_IF_NOT_OK(ParseAnnotationIds());
      break;
    case TaskType::Segmentation:
      RETURN_IF_NOT_OK(ParseImageIds());
      break;
  }
  *count = static_cast<int64_t>(image_ids_.size());
  return Status::OK();
}

Status VOCOp::ComputeColMap() {
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

Status VOCOp::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  if ((*output_class_indexing).empty()) {
    if (task_type_ != TaskType::Detection) {
      MS_LOG(ERROR) << "Invalid parameter, GetClassIndexing only valid in \"Detection\" task.";
      RETURN_STATUS_UNEXPECTED("Invalid parameter, GetClassIndexing only valid in \"Detection\" task.");
    }
    RETURN_IF_NOT_OK(ParseImageIds());
    RETURN_IF_NOT_OK(ParseAnnotationIds());
    for (const auto &label : label_index_) {
      if (!class_index_.empty()) {
        (*output_class_indexing)
          .emplace_back(std::make_pair(label.first, std::vector<int32_t>(1, class_index_[label.first])));
      } else {
        (*output_class_indexing).emplace_back(std::make_pair(label.first, std::vector<int32_t>(1, label.second)));
      }
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
