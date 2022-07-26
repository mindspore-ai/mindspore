/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/coco_op.h"

#include <algorithm>
#include <fstream>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
const char kJsonImages[] = "images";
const char kJsonImagesFileName[] = "file_name";
const char kJsonId[] = "id";
const char kJsonAnnotations[] = "annotations";
const char kJsonAnnoSegmentation[] = "segmentation";
const char kJsonAnnoCaption[] = "caption";
const char kJsonAnnoCounts[] = "counts";
const char kJsonAnnoSegmentsInfo[] = "segments_info";
const char kJsonAnnoIscrowd[] = "iscrowd";
const char kJsonAnnoBbox[] = "bbox";
const char kJsonAnnoArea[] = "area";
const char kJsonAnnoImageId[] = "image_id";
const char kJsonAnnoNumKeypoints[] = "num_keypoints";
const char kJsonAnnoKeypoints[] = "keypoints";
const char kJsonAnnoCategoryId[] = "category_id";
const char kJsonCategories[] = "categories";
const char kJsonCategoriesIsthing[] = "isthing";
const char kJsonCategoriesName[] = "name";
const float kDefaultPadValue = -1.0;
const unsigned int kPadValueZero = 0;

#ifdef ENABLE_PYTHON
CocoOp::CocoOp(const TaskType &task_type, const std::string &image_folder_path, const std::string &annotation_path,
               int32_t num_workers, int32_t queue_size, bool decode, std::unique_ptr<DataSchema> data_schema,
               std::shared_ptr<SamplerRT> sampler, bool extra_metadata, py::function decrypt)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      decode_(decode),
      task_type_(task_type),
      image_folder_path_(image_folder_path),
      annotation_path_(annotation_path),
      data_schema_(std::move(data_schema)),
      extra_metadata_(extra_metadata),
      decrypt_(std::move(decrypt)) {}
#else
CocoOp::CocoOp(const TaskType &task_type, const std::string &image_folder_path, const std::string &annotation_path,
               int32_t num_workers, int32_t queue_size, bool decode, std::unique_ptr<DataSchema> data_schema,
               std::shared_ptr<SamplerRT> sampler, bool extra_metadata)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      decode_(decode),
      task_type_(task_type),
      image_folder_path_(image_folder_path),
      annotation_path_(annotation_path),
      data_schema_(std::move(data_schema)),
      extra_metadata_(extra_metadata) {}
#endif

void CocoOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nCOCO Directory: " << image_folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status CocoOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::string image_id = image_ids_[row_id];
  std::shared_ptr<Tensor> image;
  auto real_path = FileUtils::GetRealPath(image_folder_path_.c_str());
  if (!real_path.has_value()) {
    RETURN_STATUS_UNEXPECTED("Invalid file path, COCO dataset image folder: " + image_folder_path_ +
                             " does not exist.");
  }
  Path image_folder(real_path.value());
  Path kImageFile = image_folder / image_id;
  RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile.ToString(), &image));
  if (task_type_ == TaskType::Captioning) {
    std::shared_ptr<Tensor> captions;
    auto itr = captions_map_.find(image_id);
    if (itr == captions_map_.end()) {
      RETURN_STATUS_UNEXPECTED("Invalid annotation, the attribute of 'image_id': " + image_id +
                               " is missing from image node in annotation file: " + annotation_path_);
    }
    auto captions_str = itr->second;
    RETURN_IF_NOT_OK(
      Tensor::CreateFromVector(captions_str, TensorShape({static_cast<dsize_t>(captions_str.size()), 1}), &captions));
    RETURN_IF_NOT_OK(LoadCaptioningTensorRow(row_id, image_id, image, captions, trow));
    return Status::OK();
  }
  std::shared_ptr<Tensor> coordinate;
  auto itr = coordinate_map_.find(image_id);
  if (itr == coordinate_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation, the attribute of 'image_id': " + image_id +
                             " is missing from image node in annotation file: " + annotation_path_);
  }

  auto bboxRow = itr->second;
  std::vector<float> bbox_row;
  auto bbox_row_num = static_cast<dsize_t>(bboxRow.size());
  dsize_t bbox_column_num = 0;
  std::for_each(bboxRow.begin(), bboxRow.end(), [&](const auto &bbox) {
    if (static_cast<dsize_t>(bbox.size()) > bbox_column_num) {
      bbox_column_num = static_cast<dsize_t>(bbox.size());
    }
  });

  for (auto bbox : bboxRow) {
    bbox_row.insert(bbox_row.end(), bbox.begin(), bbox.end());
    dsize_t pad_len = bbox_column_num - static_cast<dsize_t>(bbox.size());
    if (pad_len > 0) {
      for (dsize_t i = 0; i < pad_len; i++) {
        bbox_row.push_back(kDefaultPadValue);
      }
    }
  }

  std::vector<dsize_t> bbox_dim = {bbox_row_num, bbox_column_num};
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(bbox_row, TensorShape(bbox_dim), &coordinate));

  if (task_type_ == TaskType::Detection) {
    RETURN_IF_NOT_OK(LoadDetectionTensorRow(row_id, image_id, image, coordinate, trow));
  } else if (task_type_ == TaskType::Stuff || task_type_ == TaskType::Keypoint) {
    RETURN_IF_NOT_OK(LoadSimpleTensorRow(row_id, image_id, image, coordinate, trow));
  } else if (task_type_ == TaskType::Panoptic) {
    RETURN_IF_NOT_OK(LoadMixTensorRow(row_id, image_id, image, coordinate, trow));
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, task type should be Detection, Stuff, Panoptic or Captioning.");
  }

  return Status::OK();
}

Status CocoOp::LoadDetectionTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                      std::shared_ptr<Tensor> coordinate, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::shared_ptr<Tensor> category_id, iscrowd;
  std::vector<uint32_t> category_id_row;
  std::vector<uint32_t> iscrowd_row;
  auto itr_item = simple_item_map_.find(image_id);
  if (itr_item == simple_item_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation, the attribute of 'image_id': " + image_id +
                             " is missing in the node of image from annotation file: " + annotation_path_ + ".");
  }

  std::vector<uint32_t> annotation = itr_item->second;
  for (int64_t i = 0; i < annotation.size(); i++) {
    if (i % 2 == 0) {
      category_id_row.push_back(annotation[i]);
    } else if (i % 2 == 1) {
      iscrowd_row.push_back(annotation[i]);
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(
    category_id_row, TensorShape({static_cast<dsize_t>(category_id_row.size()), 1}), &category_id));

  RETURN_IF_NOT_OK(
    Tensor::CreateFromVector(iscrowd_row, TensorShape({static_cast<dsize_t>(iscrowd_row.size()), 1}), &iscrowd));

  (*trow) = TensorRow(row_id, {std::move(image), std::move(coordinate), std::move(category_id), std::move(iscrowd)});
  Path image_folder(image_folder_path_);
  Path image_full_path = image_folder / image_id;
  std::vector<std::string> path_list = {image_full_path.ToString(), annotation_path_, annotation_path_,
                                        annotation_path_};
  if (extra_metadata_) {
    std::string img_id;
    size_t pos = image_id.find('.');
    if (pos == std::string::npos) {
      RETURN_STATUS_UNEXPECTED("Invalid image, 'image_id': " + image_id + " should be with suffix like \".jpg\"");
    }
    std::copy(image_id.begin(), image_id.begin() + pos, std::back_inserter(img_id));
    std::shared_ptr<Tensor> filename;
    RETURN_IF_NOT_OK(Tensor::CreateScalar(img_id, &filename));
    trow->push_back(std::move(filename));
    path_list.push_back(image_full_path.ToString());
  }
  trow->setPath(path_list);
  return Status::OK();
}

Status CocoOp::LoadSimpleTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                   std::shared_ptr<Tensor> coordinate, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::shared_ptr<Tensor> item;
  std::vector<uint32_t> item_queue;
  Path image_folder(image_folder_path_);
  auto itr_item = simple_item_map_.find(image_id);
  if (itr_item == simple_item_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid image_id, the attribute of 'image_id': " + image_id +
                             " is missing in the node of 'image' from annotation file: " + annotation_path_);
  }

  item_queue = itr_item->second;
  std::vector<dsize_t> bbox_dim = {static_cast<dsize_t>(item_queue.size()), 1};
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(item_queue, TensorShape(bbox_dim), &item));

  (*trow) = TensorRow(row_id, {std::move(image), std::move(coordinate), std::move(item)});
  Path image_full_path = image_folder / image_id;
  std::vector<std::string> path_list = {image_full_path.ToString(), annotation_path_, annotation_path_};
  if (extra_metadata_) {
    std::string img_id;
    size_t pos = image_id.find('.');
    if (pos == std::string::npos) {
      RETURN_STATUS_UNEXPECTED("Invalid image, 'image_id': " + image_id + " should be with suffix like \".jpg\"");
    }
    std::copy(image_id.begin(), image_id.begin() + pos, std::back_inserter(img_id));
    std::shared_ptr<Tensor> filename;
    RETURN_IF_NOT_OK(Tensor::CreateScalar(img_id, &filename));
    trow->push_back(std::move(filename));
    path_list.push_back(image_full_path.ToString());
  }
  trow->setPath(path_list);
  return Status::OK();
}

Status CocoOp::LoadCaptioningTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                       std::shared_ptr<Tensor> captions, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  (*trow) = TensorRow(row_id, {std::move(image), std::move(captions)});
  Path image_folder(image_folder_path_);
  Path image_full_path = image_folder / image_id;
  std::vector<std::string> path_list = {image_full_path.ToString(), annotation_path_};
  if (extra_metadata_) {
    std::string img_id;
    size_t pos = image_id.find('.');
    if (pos == std::string::npos) {
      RETURN_STATUS_UNEXPECTED("Invalid image, 'image_id': " + image_id + " should be with suffix like \".jpg\".");
    }
    std::copy(image_id.begin(), image_id.begin() + pos, std::back_inserter(img_id));
    std::shared_ptr<Tensor> filename;
    RETURN_IF_NOT_OK(Tensor::CreateScalar(img_id, &filename));
    trow->push_back(std::move(filename));
    path_list.push_back(image_full_path.ToString());
  }
  trow->setPath(path_list);
  return Status::OK();
}

Status CocoOp::LoadMixTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                std::shared_ptr<Tensor> coordinate, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::shared_ptr<Tensor> category_id, iscrowd, area;
  std::vector<uint32_t> category_id_row;
  std::vector<uint32_t> iscrowd_row;
  std::vector<uint32_t> area_row;
  auto itr_item = simple_item_map_.find(image_id);
  if (itr_item == simple_item_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid image_id, the attribute of 'image_id': " + image_id +
                             " is missing in the node of 'image' from annotation file: " + annotation_path_);
  }

  std::vector<uint32_t> annotation = itr_item->second;
  for (int64_t i = 0; i < annotation.size(); i++) {
    if (i % 3 == 0) {
      category_id_row.push_back(annotation[i]);
    } else if (i % 3 == 1) {
      iscrowd_row.push_back(annotation[i]);
    } else if (i % 3 == 2) {
      area_row.push_back(annotation[i]);
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(
    category_id_row, TensorShape({static_cast<dsize_t>(category_id_row.size()), 1}), &category_id));

  RETURN_IF_NOT_OK(
    Tensor::CreateFromVector(iscrowd_row, TensorShape({static_cast<dsize_t>(iscrowd_row.size()), 1}), &iscrowd));

  RETURN_IF_NOT_OK(Tensor::CreateFromVector(area_row, TensorShape({static_cast<dsize_t>(area_row.size()), 1}), &area));

  (*trow) = TensorRow(
    row_id, {std::move(image), std::move(coordinate), std::move(category_id), std::move(iscrowd), std::move(area)});
  Path image_folder(image_folder_path_);
  Path image_full_path = image_folder / image_id;
  std::vector<std::string> path_list = {image_full_path.ToString(), annotation_path_, annotation_path_,
                                        annotation_path_, annotation_path_};
  if (extra_metadata_) {
    std::string img_id;
    size_t pos = image_id.find('.');
    if (pos == std::string::npos) {
      RETURN_STATUS_UNEXPECTED("Invalid image, " + image_id + " should be with suffix like \".jpg\"");
    }
    std::copy(image_id.begin(), image_id.begin() + pos, std::back_inserter(img_id));
    std::shared_ptr<Tensor> filename;
    RETURN_IF_NOT_OK(Tensor::CreateScalar(img_id, &filename));
    trow->push_back(std::move(filename));
    path_list.push_back(image_full_path.ToString());
  }
  trow->setPath(path_list);
  return Status::OK();
}

template <typename T>
Status CocoOp::SearchNodeInJson(const nlohmann::json &input_tree, std::string node_name, T *output_node) {
  auto node = input_tree.find(node_name);
  CHECK_FAIL_RETURN_UNEXPECTED(node != input_tree.end(), "Invalid annotation, the attribute of '" + node_name +
                                                           "' is missing in annotation file: " + annotation_path_ +
                                                           ".");
  (*output_node) = *node;
  return Status::OK();
}

Status CocoOp::PrepareData() {
  nlohmann::json js;
  try {
    auto realpath = FileUtils::GetRealPath(annotation_path_.c_str());
    if (!realpath.has_value()) {
      std::string err_msg = "Invalid file path, Coco Dataset annotation file: " + annotation_path_ + " does not exist.";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }

    std::ifstream in(realpath.value());
    CHECK_FAIL_RETURN_UNEXPECTED(in.is_open(), "Invalid annotation file, Coco Dataset annotation file: " +
                                                 annotation_path_ + " open failed, permission denied!");
    in >> js;
  } catch (const std::exception &err) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation file, Coco Dataset annotation file:" + annotation_path_ +
                             " load failed, error description: " + std::string(err.what()));
  }

  std::vector<std::string> image_que;
  nlohmann::json image_list;
  RETURN_IF_NOT_OK(SearchNodeInJson(js, std::string(kJsonImages), &image_list));
  RETURN_IF_NOT_OK(ImageColumnLoad(image_list, &image_que));
  if (task_type_ == TaskType::Detection || task_type_ == TaskType::Panoptic) {
    nlohmann::json node_categories;
    RETURN_IF_NOT_OK(SearchNodeInJson(js, std::string(kJsonCategories), &node_categories));
    RETURN_IF_NOT_OK(CategoriesColumnLoad(node_categories));
  }
  nlohmann::json annotations_list;
  RETURN_IF_NOT_OK(SearchNodeInJson(js, std::string(kJsonAnnotations), &annotations_list));
  for (const auto &annotation : annotations_list) {
    int32_t image_id = 0, id = 0;
    std::string file_name;
    RETURN_IF_NOT_OK(SearchNodeInJson(annotation, std::string(kJsonAnnoImageId), &image_id));
    auto itr_file = image_index_.find(image_id);
    CHECK_FAIL_RETURN_UNEXPECTED(itr_file != image_index_.end(),
                                 "Invalid annotation, the attribute of 'image_id': " + std::to_string(image_id) +
                                   " is missing in the node of 'image' from annotation file: " + annotation_path_);
    file_name = itr_file->second;
    switch (task_type_) {
      case TaskType::Detection:
        RETURN_IF_NOT_OK(SearchNodeInJson(annotation, std::string(kJsonId), &id));
        RETURN_IF_NOT_OK(DetectionColumnLoad(annotation, file_name, id));
        break;
      case TaskType::Stuff:
        RETURN_IF_NOT_OK(SearchNodeInJson(annotation, std::string(kJsonId), &id));
        RETURN_IF_NOT_OK(StuffColumnLoad(annotation, file_name, id));
        break;
      case TaskType::Keypoint:
        RETURN_IF_NOT_OK(SearchNodeInJson(annotation, std::string(kJsonId), &id));
        RETURN_IF_NOT_OK(KeypointColumnLoad(annotation, file_name, id));
        break;
      case TaskType::Panoptic:
        RETURN_IF_NOT_OK(PanopticColumnLoad(annotation, file_name, image_id));
        break;
      case TaskType::Captioning:
        RETURN_IF_NOT_OK(SearchNodeInJson(annotation, std::string(kJsonId), &id));
        RETURN_IF_NOT_OK(CaptionColumnLoad(annotation, file_name, id));
        break;
      default:
        RETURN_STATUS_UNEXPECTED(
          "Invalid parameter, task type should be Detection, Stuff, Keypoint, Panoptic or Captioning.");
    }
  }
  if (task_type_ == TaskType::Captioning) {
    for (const auto &img : image_que) {
      if (captions_map_.find(img) != captions_map_.end()) {
        image_ids_.push_back(img);
      }
    }
  } else {
    for (const auto &img : image_que) {
      if (coordinate_map_.find(img) != coordinate_map_.end()) {
        image_ids_.push_back(img);
      }
    }
  }
  num_rows_ = image_ids_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ != 0,
    "Invalid data, 'CocoDataset' API can't read the data file (interface mismatch or no data found). "
    "Check file in directory: " +
      image_folder_path_ + ".");
  return Status::OK();
}

Status CocoOp::ImageColumnLoad(const nlohmann::json &image_tree, std::vector<std::string> *image_vec) {
  if (image_tree.empty()) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation, the 'image' node is missing in annotation file: " + annotation_path_ +
                             ".");
  }
  for (const auto &img : image_tree) {
    std::string file_name;
    int32_t id = 0;
    RETURN_IF_NOT_OK(SearchNodeInJson(img, std::string(kJsonImagesFileName), &file_name));
    RETURN_IF_NOT_OK(SearchNodeInJson(img, std::string(kJsonId), &id));

    image_index_[id] = file_name;
    image_vec->push_back(file_name);
  }
  return Status::OK();
}

Status CocoOp::DetectionColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                                   const int32_t &unique_id) {
  std::vector<float> bbox;
  nlohmann::json node_bbox;
  uint32_t category_id = 0, iscrowd = 0;
  RETURN_IF_NOT_OK(SearchNodeInJson(annotation_tree, std::string(kJsonAnnoBbox), &node_bbox));
  RETURN_IF_NOT_OK(SearchNodeInJson(annotation_tree, std::string(kJsonAnnoCategoryId), &category_id));
  auto search_category = category_set_.find(category_id);
  if (search_category == category_set_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation, the attribute of 'category_id': " + std::to_string(category_id) +
                             " is missing in the node of 'categories' from annotation file: " + annotation_path_);
  }
  auto node_iscrowd = annotation_tree.find(kJsonAnnoIscrowd);
  if (node_iscrowd != annotation_tree.end()) {
    iscrowd = *node_iscrowd;
  }
  bbox.insert(bbox.end(), node_bbox.begin(), node_bbox.end());
  coordinate_map_[image_file].push_back(bbox);
  simple_item_map_[image_file].push_back(category_id);
  simple_item_map_[image_file].push_back(iscrowd);
  return Status::OK();
}

Status CocoOp::StuffColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                               const int32_t &unique_id) {
  uint32_t iscrowd = 0;
  std::vector<float> bbox;
  RETURN_IF_NOT_OK(SearchNodeInJson(annotation_tree, std::string(kJsonAnnoIscrowd), &iscrowd));
  simple_item_map_[image_file].push_back(iscrowd);
  nlohmann::json segmentation;
  RETURN_IF_NOT_OK(SearchNodeInJson(annotation_tree, std::string(kJsonAnnoSegmentation), &segmentation));
  if (iscrowd == 0) {
    for (auto item : segmentation) {
      if (!bbox.empty()) {
        bbox.clear();
      }
      bbox.insert(bbox.end(), item.begin(), item.end());
      coordinate_map_[image_file].push_back(bbox);
    }
  } else if (iscrowd == 1) {
    nlohmann::json segmentation_count;
    RETURN_IF_NOT_OK(SearchNodeInJson(segmentation, std::string(kJsonAnnoCounts), &segmentation_count));
    bbox.insert(bbox.end(), segmentation_count.begin(), segmentation_count.end());
    coordinate_map_[image_file].push_back(bbox);
  }
  return Status::OK();
}

Status CocoOp::KeypointColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                                  const int32_t &unique_id) {
  auto itr_num_keypoint = annotation_tree.find(kJsonAnnoNumKeypoints);
  if (itr_num_keypoint == annotation_tree.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation, the 'num_keypoint' node is missing in annotation file: " +
                             annotation_path_ + " where 'image_id': " + std::to_string(unique_id) + ".");
  }
  simple_item_map_[image_file].push_back(*itr_num_keypoint);
  auto itr_keypoint = annotation_tree.find(kJsonAnnoKeypoints);
  if (itr_keypoint == annotation_tree.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation, the 'keypoint' node is missing in annotation file: " +
                             annotation_path_ + " where 'image_id': " + std::to_string(unique_id) + ".");
  }
  coordinate_map_[image_file].push_back(*itr_keypoint);
  return Status::OK();
}

Status CocoOp::PanopticColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                                  const int32_t &image_id) {
  auto itr_segments = annotation_tree.find(kJsonAnnoSegmentsInfo);
  if (itr_segments == annotation_tree.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid annotation, the 'segments_info' node is missing in annotation file: " +
                             annotation_path_ + " where 'image_id': " + std::to_string(image_id) + ".");
  }
  for (auto info : *itr_segments) {
    std::vector<float> bbox;
    uint32_t category_id = 0;
    auto itr_bbox = info.find(kJsonAnnoBbox);
    if (itr_bbox == info.end()) {
      RETURN_STATUS_UNEXPECTED(
        "Invalid annotation, the 'bbox' attribute is missing in the node of 'segments_info' where 'image_id': " +
        std::to_string(image_id) + " from annotation file: " + annotation_path_ + ".");
    }
    bbox.insert(bbox.end(), itr_bbox->begin(), itr_bbox->end());
    coordinate_map_[image_file].push_back(bbox);

    RETURN_IF_NOT_OK(SearchNodeInJson(info, std::string(kJsonAnnoCategoryId), &category_id));
    auto search_category = category_set_.find(category_id);
    if (search_category == category_set_.end()) {
      RETURN_STATUS_UNEXPECTED("Invalid annotation, the attribute of 'category_id': " + std::to_string(category_id) +
                               " is missing in the node of 'categories' from " + annotation_path_ + ".");
    }
    auto itr_iscrowd = info.find(kJsonAnnoIscrowd);
    if (itr_iscrowd == info.end()) {
      RETURN_STATUS_UNEXPECTED(
        "Invalid annotation, the attribute of 'iscrowd' is missing in the node of 'segments_info' where 'image_id': " +
        std::to_string(image_id) + " from annotation file: " + annotation_path_ + ".");
    }
    auto itr_area = info.find(kJsonAnnoArea);
    if (itr_area == info.end()) {
      RETURN_STATUS_UNEXPECTED(
        "Invalid annotation, the attribute of 'area' is missing in the node of 'segments_info' where 'image_id': " +
        std::to_string(image_id) + " from annotation file: " + annotation_path_ + ".");
    }
    simple_item_map_[image_file].push_back(category_id);
    simple_item_map_[image_file].push_back(*itr_iscrowd);
    simple_item_map_[image_file].push_back(*itr_area);
  }
  return Status::OK();
}

Status CocoOp::CaptionColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                                 const int32_t &unique_id) {
  std::string caption;
  RETURN_IF_NOT_OK(SearchNodeInJson(annotation_tree, std::string(kJsonAnnoCaption), &caption));
  captions_map_[image_file].push_back(caption);
  return Status::OK();
}

Status CocoOp::CategoriesColumnLoad(const nlohmann::json &categories_tree) {
  if (categories_tree.empty()) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid annotation, the 'categories' node is missing in annotation file: " + annotation_path_ + ".");
  }
  for (auto category : categories_tree) {
    int32_t id = 0;
    std::string name;
    std::vector<int32_t> label_info;
    auto itr_id = category.find(kJsonId);
    if (itr_id == category.end()) {
      RETURN_STATUS_UNEXPECTED(
        "Invalid annotation, the attribute of 'id' is missing in the node of 'categories' from annotation file: " +
        annotation_path_);
    }
    id = *itr_id;
    label_info.push_back(id);
    category_set_.insert(id);

    auto itr_name = category.find(kJsonCategoriesName);
    CHECK_FAIL_RETURN_UNEXPECTED(
      itr_name != category.end(),
      "Invalid annotation, the attribute of 'name' is missing in the node of 'categories' where 'id': " +
        std::to_string(id));
    name = *itr_name;

    if (task_type_ == TaskType::Panoptic) {
      auto itr_isthing = category.find(kJsonCategoriesIsthing);
      CHECK_FAIL_RETURN_UNEXPECTED(itr_isthing != category.end(),
                                   "Invalid annotation, the attribute of 'isthing' is missing in the node of "
                                   "'categories' from annotation file: " +
                                     annotation_path_);
      label_info.push_back(*itr_isthing);
    }
    label_index_.emplace_back(std::make_pair(name, label_info));
  }
  return Status::OK();
}

Status CocoOp::ReadImageToTensor(const std::string &path, std::shared_ptr<Tensor> *tensor) const {
#ifdef ENABLE_PYTHON
  RETURN_IF_NOT_OK(MappableLeafOp::ImageDecrypt(path, tensor, decrypt_));
#else
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(path, tensor));
#endif

  if (decode_) {
    Status rc = Decode(*tensor, tensor);
    CHECK_FAIL_RETURN_UNEXPECTED(
      rc.IsOk(), "Invalid image, failed to decode " + path + ": the image is broken or permission denied.");
  }
  return Status::OK();
}

Status CocoOp::CountTotalRows(int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  RETURN_IF_NOT_OK(PrepareData());
  *count = static_cast<int64_t>(image_ids_.size());
  return Status::OK();
}

Status CocoOp::ComputeColMap() {
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

Status CocoOp::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  RETURN_UNEXPECTED_IF_NULL(output_class_indexing);
  if ((*output_class_indexing).empty()) {
    if ((task_type_ != TaskType::Detection) && (task_type_ != TaskType::Panoptic)) {
      MS_LOG(ERROR) << "Invalid task, only 'Detection' and 'Panoptic' task support GetClassIndex.";
      RETURN_STATUS_UNEXPECTED("Invalid task, only 'Detection' and 'Panoptic' task support GetClassIndex.");
    }
    RETURN_IF_NOT_OK(PrepareData());
    for (const auto &label : label_index_) {
      (*output_class_indexing).emplace_back(std::make_pair(label.first, label.second));
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
