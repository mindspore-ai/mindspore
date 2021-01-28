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
#include "minddata/dataset/engine/datasetops/source/coco_op.h"

#include <algorithm>
#include <fstream>
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
const char kColumnImage[] = "image";
const char kJsonImages[] = "images";
const char kJsonImagesFileName[] = "file_name";
const char kJsonId[] = "id";
const char kJsonAnnotations[] = "annotations";
const char kJsonAnnoSegmentation[] = "segmentation";
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

CocoOp::Builder::Builder() : builder_decode_(false), builder_sampler_(nullptr) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
  builder_task_type_ = TaskType::Detection;
}

Status CocoOp::Builder::Build(std::shared_ptr<CocoOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (builder_sampler_ == nullptr) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    builder_sampler_ = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }
  builder_schema_ = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(builder_schema_->AddColumn(
    ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  switch (builder_task_type_) {
    case TaskType::Detection:
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoBbox), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoCategoryId), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoIscrowd), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case TaskType::Stuff:
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoSegmentation), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoIscrowd), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case TaskType::Keypoint:
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoKeypoints), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoNumKeypoints), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case TaskType::Panoptic:
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoBbox), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoCategoryId), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoIscrowd), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(builder_schema_->AddColumn(
        ColDescriptor(std::string(kJsonAnnoArea), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    default:
      RETURN_STATUS_UNEXPECTED("Invalid parameter, task type should be Detection, Stuff, Keypoint or Panoptic.");
  }
  *ptr = std::make_shared<CocoOp>(builder_task_type_, builder_dir_, builder_file_, builder_num_workers_,
                                  builder_rows_per_buffer_, builder_op_connector_size_, builder_decode_,
                                  std::move(builder_schema_), std::move(builder_sampler_));
  return Status::OK();
}

Status CocoOp::Builder::SanityCheck() {
  Path dir(builder_dir_);
  Path file(builder_file_);
  std::string err_msg;
  err_msg += dir.IsDirectory() == false
               ? "Invalid parameter, Coco image folder path is invalid or not set, path: " + builder_dir_ + ".\n"
               : "";
  err_msg += file.Exists() == false
               ? "Invalid parameter, Coco annotation json path is invalid or not set, path: " + builder_dir_ + ".\n"
               : "";
  err_msg += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                           std::to_string(builder_num_workers_) + ".\n"
                                       : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err_msg);
}

CocoOp::CocoOp(const TaskType &task_type, const std::string &image_folder_path, const std::string &annotation_path,
               int32_t num_workers, int32_t rows_per_buffer, int32_t queue_size, bool decode,
               std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : ParallelOp(num_workers, queue_size, std::move(sampler)),
      decode_(decode),
      row_cnt_(0),
      buf_cnt_(0),
      task_type_(task_type),
      image_folder_path_(image_folder_path),
      annotation_path_(annotation_path),
      rows_per_buffer_(rows_per_buffer),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers_, queue_size);
}

Status CocoOp::TraverseSampleIds(const std::shared_ptr<Tensor> &sample_ids, std::vector<int64_t> *keys) {
  for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
    if ((*itr) > num_rows_) continue;
    keys->push_back(*itr);
    row_cnt_++;
    if (row_cnt_ % rows_per_buffer_ == 0) {
      RETURN_IF_NOT_OK(io_block_queues_[buf_cnt_++ % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(*keys, IOBlock::kDeIoBlockNone))));
      keys->clear();
    }
  }
  return Status::OK();
}

Status CocoOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  std::unique_ptr<DataBuffer> sampler_buffer;
  RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
  while (true) {
    std::vector<int64_t> keys;
    keys.reserve(rows_per_buffer_);
    while (sampler_buffer->eoe() == false) {
      std::shared_ptr<Tensor> sample_ids;
      RETURN_IF_NOT_OK(sampler_buffer->GetTensor(&sample_ids, 0, 0));
      if (sample_ids->type() != DataType(DataType::DE_INT64)) {
        RETURN_STATUS_UNEXPECTED("Invalid parameter, data type of Sampler Tensor isn't int64, got " +
                                 sample_ids->type().ToString());
      }
      RETURN_IF_NOT_OK(TraverseSampleIds(sample_ids, &keys));
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    if (keys.empty() == false) {
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
    }
    if (IsLastIteration()) {
      std::unique_ptr<IOBlock> eoe_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe);
      std::unique_ptr<IOBlock> eof_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof);
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::move(eoe_block)));
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::move(eof_block)));
      for (int32_t i = 0; i < num_workers_; i++) {
        RETURN_IF_NOT_OK(
          io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
      }
      return Status::OK();
    } else {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    }

    if (epoch_sync_flag_) {
      // If epoch_sync_flag_ is set, then master thread sleeps until all the worker threads have finished their job for
      // the current epoch.
      RETURN_IF_NOT_OK(WaitForWorkers());
    }
    // If not the last repeat, self-reset and go to loop again.
    if (!IsLastIteration()) {
      RETURN_IF_NOT_OK(Reset());
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    UpdateRepeatAndEpochCounter();
  }
}

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

Status CocoOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(sampler_->ResetSampler());
  row_cnt_ = 0;
  return Status::OK();
}

Status CocoOp::LoadTensorRow(row_id_type row_id, const std::string &image_id, TensorRow *trow) {
  std::shared_ptr<Tensor> image, coordinate;
  auto itr = coordinate_map_.find(image_id);
  if (itr == coordinate_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid data, image_id: " + image_id +
                             " in annotation node is not found in image node in json file.");
  }

  std::string kImageFile = image_folder_path_ + std::string("/") + image_id;
  RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile, data_schema_->column(0), &image));

  auto bboxRow = itr->second;
  std::vector<float> bbox_row;
  dsize_t bbox_row_num = static_cast<dsize_t>(bboxRow.size());
  dsize_t bbox_column_num = 0;
  for (auto bbox : bboxRow) {
    if (static_cast<dsize_t>(bbox.size()) > bbox_column_num) {
      bbox_column_num = static_cast<dsize_t>(bbox.size());
    }
  }

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
    RETURN_STATUS_UNEXPECTED("Invalid parameter, task type should be Detection, Stuff or Panoptic.");
  }

  return Status::OK();
}

// When task is Detection, user can get data with four columns:
// column ["image"] with datatype=uint8
// column ["bbox"] with datatype=float32
// column ["category_id"] with datatype=uint32
// column ["iscrowd"] with datatype=uint32
// By the way, column ["iscrowd"] is used for some testcases, like fasterRcnn.
// If "iscrowd" is not existed, user will get default value 0.
Status CocoOp::LoadDetectionTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                      std::shared_ptr<Tensor> coordinate, TensorRow *trow) {
  std::shared_ptr<Tensor> category_id, iscrowd;
  std::vector<uint32_t> category_id_row;
  std::vector<uint32_t> iscrowd_row;
  auto itr_item = simple_item_map_.find(image_id);
  if (itr_item == simple_item_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid data, image_id: " + image_id +
                             " in annotation node is not found in image node in json file.");
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
  std::string image_full_path = image_folder_path_ + std::string("/") + image_id;
  trow->setPath({image_full_path, annotation_path_, annotation_path_, annotation_path_});
  return Status::OK();
}

// When task is "Stuff"/"Keypoint", user can get data with three columns:
// column ["image"] with datatype=uint8
// column ["segmentation"]/["keypoints"] with datatype=float32
// column ["iscrowd"]/["num_keypoints"] with datatype=uint32
Status CocoOp::LoadSimpleTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                   std::shared_ptr<Tensor> coordinate, TensorRow *trow) {
  std::shared_ptr<Tensor> item;
  std::vector<uint32_t> item_queue;
  auto itr_item = simple_item_map_.find(image_id);
  if (itr_item == simple_item_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid data, image_id: " + image_id +
                             " in annotation node is not found in image node in json file.");
  }

  item_queue = itr_item->second;
  std::vector<dsize_t> bbox_dim = {static_cast<dsize_t>(item_queue.size()), 1};
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(item_queue, TensorShape(bbox_dim), &item));

  (*trow) = TensorRow(row_id, {std::move(image), std::move(coordinate), std::move(item)});
  std::string image_full_path = image_folder_path_ + std::string("/") + image_id;
  trow->setPath({image_full_path, annotation_path_, annotation_path_});
  return Status::OK();
}

// When task is "Panoptic", user can get data with five columns:
// column ["image"] with datatype=uint8
// column ["bbox"] with datatype=float32
// column ["category_id"] with datatype=uint32
// column ["iscrowd"] with datatype=uint32
// column ["area"] with datatype=uint32
Status CocoOp::LoadMixTensorRow(row_id_type row_id, const std::string &image_id, std::shared_ptr<Tensor> image,
                                std::shared_ptr<Tensor> coordinate, TensorRow *trow) {
  std::shared_ptr<Tensor> category_id, iscrowd, area;
  std::vector<uint32_t> category_id_row;
  std::vector<uint32_t> iscrowd_row;
  std::vector<uint32_t> area_row;
  auto itr_item = simple_item_map_.find(image_id);
  if (itr_item == simple_item_map_.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid data, image_id: " + image_id +
                             " in annotation node is not found in image node in json file.");
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
  std::string image_full_path = image_folder_path_ + std::string("/") + image_id;
  trow->setPath({image_full_path, annotation_path_, annotation_path_, annotation_path_, annotation_path_});
  return Status::OK();
}

Status CocoOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  TensorRow trow;
  for (const int64_t &key : keys) {
    RETURN_IF_NOT_OK(this->LoadTensorRow(key, image_ids_[key], &trow));
    deq->push_back(std::move(trow));
  }
  (*db)->set_tensor_table(std::move(deq));
  return Status::OK();
}

Status CocoOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  int64_t buffer_id = worker_id;
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->wait() == true) {
      // Sync io_block is a signal that master thread wants us to pause and sync with other workers.
      // The last guy who comes to this sync point should reset the counter and wake up the master thread.
      if (++num_workers_paused_ == num_workers_) {
        wait_for_workers_post_.Set();
      }
    } else if (io_block->eoe() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE)));
      buffer_id = worker_id;
    } else if (io_block->eof() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, (std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF))));
    } else {
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(io_block->GetKeys(&keys));
      if (keys.empty() == true) return Status::OK();
      std::unique_ptr<DataBuffer> db = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
      RETURN_IF_NOT_OK(LoadBuffer(keys, &db));
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db)));
      buffer_id += num_workers_;
    }
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("Unexpected nullptr received in worker");
}

template <typename T>
Status CocoOp::SearchNodeInJson(const nlohmann::json &input_tree, std::string node_name, T *output_node) {
  auto node = input_tree.find(node_name);
  CHECK_FAIL_RETURN_UNEXPECTED(node != input_tree.end(), "Invalid data, invalid node found in json: " + node_name);
  (*output_node) = *node;
  return Status::OK();
}

Status CocoOp::ParseAnnotationIds() {
  nlohmann::json js;
  try {
    std::ifstream in(annotation_path_);
    in >> js;
  } catch (const std::exception &err) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open json file: " + annotation_path_);
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
  for (auto annotation : annotations_list) {
    int32_t image_id = 0, id = 0;
    std::string file_name;
    RETURN_IF_NOT_OK(SearchNodeInJson(annotation, std::string(kJsonAnnoImageId), &image_id));
    auto itr_file = image_index_.find(image_id);
    if (itr_file == image_index_.end()) {
      RETURN_STATUS_UNEXPECTED("Invalid data, image_id: " + std::to_string(image_id) +
                               " in annotation node is not found in image node in json file.");
    }
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
      default:
        RETURN_STATUS_UNEXPECTED("Invalid parameter, task type should be Detection, Stuff, Keypoint or Panoptic.");
    }
  }
  for (auto img : image_que) {
    if (coordinate_map_.find(img) != coordinate_map_.end()) image_ids_.push_back(img);
  }
  num_rows_ = image_ids_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API CocoDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

Status CocoOp::ImageColumnLoad(const nlohmann::json &image_tree, std::vector<std::string> *image_vec) {
  if (image_tree.size() == 0) {
    RETURN_STATUS_UNEXPECTED("Invalid data, no \"image\" node found in json file: " + annotation_path_);
  }
  for (auto img : image_tree) {
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
  if (search_category == category_set_.end())
    RETURN_STATUS_UNEXPECTED("Invalid data, category_id can't find in categories where category_id: " +
                             std::to_string(category_id));
  auto node_iscrowd = annotation_tree.find(kJsonAnnoIscrowd);
  if (node_iscrowd != annotation_tree.end()) iscrowd = *node_iscrowd;
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
      if (bbox.size() > 0) bbox.clear();
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
  if (itr_num_keypoint == annotation_tree.end())
    RETURN_STATUS_UNEXPECTED("Invalid data, no num_keypoint found in annotations where id: " +
                             std::to_string(unique_id));
  simple_item_map_[image_file].push_back(*itr_num_keypoint);
  auto itr_keypoint = annotation_tree.find(kJsonAnnoKeypoints);
  if (itr_keypoint == annotation_tree.end())
    RETURN_STATUS_UNEXPECTED("Invalid data, no keypoint found in annotations where id: " + std::to_string(unique_id));
  coordinate_map_[image_file].push_back(*itr_keypoint);
  return Status::OK();
}

Status CocoOp::PanopticColumnLoad(const nlohmann::json &annotation_tree, const std::string &image_file,
                                  const int32_t &image_id) {
  auto itr_segments = annotation_tree.find(kJsonAnnoSegmentsInfo);
  if (itr_segments == annotation_tree.end())
    RETURN_STATUS_UNEXPECTED("Invalid data, no segments_info found in annotations where image_id: " +
                             std::to_string(image_id));
  for (auto info : *itr_segments) {
    std::vector<float> bbox;
    uint32_t category_id = 0;
    auto itr_bbox = info.find(kJsonAnnoBbox);
    if (itr_bbox == info.end())
      RETURN_STATUS_UNEXPECTED("Invalid data, no bbox found in segments_info where image_id: " +
                               std::to_string(image_id));
    bbox.insert(bbox.end(), itr_bbox->begin(), itr_bbox->end());
    coordinate_map_[image_file].push_back(bbox);

    RETURN_IF_NOT_OK(SearchNodeInJson(info, std::string(kJsonAnnoCategoryId), &category_id));
    auto search_category = category_set_.find(category_id);
    if (search_category == category_set_.end())
      RETURN_STATUS_UNEXPECTED("Invalid data, category_id can't find in categories where category_id: " +
                               std::to_string(category_id));
    auto itr_iscrowd = info.find(kJsonAnnoIscrowd);
    if (itr_iscrowd == info.end())
      RETURN_STATUS_UNEXPECTED("Invalid data, no iscrowd found in segments_info where image_id: " +
                               std::to_string(image_id));
    auto itr_area = info.find(kJsonAnnoArea);
    if (itr_area == info.end())
      RETURN_STATUS_UNEXPECTED("Invalid data, no area found in segments_info where image_id: " +
                               std::to_string(image_id));
    simple_item_map_[image_file].push_back(category_id);
    simple_item_map_[image_file].push_back(*itr_iscrowd);
    simple_item_map_[image_file].push_back(*itr_area);
  }
  return Status::OK();
}

Status CocoOp::CategoriesColumnLoad(const nlohmann::json &categories_tree) {
  if (categories_tree.size() == 0) {
    RETURN_STATUS_UNEXPECTED("Invalid file, no categories found in annotation_path: " + annotation_path_);
  }
  for (auto category : categories_tree) {
    int32_t id = 0;
    std::string name;
    std::vector<int32_t> label_info;
    auto itr_id = category.find(kJsonId);
    if (itr_id == category.end()) {
      RETURN_STATUS_UNEXPECTED("Invalid data, no json id found in categories of " + annotation_path_);
    }
    id = *itr_id;
    label_info.push_back(id);
    category_set_.insert(id);

    auto itr_name = category.find(kJsonCategoriesName);
    CHECK_FAIL_RETURN_UNEXPECTED(
      itr_name != category.end(),
      "Invalid data, no categories name found in categories where id: " + std::to_string(id));
    name = *itr_name;

    if (task_type_ == TaskType::Panoptic) {
      auto itr_isthing = category.find(kJsonCategoriesIsthing);
      CHECK_FAIL_RETURN_UNEXPECTED(itr_isthing != category.end(),
                                   "Invalid data, nothing found in categories of " + annotation_path_);
      label_info.push_back(*itr_isthing);
    }
    label_index_.emplace_back(std::make_pair(name, label_info));
  }
  return Status::OK();
}

Status CocoOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

Status CocoOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&CocoOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->ParseAnnotationIds());
  RETURN_IF_NOT_OK(this->InitSampler());
  return Status::OK();
}

Status CocoOp::ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor) {
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(path, tensor));

  if (decode_ == true) {
    Status rc = Decode(*tensor, tensor);
    CHECK_FAIL_RETURN_UNEXPECTED(rc.IsOk(), "Invalid data, failed to decode image: " + path);
  }
  return Status::OK();
}

Status CocoOp::CountTotalRows(const std::string &dir, const std::string &file, const std::string &task,
                              int64_t *count) {
  std::shared_ptr<CocoOp> op;
  RETURN_IF_NOT_OK(Builder().SetDir(dir).SetFile(file).SetTask(task).Build(&op));
  RETURN_IF_NOT_OK(op->ParseAnnotationIds());
  *count = static_cast<int64_t>(op->image_ids_.size());
  return Status::OK();
}

Status CocoOp::GetClassIndexing(const std::string &dir, const std::string &file, const std::string &task,
                                std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  std::shared_ptr<CocoOp> op;
  RETURN_IF_NOT_OK(Builder().SetDir(dir).SetFile(file).SetTask(task).Build(&op));
  RETURN_IF_NOT_OK(op->ParseAnnotationIds());
  *output_class_indexing = op->label_index_;
  return Status::OK();
}

Status CocoOp::ComputeColMap() {
  // Set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->column(i).name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status CocoOp::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  if ((*output_class_indexing).empty()) {
    if ((task_type_ != TaskType::Detection) && (task_type_ != TaskType::Panoptic)) {
      MS_LOG(ERROR) << "Class index only valid in \"Detection\" and \"Panoptic\" task.";
      RETURN_STATUS_UNEXPECTED("GetClassIndexing: Get Class Index failed in CocoOp.");
    }
    std::shared_ptr<CocoOp> op;
    std::string task_type;
    switch (task_type_) {
      case TaskType::Detection:
        task_type = "Detection";
        break;
      case TaskType::Panoptic:
        task_type = "Panoptic";
        break;
    }
    RETURN_IF_NOT_OK(Builder().SetDir(image_folder_path_).SetFile(annotation_path_).SetTask(task_type).Build(&op));
    RETURN_IF_NOT_OK(op->ParseAnnotationIds());
    for (const auto label : op->label_index_) {
      (*output_class_indexing).emplace_back(std::make_pair(label.first, label.second));
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
