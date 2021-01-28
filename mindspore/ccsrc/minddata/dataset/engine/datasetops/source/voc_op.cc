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
#include <iomanip>

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

VOCOp::Builder::Builder() : builder_decode_(false), builder_sampler_(nullptr) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
  builder_task_type_ = TaskType::Segmentation;
}

Status VOCOp::Builder::Build(std::shared_ptr<VOCOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (builder_sampler_ == nullptr) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    builder_sampler_ = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }
  builder_schema_ = std::make_unique<DataSchema>();
  if (builder_task_type_ == TaskType::Segmentation) {
    RETURN_IF_NOT_OK(builder_schema_->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(builder_schema_->AddColumn(
      ColDescriptor(std::string(kColumnTarget), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  } else if (builder_task_type_ == TaskType::Detection) {
    RETURN_IF_NOT_OK(builder_schema_->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(builder_schema_->AddColumn(
      ColDescriptor(std::string(kColumnBbox), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(builder_schema_->AddColumn(
      ColDescriptor(std::string(kColumnLabel), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(builder_schema_->AddColumn(
      ColDescriptor(std::string(kColumnDifficult), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(builder_schema_->AddColumn(
      ColDescriptor(std::string(kColumnTruncate), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  }
  *ptr = std::make_shared<VOCOp>(builder_task_type_, builder_usage_, builder_dir_, builder_labels_to_read_,
                                 builder_num_workers_, builder_rows_per_buffer_, builder_op_connector_size_,
                                 builder_decode_, std::move(builder_schema_), std::move(builder_sampler_));
  return Status::OK();
}

Status VOCOp::Builder::SanityCheck() {
  Path dir(builder_dir_);
  std::string err_msg;
  err_msg += dir.IsDirectory() == false
               ? "Invalid parameter, VOC path is invalid or not set, path: " + builder_dir_ + ".\n"
               : "";
  err_msg += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                           std::to_string(builder_num_workers_) + ".\n"
                                       : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err_msg);
}

VOCOp::VOCOp(const TaskType &task_type, const std::string &task_mode, const std::string &folder_path,
             const std::map<std::string, int32_t> &class_index, int32_t num_workers, int32_t rows_per_buffer,
             int32_t queue_size, bool decode, std::unique_ptr<DataSchema> data_schema,
             std::shared_ptr<SamplerRT> sampler)
    : ParallelOp(num_workers, queue_size, std::move(sampler)),
      decode_(decode),
      row_cnt_(0),
      buf_cnt_(0),
      task_type_(task_type),
      usage_(task_mode),
      folder_path_(folder_path),
      class_index_(class_index),
      rows_per_buffer_(rows_per_buffer),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers_, queue_size);
}

Status VOCOp::TraverseSampleIds(const std::shared_ptr<Tensor> &sample_ids, std::vector<int64_t> *keys) {
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

Status VOCOp::operator()() {
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

Status VOCOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(sampler_->ResetSampler());
  row_cnt_ = 0;
  return Status::OK();
}

Status VOCOp::LoadTensorRow(row_id_type row_id, const std::string &image_id, TensorRow *trow) {
  if (task_type_ == TaskType::Segmentation) {
    std::shared_ptr<Tensor> image, target;
    const std::string kImageFile =
      folder_path_ + std::string(kJPEGImagesFolder) + image_id + std::string(kImageExtension);
    const std::string kTargetFile =
      folder_path_ + std::string(kSegmentationClassFolder) + image_id + std::string(kSegmentationExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile, data_schema_->column(0), &image));
    RETURN_IF_NOT_OK(ReadImageToTensor(kTargetFile, data_schema_->column(1), &target));
    (*trow) = TensorRow(row_id, {std::move(image), std::move(target)});
    trow->setPath({kImageFile, kTargetFile});
  } else if (task_type_ == TaskType::Detection) {
    std::shared_ptr<Tensor> image;
    TensorRow annotation;
    const std::string kImageFile =
      folder_path_ + std::string(kJPEGImagesFolder) + image_id + std::string(kImageExtension);
    const std::string kAnnotationFile =
      folder_path_ + std::string(kAnnotationsFolder) + image_id + std::string(kAnnotationExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile, data_schema_->column(0), &image));
    RETURN_IF_NOT_OK(ReadAnnotationToTensor(kAnnotationFile, &annotation));
    trow->setId(row_id);
    trow->setPath({kImageFile, kAnnotationFile, kAnnotationFile, kAnnotationFile, kAnnotationFile});
    trow->push_back(std::move(image));
    trow->insert(trow->end(), annotation.begin(), annotation.end());
  }
  return Status::OK();
}

Status VOCOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  TensorRow trow;
  for (const uint64_t &key : keys) {
    RETURN_IF_NOT_OK(this->LoadTensorRow(key, image_ids_[key], &trow));
    deq->push_back(std::move(trow));
  }
  (*db)->set_tensor_table(std::move(deq));
  return Status::OK();
}

Status VOCOp::WorkerEntry(int32_t worker_id) {
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

Status VOCOp::ParseImageIds() {
  std::string image_sets_file;
  if (task_type_ == TaskType::Segmentation) {
    image_sets_file = folder_path_ + std::string(kImageSetsSegmentation) + usage_ + std::string(kImageSetsExtension);
  } else if (task_type_ == TaskType::Detection) {
    image_sets_file = folder_path_ + std::string(kImageSetsMain) + usage_ + std::string(kImageSetsExtension);
  }
  std::ifstream in_file;
  in_file.open(image_sets_file);
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
    const std::string kAnnotationName =
      folder_path_ + std::string(kAnnotationsFolder) + id + std::string(kAnnotationExtension);
    RETURN_IF_NOT_OK(ParseAnnotationBbox(kAnnotationName));
    if (annotation_map_.find(kAnnotationName) != annotation_map_.end()) {
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
      "Invalid data, no valid data matching the dataset API VOCDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

void VOCOp::ParseNodeValue(XMLElement *bbox_node, const char *name, float *value) {
  *value = 0.0;
  if (bbox_node != nullptr) {
    XMLElement *node = bbox_node->FirstChildElement(name);
    if (node != nullptr) *value = node->FloatText();
  }
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
  if (annotation.size() > 0) annotation_map_[path] = annotation;
  return Status::OK();
}

Status VOCOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
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

Status VOCOp::CountTotalRows(const std::string &dir, const std::string &task_type, const std::string &task_mode,
                             const std::map<std::string, int32_t> &input_class_indexing, int64_t *count) {
  if (task_type == "Detection") {
    std::shared_ptr<VOCOp> op;
    RETURN_IF_NOT_OK(
      Builder().SetDir(dir).SetTask(task_type).SetUsage(task_mode).SetClassIndex(input_class_indexing).Build(&op));
    RETURN_IF_NOT_OK(op->ParseImageIds());
    RETURN_IF_NOT_OK(op->ParseAnnotationIds());
    *count = static_cast<int64_t>(op->image_ids_.size());
  } else if (task_type == "Segmentation") {
    std::shared_ptr<VOCOp> op;
    RETURN_IF_NOT_OK(Builder().SetDir(dir).SetTask(task_type).SetUsage(task_mode).Build(&op));
    RETURN_IF_NOT_OK(op->ParseImageIds());
    *count = static_cast<int64_t>(op->image_ids_.size());
  }

  return Status::OK();
}

#ifdef ENABLE_PYTHON
Status VOCOp::GetClassIndexing(const std::string &dir, const std::string &task_type, const std::string &task_mode,
                               const py::dict &dict, std::map<std::string, int32_t> *output_class_indexing) {
  std::map<std::string, int32_t> input_class_indexing;
  for (auto p : dict) {
    (void)input_class_indexing.insert(std::pair<std::string, int32_t>(py::reinterpret_borrow<py::str>(p.first),
                                                                      py::reinterpret_borrow<py::int_>(p.second)));
  }

  if (!input_class_indexing.empty()) {
    *output_class_indexing = input_class_indexing;
  } else {
    std::shared_ptr<VOCOp> op;
    RETURN_IF_NOT_OK(
      Builder().SetDir(dir).SetTask(task_type).SetUsage(task_mode).SetClassIndex(input_class_indexing).Build(&op));
    RETURN_IF_NOT_OK(op->ParseImageIds());
    RETURN_IF_NOT_OK(op->ParseAnnotationIds());
    for (const auto label : op->label_index_) {
      (*output_class_indexing).insert(std::make_pair(label.first, label.second));
    }
  }

  return Status::OK();
}
#endif

Status VOCOp::ComputeColMap() {
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

Status VOCOp::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  if ((*output_class_indexing).empty()) {
    if (task_type_ != TaskType::Detection) {
      MS_LOG(ERROR) << "Class index only valid in \"Detection\" task.";
      RETURN_STATUS_UNEXPECTED("GetClassIndexing: Get Class Index failed in VOCOp.");
    }
    std::shared_ptr<VOCOp> op;
    RETURN_IF_NOT_OK(
      Builder().SetDir(folder_path_).SetTask("Detection").SetUsage(usage_).SetClassIndex(class_index_).Build(&op));
    RETURN_IF_NOT_OK(op->ParseImageIds());
    RETURN_IF_NOT_OK(op->ParseAnnotationIds());
    for (const auto label : op->label_index_) {
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
