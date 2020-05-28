/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/engine/datasetops/source/voc_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include "./tinyxml2.h"
#include "common/utils.h"
#include "dataset/core/config_manager.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLError;
namespace mindspore {
namespace dataset {
const char kColumnImage[] = "image";
const char kColumnTarget[] = "target";
const char kColumnAnnotation[] = "annotation";
const char kJPEGImagesFolder[] = "/JPEGImages/";
const char kSegmentationClassFolder[] = "/SegmentationClass/";
const char kAnnotationsFolder[] = "/Annotations/";
const char kImageSetsSegmentation[] = "/ImageSets/Segmentation/";
const char kImageSetsMain[] = "/ImageSets/Main/";
const char kImageExtension[] = ".jpg";
const char kSegmentationExtension[] = ".png";
const char kAnnotationExtension[] = ".xml";
const char kImageSetsExtension[] = ".txt";

VOCOp::Builder::Builder() : builder_decode_(false), builder_num_samples_(0), builder_sampler_(nullptr) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status VOCOp::Builder::Build(std::shared_ptr<VOCOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (builder_sampler_ == nullptr) {
    builder_sampler_ = std::make_shared<SequentialSampler>();
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
      ColDescriptor(std::string(kColumnAnnotation), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  }
  *ptr = std::make_shared<VOCOp>(builder_task_type_, builder_task_mode_, builder_dir_, builder_labels_to_read_,
                                 builder_num_workers_, builder_rows_per_buffer_, builder_op_connector_size_,
                                 builder_num_samples_, builder_decode_, std::move(builder_schema_),
                                 std::move(builder_sampler_));
  return Status::OK();
}

Status VOCOp::Builder::SanityCheck() {
  Path dir(builder_dir_);
  std::string err_msg;
  err_msg += dir.IsDirectory() == false ? "VOC path is invalid or not set\n" : "";
  err_msg += builder_num_workers_ <= 0 ? "Num of parallel workers is set to 0 or negative\n" : "";
  err_msg += builder_num_samples_ < 0 ? "num_samples is negative\n" : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
}

VOCOp::VOCOp(const TaskType &task_type, const std::string &task_mode, const std::string &folder_path,
             const std::map<std::string, int32_t> &class_index, int32_t num_workers, int32_t rows_per_buffer,
             int32_t queue_size, int64_t num_samples, bool decode, std::unique_ptr<DataSchema> data_schema,
             std::shared_ptr<Sampler> sampler)
    : ParallelOp(num_workers, queue_size),
      decode_(decode),
      row_cnt_(0),
      buf_cnt_(0),
      num_rows_(0),
      num_samples_(num_samples),
      task_type_(task_type),
      task_mode_(task_mode),
      folder_path_(folder_path),
      class_index_(class_index),
      rows_per_buffer_(rows_per_buffer),
      sampler_(std::move(sampler)),
      data_schema_(std::move(data_schema)) {
  // Set the column name map (base class field)
  for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    column_name_id_map_[data_schema_->column(i).name()] = i;
  }
  io_block_queues_.Init(num_workers_, queue_size);
}

Status VOCOp::TraverseSampleIds(const std::shared_ptr<Tensor> &sample_ids, std::vector<int64_t> *keys) {
  for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
    if ((*itr) > num_rows_) continue;
    if (row_cnt_ == num_samples_) break;
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
  RETURN_IF_NOT_OK(sampler_->GetNextBuffer(&sampler_buffer));
  while (true) {
    std::vector<int64_t> keys;
    keys.reserve(rows_per_buffer_);
    while (sampler_buffer->eoe() == false) {
      std::shared_ptr<Tensor> sample_ids;
      RETURN_IF_NOT_OK(sampler_buffer->GetTensor(&sample_ids, 0, 0));
      if (sample_ids->type() != DataType(DataType::DE_INT64)) {
        RETURN_STATUS_UNEXPECTED("Sampler Tensor isn't int64");
      }
      RETURN_IF_NOT_OK(TraverseSampleIds(sample_ids, &keys));
      RETURN_IF_NOT_OK(sampler_->GetNextBuffer(&sampler_buffer));
    }
    if (keys.empty() == false) {
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
    }
    if (!BitTest(op_ctrl_flags_, kDeOpRepeated) || BitTest(op_ctrl_flags_, kDeOpLastRepeat)) {
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
      RETURN_IF_NOT_OK(wp_.Wait());
      wp_.Clear();
      RETURN_IF_NOT_OK(sampler_->GetNextBuffer(&sampler_buffer));
    }
  }
}

void VOCOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <VOCOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nVOC Directory: " << folder_path_ << "\n\n";
  }
}

Status VOCOp::Reset() {
  RETURN_IF_NOT_OK(sampler_->Reset());
  row_cnt_ = 0;
  wp_.Set();
  return Status::OK();
}

Status VOCOp::GetNumSamples(int64_t *num) const {
  if (num == nullptr || num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "There is no valid data matching the dataset API VOCDataset.Please check file path or dataset API "
      "validation first.");
  }
  (*num) = num_samples_;
  return Status::OK();
}

Status VOCOp::LoadTensorRow(const std::string &image_id, TensorRow *trow) {
  if (task_type_ == TaskType::Segmentation) {
    std::shared_ptr<Tensor> image, target;
    const std::string kImageFile =
      folder_path_ + std::string(kJPEGImagesFolder) + image_id + std::string(kImageExtension);
    const std::string kTargetFile =
      folder_path_ + std::string(kSegmentationClassFolder) + image_id + std::string(kSegmentationExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile, data_schema_->column(0), &image));
    RETURN_IF_NOT_OK(ReadImageToTensor(kTargetFile, data_schema_->column(1), &target));
    (*trow) = {std::move(image), std::move(target)};
  } else if (task_type_ == TaskType::Detection) {
    std::shared_ptr<Tensor> image, annotation;
    const std::string kImageFile =
      folder_path_ + std::string(kJPEGImagesFolder) + image_id + std::string(kImageExtension);
    const std::string kAnnotationFile =
      folder_path_ + std::string(kAnnotationsFolder) + image_id + std::string(kAnnotationExtension);
    RETURN_IF_NOT_OK(ReadImageToTensor(kImageFile, data_schema_->column(0), &image));
    RETURN_IF_NOT_OK(ReadAnnotationToTensor(kAnnotationFile, data_schema_->column(1), &annotation));
    (*trow) = {std::move(image), std::move(annotation)};
  }
  return Status::OK();
}

Status VOCOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  TensorRow trow;
  for (const uint64_t &key : keys) {
    RETURN_IF_NOT_OK(this->LoadTensorRow(image_ids_[key], &trow));
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
    if (io_block->eoe() == true) {
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
    image_sets_file =
      folder_path_ + std::string(kImageSetsSegmentation) + task_mode_ + std::string(kImageSetsExtension);
  } else if (task_type_ == TaskType::Detection) {
    image_sets_file = folder_path_ + std::string(kImageSetsMain) + task_mode_ + std::string(kImageSetsExtension);
  }
  std::ifstream in_file;
  in_file.open(image_sets_file);
  if (in_file.fail()) {
    RETURN_STATUS_UNEXPECTED("Fail to open file: " + image_sets_file);
  }
  std::string id;
  while (getline(in_file, id)) {
    if (id[id.size() - 1] == '\r') {
      image_ids_.push_back(id.substr(0, id.size() - 1));
    } else {
      image_ids_.push_back(id);
    }
  }
  in_file.close();
  image_ids_.shrink_to_fit();
  num_rows_ = image_ids_.size();
  num_samples_ = (num_samples_ == 0 || num_samples_ > num_rows_) ? num_rows_ : num_samples_;
  return Status::OK();
}

Status VOCOp::ParseAnnotationIds() {
  std::vector<std::string> new_image_ids;
  for (auto id : image_ids_) {
    const std::string kAnnotationName =
      folder_path_ + std::string(kAnnotationsFolder) + id + std::string(kAnnotationExtension);
    RETURN_IF_NOT_OK(ParseAnnotationBbox(kAnnotationName));
    if (label_map_.find(kAnnotationName) != label_map_.end()) {
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
  num_samples_ = (num_samples_ == 0 || num_samples_ > num_rows_) ? num_rows_ : num_samples_;
  return Status::OK();
}

Status VOCOp::ParseAnnotationBbox(const std::string &path) {
  if (!Path(path).Exists()) {
    RETURN_STATUS_UNEXPECTED("File is not found : " + path);
  }
  Bbox bbox;
  XMLDocument doc;
  XMLError e = doc.LoadFile(common::SafeCStr(path));
  if (e != XMLError::XML_SUCCESS) {
    RETURN_STATUS_UNEXPECTED("Xml load failed");
  }
  XMLElement *root = doc.RootElement();
  if (root == nullptr) {
    RETURN_STATUS_UNEXPECTED("Xml load root element error");
  }
  XMLElement *object = root->FirstChildElement("object");
  if (object == nullptr) {
    RETURN_STATUS_UNEXPECTED("No object find in " + path);
  }
  while (object != nullptr) {
    std::string label_name;
    uint32_t xmin = 0, ymin = 0, xmax = 0, ymax = 0, truncated = 0, difficult = 0;
    XMLElement *name_node = object->FirstChildElement("name");
    if (name_node != nullptr) label_name = name_node->GetText();
    XMLElement *truncated_node = object->FirstChildElement("truncated");
    if (truncated_node != nullptr) truncated = truncated_node->UnsignedText();
    XMLElement *difficult_node = object->FirstChildElement("difficult");
    if (difficult_node != nullptr) difficult = difficult_node->UnsignedText();

    XMLElement *bbox_node = object->FirstChildElement("bndbox");
    if (bbox_node != nullptr) {
      XMLElement *xmin_node = bbox_node->FirstChildElement("xmin");
      if (xmin_node != nullptr) xmin = xmin_node->UnsignedText();
      XMLElement *ymin_node = bbox_node->FirstChildElement("ymin");
      if (ymin_node != nullptr) ymin = ymin_node->UnsignedText();
      XMLElement *xmax_node = bbox_node->FirstChildElement("xmax");
      if (xmax_node != nullptr) xmax = xmax_node->UnsignedText();
      XMLElement *ymax_node = bbox_node->FirstChildElement("ymax");
      if (ymax_node != nullptr) ymax = ymax_node->UnsignedText();
    } else {
      RETURN_STATUS_UNEXPECTED("bndbox dismatch in " + path);
    }
    if (label_name != "" && (class_index_.empty() || class_index_.find(label_name) != class_index_.end()) && xmin > 0 &&
        ymin > 0 && xmax > xmin && ymax > ymin) {
      std::vector<uint32_t> bbox_list = {xmin, ymin, xmax - xmin, ymax - ymin, truncated, difficult};
      bbox.emplace_back(std::make_pair(label_name, bbox_list));
      label_index_[label_name] = 0;
    }
    object = object->NextSiblingElement("object");
  }
  if (bbox.size() > 0) label_map_[path] = bbox;
  return Status::OK();
}

Status VOCOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

Status VOCOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("tree_ not set");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wp_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&VOCOp::WorkerEntry, this, std::placeholders::_1)));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->ParseImageIds());
  if (task_type_ == TaskType::Detection) {
    RETURN_IF_NOT_OK(this->ParseAnnotationIds());
  }
  RETURN_IF_NOT_OK(this->InitSampler());
  return Status::OK();
}

Status VOCOp::ReadImageToTensor(const std::string &path, const ColDescriptor &col, std::shared_ptr<Tensor> *tensor) {
  std::ifstream fs;
  fs.open(path, std::ios::binary | std::ios::in);
  if (fs.fail()) {
    RETURN_STATUS_UNEXPECTED("Fail to open file: " + path);
  }
  int64_t num_elements = fs.seekg(0, std::ios::end).tellg();
  (void)fs.seekg(0, std::ios::beg);
  RETURN_IF_NOT_OK(
    Tensor::CreateTensor(tensor, col.tensorImpl(), TensorShape(std::vector<dsize_t>(1, num_elements)), col.type()));
  (void)fs.read(reinterpret_cast<char *>((*tensor)->GetMutableBuffer()), num_elements);
  fs.close();
  if (decode_ == true) {
    Status rc = Decode(*tensor, tensor);
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("fail to decode file: " + path);
    }
  }
  return Status::OK();
}

Status VOCOp::ReadAnnotationToTensor(const std::string &path, const ColDescriptor &col,
                                     std::shared_ptr<Tensor> *tensor) {
  Bbox bbox_info = label_map_[path];
  std::vector<uint32_t> bbox_row;
  dsize_t bbox_column_num = 0, bbox_num = 0;
  for (auto box : bbox_info) {
    if (label_index_.find(box.first) != label_index_.end()) {
      std::vector<uint32_t> bbox;
      if (class_index_.find(box.first) != class_index_.end()) {
        bbox.emplace_back(class_index_[box.first]);
      } else {
        bbox.emplace_back(label_index_[box.first]);
      }
      bbox.insert(bbox.end(), box.second.begin(), box.second.end());
      bbox_row.insert(bbox_row.end(), bbox.begin(), bbox.end());
      if (bbox_column_num == 0) {
        bbox_column_num = static_cast<dsize_t>(bbox.size());
      }
      bbox_num++;
    }
  }

  std::vector<dsize_t> bbox_dim = {bbox_num, bbox_column_num};
  RETURN_IF_NOT_OK(Tensor::CreateTensor(tensor, col.tensorImpl(), TensorShape(bbox_dim), col.type(),
                                        reinterpret_cast<unsigned char *>(&bbox_row[0])));
  return Status::OK();
}

// Derived from RandomAccessOp
Status VOCOp::GetNumRowsInDataset(int64_t *num) const {
  if (num == nullptr || num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "There is no valid data matching the dataset API VOCDataset.Please check file path or dataset API "
      "validation first.");
  }
  (*num) = num_rows_;
  return Status::OK();
}

Status VOCOp::CountTotalRows(const std::string &dir, const std::string &task_type, const std::string &task_mode,
                             const py::dict &dict, int64_t numSamples, int64_t *count) {
  if (task_type == "Detection") {
    std::map<std::string, int32_t> input_class_indexing;
    for (auto p : dict) {
      (void)input_class_indexing.insert(std::pair<std::string, int32_t>(py::reinterpret_borrow<py::str>(p.first),
                                                                        py::reinterpret_borrow<py::int_>(p.second)));
    }

    std::shared_ptr<VOCOp> op;
    RETURN_IF_NOT_OK(
      Builder().SetDir(dir).SetTask(task_type).SetMode(task_mode).SetClassIndex(input_class_indexing).Build(&op));
    RETURN_IF_NOT_OK(op->ParseImageIds());
    RETURN_IF_NOT_OK(op->ParseAnnotationIds());
    *count = static_cast<int64_t>(op->image_ids_.size());
  } else if (task_type == "Segmentation") {
    std::shared_ptr<VOCOp> op;
    RETURN_IF_NOT_OK(Builder().SetDir(dir).SetTask(task_type).SetMode(task_mode).Build(&op));
    RETURN_IF_NOT_OK(op->ParseImageIds());
    *count = static_cast<int64_t>(op->image_ids_.size());
  }
  *count = (numSamples == 0 || *count < numSamples) ? *count : numSamples;

  return Status::OK();
}

Status VOCOp::GetClassIndexing(const std::string &dir, const std::string &task_type, const std::string &task_mode,
                               const py::dict &dict, int64_t numSamples,
                               std::map<std::string, int32_t> *output_class_indexing) {
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
      Builder().SetDir(dir).SetTask(task_type).SetMode(task_mode).SetClassIndex(input_class_indexing).Build(&op));
    RETURN_IF_NOT_OK(op->ParseImageIds());
    RETURN_IF_NOT_OK(op->ParseAnnotationIds());
    for (const auto label : op->label_index_) {
      (*output_class_indexing).insert(std::make_pair(label.first, label.second));
    }
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
