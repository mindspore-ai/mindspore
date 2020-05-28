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
#include "dataset/engine/datasetops/source/manifest_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

#include "common/utils.h"
#include "dataset/core/config_manager.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
ManifestOp::Builder::Builder() : builder_sampler_(nullptr), builder_num_samples_(0), builder_decode_(false) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status ManifestOp::Builder::Build(std::shared_ptr<ManifestOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (builder_sampler_ == nullptr) {
    builder_sampler_ = std::make_shared<SequentialSampler>();
  }
  builder_schema_ = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    builder_schema_->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    builder_schema_->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  *ptr = std::make_shared<ManifestOp>(
    builder_num_workers_, builder_rows_per_buffer_, builder_file_, builder_op_connector_size_, builder_num_samples_,
    builder_decode_, builder_labels_to_read_, std::move(builder_schema_), std::move(builder_sampler_), builder_usage_);
  return Status::OK();
}

Status ManifestOp::Builder::SanityCheck() {
  std::string err_msg;
  err_msg += builder_file_.empty() ? "Manifest file is not set\n" : "";
  err_msg += builder_num_workers_ <= 0 ? "Num of parallel workers smaller than 1\n" : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
}

ManifestOp::ManifestOp(int32_t num_works, int32_t rows_per_buffer, std::string file, int32_t queue_size,
                       int64_t num_samples, bool decode, const std::map<std::string, int32_t> &class_index,
                       std::unique_ptr<DataSchema> data_schema, std::shared_ptr<Sampler> sampler, std::string usage)
    : ParallelOp(num_works, queue_size),
      rows_per_buffer_(rows_per_buffer),
      io_block_pushed_(0),
      row_cnt_(0),
      sampler_ind_(0),
      data_schema_(std::move(data_schema)),
      file_(file),
      class_index_(class_index),
      sampler_(std::move(sampler)),
      num_samples_(num_samples),
      num_rows_(0),
      decode_(decode),
      usage_(usage),
      buf_cnt_(0) {
  // Set the column name map (base class field)
  for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    column_name_id_map_[data_schema_->column(i).name()] = i;
  }
  io_block_queues_.Init(num_workers_, queue_size);
  (void)std::transform(usage_.begin(), usage_.end(), usage_.begin(), ::tolower);
}

// Main logic, Register Queue with TaskGroup, launch all threads and do the functor's work
Status ManifestOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  std::unique_ptr<DataBuffer> sampler_buffer;
  RETURN_IF_NOT_OK(sampler_->GetNextBuffer(&sampler_buffer));
  return AddIoBlock(&sampler_buffer);
}

Status ManifestOp::AddIoBlock(std::unique_ptr<DataBuffer> *sampler_buffer) {
  while (true) {  // each iterator is 1 epoch
    std::vector<int64_t> keys;
    keys.reserve(rows_per_buffer_);
    while (!(*sampler_buffer)->eoe()) {
      TensorRow sample_row;
      RETURN_IF_NOT_OK((*sampler_buffer)->PopRow(&sample_row));
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
        if ((*itr) >= num_rows_) continue;    // index out of bound, skipping
        if (row_cnt_ >= num_samples_) break;  // enough row read, break for loop
        keys.push_back(*itr);
        row_cnt_++;
        if (row_cnt_ % rows_per_buffer_ == 0) {
          RETURN_IF_NOT_OK(io_block_queues_[buf_cnt_++ % num_workers_]->Add(
            std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
          keys.clear();
        }
      }
      RETURN_IF_NOT_OK(sampler_->GetNextBuffer(sampler_buffer));
    }
    if (keys.empty() == false) {
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
    }
    if (!BitTest(op_ctrl_flags_, kDeOpRepeated) || BitTest(op_ctrl_flags_, kDeOpLastRepeat)) {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof)));
      for (int32_t i = 0; i < num_workers_; i++) {
        RETURN_IF_NOT_OK(
          io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
      }
      return Status::OK();
    } else {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
      RETURN_IF_NOT_OK(wp_.Wait());  // Master thread goes to sleep after it has made all the IOBlocks
      wp_.Clear();
      RETURN_IF_NOT_OK(sampler_->GetNextBuffer(sampler_buffer));
    }
  }
}

Status ManifestOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("tree_ not set");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wp_.Register(tree_->AllTasks()));

  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&ManifestOp::WorkerEntry, this, std::placeholders::_1)));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(ParseManifestFile());
  RETURN_IF_NOT_OK(CountDatasetInfo());
  RETURN_IF_NOT_OK(InitSampler());
  return Status::OK();
}

// contains the main logic of pulling a IOBlock from IOBlockQueue, load a buffer and push the buffer to out_connector_
// IMPORTANT: 1 IOBlock produces 1 DataBuffer
Status ManifestOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  int64_t buffer_id = worker_id;
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->eoe() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE)));
      buffer_id = worker_id;
    } else if (io_block->eof() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF)));
    } else {
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(io_block->GetKeys(&keys));
      if (keys.empty()) {
        return Status::OK();  // empty key is a quit signal for workers
      }
      std::unique_ptr<DataBuffer> db = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
      RETURN_IF_NOT_OK(LoadBuffer(keys, &db));
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db)));
      buffer_id += num_workers_;
    }
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("Unexpected nullptr received in worker");
}

// Load 1 TensorRow (image,label) using 1 ImageLabelPair. 1 function call produces 1 TensorTow in a DataBuffer
Status ManifestOp::LoadTensorRow(const std::pair<std::string, std::vector<std::string>> &data, TensorRow *trow) {
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> label;
  std::vector<int32_t> label_index(data.second.size());
  (void)std::transform(data.second.begin(), data.second.end(), label_index.begin(),
                       [this](const std::string &label_name) { return label_index_[label_name]; });
  if (label_index.size() == 1) {
    RETURN_IF_NOT_OK(Tensor::CreateTensor(&label, data_schema_->column(1).tensorImpl(), TensorShape({}),
                                          data_schema_->column(1).type(),
                                          reinterpret_cast<unsigned char *>(&label_index[0])));
  } else {
    RETURN_IF_NOT_OK(Tensor::CreateTensor(
      &label, data_schema_->column(1).tensorImpl(), TensorShape(std::vector<dsize_t>(1, label_index.size())),
      data_schema_->column(1).type(), reinterpret_cast<unsigned char *>(&label_index[0])));
  }

  std::ifstream fs;
  fs.open(data.first, std::ios::binary | std::ios::in);
  if (!fs.is_open()) {
    RETURN_STATUS_UNEXPECTED("Fail to open file: " + data.first);
  }

  int64_t num_elements = fs.seekg(0, std::ios::end).tellg();
  (void)fs.seekg(0, std::ios::beg);
  RETURN_IF_NOT_OK(Tensor::CreateTensor(&image, data_schema_->column(0).tensorImpl(),
                                        TensorShape(std::vector<dsize_t>(1, num_elements)),
                                        data_schema_->column(0).type(), nullptr));
  (void)fs.read(reinterpret_cast<char *>(image->GetMutableBuffer()), num_elements);
  if (fs.fail()) {
    fs.close();
    RETURN_STATUS_UNEXPECTED("Fail to read file: " + data.first);
  }
  fs.close();
  if (decode_ == true) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Fail to decode image:" + data.first;
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  (*trow) = {std::move(image), std::move(label)};
  return Status::OK();
}

// Looping over LoadTensorRow to make 1 DataBuffer. 1 function call produces 1 buffer
Status ManifestOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  for (const auto &key : keys) {
    TensorRow trow;
    RETURN_IF_NOT_OK(LoadTensorRow(image_labelname_[static_cast<size_t>(key)], &trow));
    deq->push_back(std::move(trow));
  }
  (*db)->set_tensor_table(std::move(deq));
  return Status::OK();
}

void ManifestOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <ManifestOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nManifest file: " << file_ << "\n\n";
  }
}

// Reset Sampler and wakeup Master thread (functor)
Status ManifestOp::Reset() {
  RETURN_IF_NOT_OK(sampler_->Reset());
  row_cnt_ = 0;
  wp_.Set();  // wake up master thread after reset is done
  return Status::OK();
}

// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status ManifestOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

// Derived from RandomAccessOp
Status ManifestOp::GetNumSamples(int64_t *num) const {
  if (num == nullptr || num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "There is no valid data matching the dataset API ManifestDataset.Please check file path or dataset API "
      "validation first.");
  }
  (*num) = num_samples_;
  return Status::OK();
}

// Derived from RandomAccessOp
Status ManifestOp::GetNumRowsInDataset(int64_t *num) const {
  if (num == nullptr || num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "There is no valid data matching the dataset API ManifestDataset.Please check file path or dataset API "
      "validation first.");
  }
  (*num) = num_rows_;
  return Status::OK();
}

// Derived from RandomAccessOp
Status ManifestOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_labelname_.empty()) {
    RETURN_STATUS_UNEXPECTED("Class indexing is invalid.");
  }

  for (size_t i = 0; i < image_labelname_.size(); i++) {
    size_t image_index = i;
    for (size_t j = 0; j < image_labelname_[image_index].second.size(); j++) {
      std::string label_name = (image_labelname_[image_index].second)[j];
      int32_t label_index = label_index_.at(label_name);
      (*cls_ids)[label_index].emplace_back(image_index);
    }
  }

  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

// Manifest file content
// {"source": "/path/to/image1.jpg", "usage":"train", annotation": ...}
// {"source": "/path/to/image2.jpg", "usage":"eval", "annotation": ...}
Status ManifestOp::ParseManifestFile() {
  std::ifstream file_handle(file_);
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Manifest file " + file_ + " can not open.");
  }
  std::string line;
  while (getline(file_handle, line)) {
    try {
      nlohmann::json js = nlohmann::json::parse(line);
      std::string image_file_path = js.value("source", "");
      // If image is not JPEG/PNG/GIF/BMP, drop it
      bool valid = false;
      RETURN_IF_NOT_OK(CheckImageType(image_file_path, &valid));
      if (!valid) {
        continue;
      }
      std::string usage = js.value("usage", "");
      (void)std::transform(usage.begin(), usage.end(), usage.begin(), ::tolower);
      if (usage != usage_) {
        continue;
      }
      std::vector<std::string> labels;
      nlohmann::json annotations = js.at("annotation");
      for (nlohmann::json::iterator it = annotations.begin(); it != annotations.end(); ++it) {
        nlohmann::json annotation = it.value();
        std::string label_name = annotation.value("name", "");
        if (label_name == "") {
          file_handle.close();
          RETURN_STATUS_UNEXPECTED("Label name is not found in manifest file for " + image_file_path);
        }
        if (class_index_.empty() || class_index_.find(label_name) != class_index_.end()) {
          if (label_index_.find(label_name) == label_index_.end()) {
            label_index_[label_name] = 0;
          }
          labels.emplace_back(label_name);
        }
      }
      if (!labels.empty()) {
        image_labelname_.emplace_back(std::make_pair(image_file_path, labels));
      }
    } catch (const std::exception &err) {
      file_handle.close();
      RETURN_STATUS_UNEXPECTED("Parse manifest file failed");
    }
  }
  file_handle.close();

  return Status::OK();
}

// Only support JPEG/PNG/GIF/BMP
Status ManifestOp::CheckImageType(const std::string &file_name, bool *valid) {
  std::ifstream file_handle;
  constexpr int read_num = 3;
  *valid = false;
  file_handle.open(file_name, std::ios::binary | std::ios::in);
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Can not open image file " + file_name);
  }
  unsigned char file_type[read_num];
  (void)file_handle.read(reinterpret_cast<char *>(file_type), read_num);

  if (file_handle.fail()) {
    file_handle.close();
    RETURN_STATUS_UNEXPECTED("Read image file failed " + file_name);
  }
  file_handle.close();
  if (file_type[0] == 0xff && file_type[1] == 0xd8 && file_type[2] == 0xff) {
    // Normal JPEGs start with \xff\xd8\xff\xe0
    // JPEG with EXIF stats with \xff\xd8\xff\xe1
    // Use \xff\xd8\xff to cover both.
    *valid = true;
  } else if (file_type[0] == 0x89 && file_type[1] == 0x50 && file_type[2] == 0x4e) {
    // It's a PNG
    *valid = true;
  } else if (file_type[0] == 0x47 && file_type[1] == 0x49 && file_type[2] == 0x46) {
    // It's a GIF
    *valid = true;
  } else if (file_type[0] == 0x42 && file_type[1] == 0x4d) {
    // It's a BMP
    *valid = true;
  }
  return Status::OK();
}

Status ManifestOp::CountDatasetInfo() {
  int32_t index = 0;
  for (auto &label : label_index_) {
    label.second = class_index_.empty() ? index : class_index_[label.first];
    index++;
  }

  num_rows_ = static_cast<int64_t>(image_labelname_.size());
  num_samples_ = (num_samples_ == 0 || num_samples_ > num_rows_) ? num_rows_ : num_samples_;
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "There is no valid data matching the dataset API ManifestDataset.Please check file path or dataset API "
      "validation first.");
  }
  return Status::OK();
}

Status ManifestOp::CountTotalRows(const std::string &file, int64_t numSamples, const py::dict &dict,
                                  const std::string &usage, int64_t *count, int64_t *numClasses) {
  // the logic of counting the number of samples is copied from ParseManifestFile()
  std::map<std::string, int32_t> map;
  for (auto p : dict) {
    (void)map.insert(std::pair<std::string, int32_t>(py::reinterpret_borrow<py::str>(p.first),
                                                     py::reinterpret_borrow<py::int_>(p.second)));
  }

  std::shared_ptr<ManifestOp> op;
  *count = 0;
  RETURN_IF_NOT_OK(
    Builder().SetManifestFile(file).SetNumSamples(numSamples).SetClassIndex(map).SetUsage(usage).Build(&op));
  RETURN_IF_NOT_OK(op->ParseManifestFile());
  *numClasses = static_cast<int64_t>(op->label_index_.size());
  *count = static_cast<int64_t>(op->image_labelname_.size());
  *count = (*count < numSamples || numSamples == 0) ? *count : numSamples;
  return Status::OK();
}

Status ManifestOp::GetClassIndexing(const std::string &file, int64_t numSamples, const py::dict &dict,
                                    const std::string &usage, std::map<std::string, int32_t> *output_class_indexing) {
  std::map<std::string, int32_t> input_class_indexing;
  for (auto p : dict) {
    (void)input_class_indexing.insert(std::pair<std::string, int32_t>(py::reinterpret_borrow<py::str>(p.first),
                                                                      py::reinterpret_borrow<py::int_>(p.second)));
  }

  if (!input_class_indexing.empty()) {
    *output_class_indexing = input_class_indexing;
  } else {
    std::shared_ptr<ManifestOp> op;
    RETURN_IF_NOT_OK(Builder()
                       .SetManifestFile(file)
                       .SetNumSamples(numSamples)
                       .SetClassIndex(input_class_indexing)
                       .SetUsage(usage)
                       .Build(&op));
    RETURN_IF_NOT_OK(op->ParseManifestFile());
    RETURN_IF_NOT_OK(op->CountDatasetInfo());
    uint32_t count = 0;
    for (const auto label : op->label_index_) {
      (*output_class_indexing).insert(std::make_pair(label.first, count));
      count++;
    }
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
