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

#include <fstream>

#include "common/utils.h"
#include "dataset/core/config_manager.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
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
  RETURN_IF_NOT_OK(
    builder_schema_->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    builder_schema_->AddColumn(ColDescriptor("target", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  *ptr = std::make_shared<VOCOp>(builder_num_workers_, builder_rows_per_buffer_, builder_dir_,
                                 builder_op_connector_size_, builder_num_samples_, builder_decode_,
                                 std::move(builder_schema_), std::move(builder_sampler_));
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

VOCOp::VOCOp(int32_t num_workers, int32_t rows_per_buffer, const std::string &folder_path, int32_t queue_size,
             int64_t num_samples, bool decode, std::unique_ptr<DataSchema> data_schema,
             std::shared_ptr<Sampler> sampler)
    : ParallelOp(num_workers, queue_size),
      decode_(decode),
      row_cnt_(0),
      buf_cnt_(0),
      num_rows_(0),
      num_samples_(num_samples),
      folder_path_(folder_path),
      rows_per_buffer_(rows_per_buffer),
      sampler_(std::move(sampler)),
      data_schema_(std::move(data_schema)) {
  for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    col_name_map_[data_schema_->column(i).name()] = i;
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
  DatasetOp::Print(out, show_all);
  out << "\nnumber of parallel workers:" << num_workers_ << "\nNumber of rows:" << num_rows_
      << "\nVOC Directory: " << folder_path_ << "\n-------------------\n";
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
  std::shared_ptr<Tensor> image, target;
  const std::string kImageDir = folder_path_ + "/JPEGImages/" + image_id + ".jpg";
  const std::string kTargetDir = folder_path_ + "/SegmentationClass/" + image_id + ".png";
  RETURN_IF_NOT_OK(ReadImageToTensor(kImageDir, data_schema_->column(0), &image));
  RETURN_IF_NOT_OK(ReadImageToTensor(kTargetDir, data_schema_->column(1), &target));
  (*trow) = {std::move(image), std::move(target)};
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
  (*db)->set_column_name_map(col_name_map_);
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
  const std::string kImageSets = "/ImageSets/Segmentation/train.txt";
  std::string image_sets_file = folder_path_ + kImageSets;
  std::ifstream in_file;
  in_file.open(image_sets_file);
  if (in_file.fail()) {
    RETURN_STATUS_UNEXPECTED("Fail to open file: " + image_sets_file);
  }
  std::string id;
  while (getline(in_file, id)) {
    image_ids_.push_back(id);
  }
  in_file.close();
  image_ids_.shrink_to_fit();
  num_rows_ = image_ids_.size();
  num_samples_ = (num_samples_ == 0 || num_samples_ > num_rows_) ? num_rows_ : num_samples_;
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
  wp_.Register(tree_->AllTasks());
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&VOCOp::WorkerEntry, this, std::placeholders::_1)));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->ParseImageIds());
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
  (void)fs.read(reinterpret_cast<char *>((*tensor)->StartAddr()), num_elements);
  fs.close();
  if (decode_ == true) {
    Status rc = Decode(*tensor, tensor);
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("fail to decode file: " + path);
    }
  }
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
}  // namespace dataset
}  // namespace mindspore
