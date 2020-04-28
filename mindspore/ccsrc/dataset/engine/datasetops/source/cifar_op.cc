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
#include "dataset/engine/datasetops/source/cifar_op.h"

#include <algorithm>
#include <fstream>
#include <utility>

#include "common/utils.h"
#include "dataset/core/config_manager.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
constexpr uint32_t kCifarImageHeight = 32;
constexpr uint32_t kCifarImageWidth = 32;
constexpr uint32_t kCifarImageChannel = 3;
constexpr uint32_t kCifarBlockImageNum = 5;
constexpr uint32_t kCifarImageSize = kCifarImageHeight * kCifarImageWidth * kCifarImageChannel;

CifarOp::Builder::Builder() : num_samples_(0), sampler_(nullptr) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  num_workers_ = cfg->num_parallel_workers();
  rows_per_buffer_ = cfg->rows_per_buffer();
  op_connect_size_ = cfg->op_connector_size();
  cifar_type_ = kCifar10;
}

Status CifarOp::Builder::Build(std::shared_ptr<CifarOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (sampler_ == nullptr) {
    sampler_ = std::make_shared<SequentialSampler>();
  }
  schema_ = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema_->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  if (cifar_type_ == kCifar10) {
    RETURN_IF_NOT_OK(
      schema_->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  } else {
    RETURN_IF_NOT_OK(schema_->AddColumn(
      ColDescriptor("coarse_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
    TensorShape another_scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(schema_->AddColumn(
      ColDescriptor("fine_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &another_scalar)));
  }

  *ptr = std::make_shared<CifarOp>(cifar_type_, num_workers_, rows_per_buffer_, dir_, op_connect_size_, num_samples_,
                                   std::move(schema_), std::move(sampler_));
  return Status::OK();
}

Status CifarOp::Builder::SanityCheck() {
  Path dir(dir_);
  std::string err_msg;
  err_msg += dir.IsDirectory() == false ? "Cifar path is invalid or not set\n" : "";
  err_msg += num_workers_ <= 0 ? "Num of parallel workers is negative or 0\n" : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
}

CifarOp::CifarOp(CifarType type, int32_t num_works, int32_t rows_per_buf, const std::string &file_dir,
                 int32_t queue_size, int64_t num_samples, std::unique_ptr<DataSchema> data_schema,
                 std::shared_ptr<Sampler> sampler)
    : ParallelOp(num_works, queue_size),
      cifar_type_(type),
      rows_per_buffer_(rows_per_buf),
      folder_path_(file_dir),
      num_samples_(num_samples),
      data_schema_(std::move(data_schema)),
      sampler_(std::move(sampler)),
      num_rows_(0),
      row_cnt_(0),
      buf_cnt_(0) {
  for (uint32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    col_name_map_[data_schema_->column(i).name()] = i;
  }
  constexpr uint64_t kUtilQueueSize = 512;
  cifar_raw_data_block_ = std::make_unique<Queue<std::vector<unsigned char>>>(kUtilQueueSize);
  io_block_queues_.Init(num_workers_, queue_size);
}

// Main logic, Register Queue with TaskGroup, launch all threads and do the functor's work
Status CifarOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  std::unique_ptr<DataBuffer> sampler_buffer;
  RETURN_IF_NOT_OK(sampler_->GetNextBuffer(&sampler_buffer));
  while (true) {  // each iterator is 1 epoch
    std::vector<int64_t> keys;
    keys.reserve(rows_per_buffer_);
    while (sampler_buffer->eoe() == false) {
      TensorRow sample_row;
      RETURN_IF_NOT_OK(sampler_buffer->PopRow(&sample_row));
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); itr++) {
        keys.push_back(*itr);
        row_cnt_++;
        if ((*itr) >= num_rows_) continue;    // index out of bound, skipping
        if (row_cnt_ >= num_samples_) break;  // enough row read, break for loop
        if (row_cnt_ % rows_per_buffer_ == 0) {
          RETURN_IF_NOT_OK(io_block_queues_[buf_cnt_++ % num_workers_]->Add(
            std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
          keys.clear();
        }
      }
      RETURN_IF_NOT_OK(sampler_->GetNextBuffer(&sampler_buffer));
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
    } else {  // not the last repeat. Acquire lock, sleeps master thread, wait for the wake-up from reset
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
      RETURN_IF_NOT_OK(wp_.Wait());  // Master thread goes to sleep after it has made all the IOBlocks
      wp_.Clear();
      RETURN_IF_NOT_OK(sampler_->GetNextBuffer(&sampler_buffer));
    }
  }
}

Status CifarOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("tree_ not set");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  wp_.Register(tree_->AllTasks());
  RETURN_IF_NOT_OK(
    tree_->AllTasks()->CreateAsyncTask("Get cifar data block", std::bind(&CifarOp::ReadCifarBlockDataAsync, this)));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&CifarOp::WorkerEntry, this, std::placeholders::_1)));
  TaskManager::FindMe()->Post();
  // The order of the following 2 functions must not be changed!
  RETURN_IF_NOT_OK(ParseCifarData());  // Parse cifar data and get num rows, blocking
  RETURN_IF_NOT_OK(InitSampler());     // Pass numRows to Sampler
  return Status::OK();
}

// contains the main logic of pulling a IOBlock from IOBlockQueue, load a buffer and push the buffer to out_connector_
// IMPORTANT: 1 IOBlock produces 1 DataBuffer
Status CifarOp::WorkerEntry(int32_t worker_id) {
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
      if (keys.empty() == true) {
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

// Load 1 TensorRow (image,label). 1 function call produces 1 TensorTow in a DataBuffer
Status CifarOp::LoadTensorRow(uint64_t index, TensorRow *trow) {
  std::shared_ptr<Tensor> label;
  std::shared_ptr<Tensor> fine_label;
  std::shared_ptr<Tensor> ori_image = cifar_image_label_pairs_[index].first;
  std::shared_ptr<Tensor> copy_image =
    std::make_shared<Tensor>(ori_image->shape(), ori_image->type(), ori_image->StartAddr());
  RETURN_IF_NOT_OK(Tensor::CreateTensor(&label, data_schema_->column(1).tensorImpl(), data_schema_->column(1).shape(),
                                        data_schema_->column(1).type(),
                                        reinterpret_cast<unsigned char *>(&cifar_image_label_pairs_[index].second[0])));
  if (cifar_image_label_pairs_[index].second.size() > 1) {
    RETURN_IF_NOT_OK(Tensor::CreateTensor(
      &fine_label, data_schema_->column(2).tensorImpl(), data_schema_->column(2).shape(),
      data_schema_->column(2).type(), reinterpret_cast<unsigned char *>(&cifar_image_label_pairs_[index].second[1])));
    (*trow) = {copy_image, std::move(label), std::move(fine_label)};
  } else {
    (*trow) = {copy_image, std::move(label)};
  }

  return Status::OK();
}

// Looping over LoadTensorRow to make 1 DataBuffer. 1 function call produces 1 buffer
Status CifarOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  for (const int64_t &key : keys) {
    TensorRow trow;
    RETURN_IF_NOT_OK(LoadTensorRow(key, &trow));
    deq->push_back(std::move(trow));
  }
  (*db)->set_tensor_table(std::move(deq));
  (*db)->set_column_name_map(col_name_map_);
  return Status::OK();
}

void CifarOp::Print(std::ostream &out, bool show_all) const {
  DatasetOp::Print(out, show_all);
  out << "\nnumber of parallel workers:" << num_workers_ << "\nNumber of rows:" << num_rows_
      << "\nCifar Directory: " << folder_path_ << "\n-------------------------\n";
}

// Reset Sampler and wakeup Master thread (functor)
Status CifarOp::Reset() {
  RETURN_IF_NOT_OK(sampler_->Reset());
  row_cnt_ = 0;
  wp_.Set();  // wake up master thread after reset is done
  return Status::OK();
}

// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status CifarOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

// Derived from RandomAccessOp
Status CifarOp::GetNumSamples(int64_t *num) const {
  if (num == nullptr || num_rows_ == 0) {
    std::string api = cifar_type_ == kCifar10 ? "Cifar10Dataset" : "Cifar100Dataset";
    std::string err_msg = "There is no valid data matching the dataset API " + api +
                          ".Please check file path or dataset API validation first.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  (*num) = num_samples_;
  return Status::OK();
}

// Derived from RandomAccessOp
Status CifarOp::GetNumRowsInDataset(int64_t *num) const {
  if (num == nullptr || num_rows_ == 0) {
    std::string api = cifar_type_ == kCifar10 ? "Cifar10Dataset" : "Cifar100Dataset";
    std::string err_msg = "There is no valid data matching the dataset API " + api +
                          ".Please check file path or dataset API validation first.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  (*num) = num_rows_;
  return Status::OK();
}

Status CifarOp::ReadCifarBlockDataAsync() {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(GetCifarFiles());
  if (cifar_type_ == kCifar10) {
    RETURN_IF_NOT_OK(ReadCifar10BlockData());
  } else {
    RETURN_IF_NOT_OK(ReadCifar100BlockData());
  }

  return Status::OK();
}

Status CifarOp::ReadCifar10BlockData() {
  constexpr uint32_t num_cifar10_records = 10000;
  uint32_t block_size = (kCifarImageSize + 1) * kCifarBlockImageNum;  // about 2M
  std::vector<unsigned char> image_data(block_size * sizeof(unsigned char), 0);
  for (auto &file : cifar_files_) {
    std::ifstream in(file, std::ios::binary);
    if (!in.is_open()) {
      std::string err_msg = file + " can not be opened.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    for (uint32_t index = 0; index < num_cifar10_records / kCifarBlockImageNum; ++index) {
      (void)in.read(reinterpret_cast<char *>(&(image_data[0])), block_size * sizeof(unsigned char));
      if (in.fail()) {
        RETURN_STATUS_UNEXPECTED("Fail to read cifar file" + file);
      }
      (void)cifar_raw_data_block_->EmplaceBack(image_data);
    }
    in.close();
  }
  (void)cifar_raw_data_block_->EmplaceBack(std::vector<unsigned char>());  // end block

  return Status::OK();
}

Status CifarOp::ReadCifar100BlockData() {
  uint32_t num_cifar100_records = 0;                                  // test:10000, train:50000
  uint32_t block_size = (kCifarImageSize + 2) * kCifarBlockImageNum;  // about 2M
  std::vector<unsigned char> image_data(block_size * sizeof(unsigned char), 0);
  for (auto &file : cifar_files_) {
    int pos = file.find_last_of('/');
    if (pos == std::string::npos) {
      RETURN_STATUS_UNEXPECTED("Invalid cifar100 file path");
    }
    std::string file_name(file.substr(pos + 1));
    if (file_name.find("test") != std::string::npos) {
      num_cifar100_records = 10000;
    } else if (file_name.find("train") != std::string::npos) {
      num_cifar100_records = 50000;
    } else {
      RETURN_STATUS_UNEXPECTED("Cifar 100 file not found!");
    }

    std::ifstream in(file, std::ios::binary);
    if (!in.is_open()) {
      RETURN_STATUS_UNEXPECTED(file + " can not be opened.");
    }

    for (uint32_t index = 0; index < num_cifar100_records / kCifarBlockImageNum; index++) {
      (void)in.read(reinterpret_cast<char *>(&(image_data[0])), block_size * sizeof(unsigned char));
      if (in.fail()) {
        RETURN_STATUS_UNEXPECTED("Fail to read cifar file" + file);
      }
      (void)cifar_raw_data_block_->EmplaceBack(image_data);
    }
    in.close();
  }
  (void)cifar_raw_data_block_->EmplaceBack(std::vector<unsigned char>());  // block end
  return Status::OK();
}

Status CifarOp::GetCifarFiles() {
  // Initialize queue to hold the file names
  const std::string kExtension = ".bin";
  Path dataset_directory(folder_path_);
  auto dirIt = Path::DirIterator::OpenDirectory(&dataset_directory);
  if (dirIt) {
    while (dirIt->hasNext()) {
      Path file = dirIt->next();
      std::string filename = file.toString();
      if (filename.find(kExtension) != std::string::npos) {
        cifar_files_.push_back(filename);
        MS_LOG(INFO) << "Cifar operator found file at " << filename << ".";
      }
    }
  } else {
    std::string err_msg = "Unable to open directory " + dataset_directory.toString();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::sort(cifar_files_.begin(), cifar_files_.end());
  return Status::OK();
}

Status CifarOp::ParseCifarData() {
  std::vector<unsigned char> block;
  RETURN_IF_NOT_OK(cifar_raw_data_block_->PopFront(&block));
  uint32_t cur_block_index = 0;
  while (!block.empty()) {
    for (uint32_t index = 0; index < kCifarBlockImageNum; ++index) {
      std::vector<uint32_t> labels;
      uint32_t label = block[cur_block_index++];
      labels.push_back(label);
      if (cifar_type_ == kCifar100) {
        uint32_t fine_label = block[cur_block_index++];
        labels.push_back(fine_label);
      }

      std::shared_ptr<Tensor> image_tensor;
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&image_tensor, data_schema_->column(0).tensorImpl(),
                                            TensorShape({kCifarImageHeight, kCifarImageWidth, kCifarImageChannel}),
                                            data_schema_->column(0).type()));
      for (int ch = 0; ch < kCifarImageChannel; ++ch) {
        for (int pix = 0; pix < kCifarImageHeight * kCifarImageWidth; ++pix) {
          (image_tensor->StartAddr())[pix * kCifarImageChannel + ch] = block[cur_block_index++];
        }
      }
      cifar_image_label_pairs_.emplace_back(std::make_pair(image_tensor, labels));
    }
    RETURN_IF_NOT_OK(cifar_raw_data_block_->PopFront(&block));
    cur_block_index = 0;
  }
  cifar_image_label_pairs_.shrink_to_fit();
  num_rows_ = cifar_image_label_pairs_.size();
  num_samples_ = (num_samples_ == 0 || num_samples_ > num_rows_) ? num_rows_ : num_samples_;
  if (num_rows_ == 0) {
    std::string api = cifar_type_ == kCifar10 ? "Cifar10Dataset" : "Cifar100Dataset";
    std::string err_msg = "There is no valid data matching the dataset API " + api +
                          ".Please check file path or dataset API validation first.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  cifar_raw_data_block_->Reset();
  return Status::OK();
}

// Derived from RandomAccessOp
Status CifarOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty()) {
    RETURN_STATUS_UNEXPECTED("ImageLabelPair not set");
  }

  for (uint64_t index = 0; index < cifar_image_label_pairs_.size(); ++index) {
    uint32_t label = (cifar_image_label_pairs_[index].second)[0];
    (*cls_ids)[label].push_back(index);
  }

  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

Status CifarOp::CountTotalRows(const std::string &dir, int64_t numSamples, bool isCIFAR10, int64_t *count) {
  // the logic of counting the number of samples is copied from ReadCifar100Block() and ReadCifar10Block()
  std::shared_ptr<CifarOp> op;
  *count = 0;
  RETURN_IF_NOT_OK(Builder().SetCifarDir(dir).SetNumSamples(numSamples).SetCifarType(isCIFAR10).Build(&op));
  RETURN_IF_NOT_OK(op->GetCifarFiles());
  if (op->cifar_type_ == kCifar10) {
    constexpr int64_t num_cifar10_records = 10000;
    for (auto &file : op->cifar_files_) {
      std::ifstream in(file, std::ios::binary);
      if (!in.is_open()) {
        std::string err_msg = file + " can not be opened.";
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      *count = *count + num_cifar10_records;
    }
    *count = *count < numSamples || numSamples == 0 ? *count : numSamples;
    return Status::OK();
  } else {
    int64_t num_cifar100_records = 0;
    for (auto &file : op->cifar_files_) {
      size_t pos = file.find_last_of('/');
      if (pos == std::string::npos) {
        std::string err_msg = "Invalid cifar100 file path";
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      std::string file_name(file.substr(pos + 1));
      if (file_name.find("test") != std::string::npos) {
        num_cifar100_records = 10000;
      } else if (file_name.find("train") != std::string::npos) {
        num_cifar100_records = 50000;
      }
      std::ifstream in(file, std::ios::binary);
      if (!in.is_open()) {
        std::string err_msg = file + " can not be opened.";
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
    }
    *count = num_cifar100_records < numSamples || numSamples == 0 ? num_cifar100_records : numSamples;
    return Status::OK();
  }
}
}  // namespace dataset
}  // namespace mindspore
