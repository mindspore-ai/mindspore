/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/album_op.h"
#include <fstream>
#include <iomanip>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
AlbumOp::Builder::Builder() : builder_decode_(false), builder_sampler_(nullptr), builder_schema_file_("") {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status AlbumOp::Builder::Build(std::shared_ptr<AlbumOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (builder_sampler_ == nullptr) {
    int64_t num_samples = 0;  // default num samples of 0 means to sample entire set of data
    int64_t start_index = 0;
    builder_sampler_ = std::make_shared<SequentialSampler>(start_index, num_samples);
  }

  builder_schema_ = std::make_unique<DataSchema>();
  Path schema_file(builder_schema_file_);
  if (builder_schema_file_ == "" || !schema_file.Exists()) {
    RETURN_STATUS_UNEXPECTED("Schema not provided");
  } else {
    MS_LOG(INFO) << "Schema file provided: " << builder_schema_file_ << ".";
    builder_schema_->LoadSchemaFile(builder_schema_file_, builder_columns_to_load_);
  }
  *ptr = std::make_shared<AlbumOp>(builder_num_workers_, builder_rows_per_buffer_, builder_dir_,
                                   builder_op_connector_size_, builder_decode_, builder_extensions_,
                                   std::move(builder_schema_), std::move(builder_sampler_));
  return Status::OK();
}

Status AlbumOp::Builder::SanityCheck() {
  Path dir(builder_dir_);
  std::string err_msg;
  err_msg += dir.IsDirectory() == false ? "Album path is invalid or not set\n" : "";
  err_msg += builder_num_workers_ <= 0 ? "Num of parallel workers is set to 0\n" : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
}

AlbumOp::AlbumOp(int32_t num_wkrs, int32_t rows_per_buffer, std::string file_dir, int32_t queue_size, bool do_decode,
                 const std::set<std::string> &exts, std::unique_ptr<DataSchema> data_schema,
                 std::shared_ptr<Sampler> sampler)
    : ParallelOp(num_wkrs, queue_size),
      rows_per_buffer_(rows_per_buffer),
      folder_path_(file_dir),
      decode_(do_decode),
      extensions_(exts),
      data_schema_(std::move(data_schema)),
      sampler_(std::move(sampler)),
      row_cnt_(0),
      buf_cnt_(0),
      sampler_ind_(0),
      dirname_offset_(0) {
  // Set the column name map (base class field)
  for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
    column_name_id_map_[data_schema_->column(i).name()] = i;
  }
  io_block_queues_.Init(num_workers_, queue_size);
}

// Helper function for string comparison
bool StrComp(const std::string &a, const std::string &b) {
  // returns 1 if string a is alphabetically
  // less than string b
  // quite similar to strcmp operation
  return a < b;
}

// Single thread to go through the folder directory and gets all file names
// calculate numRows then return
Status AlbumOp::PrescanEntry() {
  Path folder(folder_path_);
  dirname_offset_ = folder_path_.length();
  std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&folder);
  if (folder.Exists() == false || dirItr == nullptr) {
    RETURN_STATUS_UNEXPECTED("Error unable to open: " + folder_path_);
  }
  MS_LOG(INFO) << "Album folder Path found: " << folder_path_ << ".";

  while (dirItr->hasNext()) {
    Path file = dirItr->next();
    if (extensions_.empty() || extensions_.find(file.Extension()) != extensions_.end()) {
      (void)image_rows_.push_back(file.toString().substr(dirname_offset_));
    } else {
      MS_LOG(INFO) << "Album operator unsupported file found: " << file.toString()
                   << ", extension: " << file.Extension() << ".";
    }
  }

  std::sort(image_rows_.begin(), image_rows_.end(), StrComp);
  num_rows_ = image_rows_.size();
  return Status::OK();
}

// Main logic, Register Queue with TaskGroup, launch all threads and do the functor's work
Status AlbumOp::operator()() {
  RETURN_IF_NOT_OK(this->PrescanEntry());
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  std::unique_ptr<DataBuffer> sampler_buffer;
  RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
  while (true) {  // each iterator is 1 epoch
    std::vector<int64_t> keys;
    keys.reserve(rows_per_buffer_);
    while (sampler_buffer->eoe() == false) {
      TensorRow sample_row;
      RETURN_IF_NOT_OK(sampler_buffer->PopRow(&sample_row));
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
        if ((*itr) >= num_rows_) continue;  // index out of bound, skipping
        keys.push_back(*itr);
        row_cnt_++;
        if (row_cnt_ % rows_per_buffer_ == 0) {
          RETURN_IF_NOT_OK(
            io_block_queues_[buf_cnt_++ % num_workers_]->Add(std::make_unique<IOBlock>(keys, IOBlock::kDeIoBlockNone)));
          keys.clear();
        }
      }
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    if (keys.empty() == false) {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(keys, IOBlock::kDeIoBlockNone)));
    }
    if (IsLastIteration()) {
      std::unique_ptr<IOBlock> eoe_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe);
      std::unique_ptr<IOBlock> eof_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof);
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::move(eoe_block)));
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::move(eof_block)));
      for (int32_t i = 0; i < num_workers_; ++i) {
        RETURN_IF_NOT_OK(
          io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
      }
      return Status::OK();
    } else {  // not the last repeat. Sleep master thread, wait for the wake-up from reset
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
      RETURN_IF_NOT_OK(wp_.Wait());  // Master thread goes to sleep after it has made all the IOBlocks
      wp_.Clear();
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    UpdateRepeatAndEpochCounter();
  }
}

// contains the main logic of pulling a IOBlock from IOBlockQueue, load a buffer and push the buffer to out_connector_
// IMPORTANT: 1 IOBlock produces 1 DataBuffer
Status AlbumOp::WorkerEntry(int32_t worker_id) {
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
      if (keys.empty() == true) return Status::OK();  // empty key is a quit signal for workers
      std::unique_ptr<DataBuffer> db = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
      RETURN_IF_NOT_OK(LoadBuffer(keys, &db));
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db)));
      buffer_id += num_workers_;
    }
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("Unexpected nullptr received in worker");
}

// Only support JPEG/PNG/GIF/BMP
// Optimization: Could take in a tensor
Status AlbumOp::CheckImageType(const std::string &file_name, bool *valid) {
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

Status AlbumOp::LoadImageTensor(const std::string &image_file_path, uint32_t col_num, TensorRow *row) {
  std::shared_ptr<Tensor> image;
  std::ifstream fs;
  fs.open(image_file_path, std::ios::binary | std::ios::in);
  if (fs.fail()) {
    MS_LOG(INFO) << "Image file not found:" << image_file_path << ".";
    // If file doesn't exist, we don't flag this as error in input check, simply skip
    return Status::OK();
  }

  MS_LOG(INFO) << "Image file found: " << image_file_path << ".";

  // check that the file is an image before decoding
  bool valid = false;
  RETURN_IF_NOT_OK(CheckImageType(image_file_path, &valid));
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_file_path, &image));
  if (decode_ && valid) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Fail to decode image:" + image_file_path;
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  row->push_back(std::move(image));
  return Status::OK();
}

Status AlbumOp::LoadStringArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row) {
  std::vector<std::string> data = json_obj;

  MS_LOG(INFO) << "String array label found: " << data << ".";
  std::shared_ptr<Tensor> label;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &label));
  row->push_back(std::move(label));
  return Status::OK();
}

Status AlbumOp::LoadStringTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row) {
  std::string data = json_obj;
  // now we iterate over the elements in json

  MS_LOG(INFO) << "String label found: " << data << ".";
  std::shared_ptr<Tensor> label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar<std::string>(data, &label));
  row->push_back(std::move(label));
  return Status::OK();
}

Status AlbumOp::LoadIntArrayTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row) {
  std::shared_ptr<Tensor> label;
  // consider templating this function to handle all ints
  if (data_schema_->column(col_num).type() == DataType(DataType::DE_INT64)) {
    std::vector<int64_t> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &label));
  } else if (data_schema_->column(col_num).type() == DataType(DataType::DE_INT32)) {
    std::vector<int32_t> data;

    // Iterate over the integer list and add those values to the output shape tensor
    auto items = json_obj.items();
    using it_type = decltype(items.begin());
    (void)std::transform(items.begin(), items.end(), std::back_inserter(data), [](it_type j) { return j.value(); });

    MS_LOG(INFO) << "Int array found: " << data << ".";
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(data, &label));
  } else {
    RETURN_STATUS_UNEXPECTED("Error in Load Int Tensor");
  }
  row->push_back(std::move(label));
  return Status::OK();
}

Status AlbumOp::LoadIDTensor(const std::string &file, uint32_t col_num, TensorRow *row) {
  if (data_schema_->column(col_num).type() == DataType(DataType::DE_STRING)) {
    std::shared_ptr<Tensor> id;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<std::string>(file, &id));
    row->push_back(std::move(id));
    return Status::OK();
  }
  // hack to get the file name without extension, the 1 is to get rid of the backslash character
  int64_t image_id = std::atoi(file.substr(1, file.find(".")).c_str());
  std::shared_ptr<Tensor> id;
  RETURN_IF_NOT_OK(Tensor::CreateScalar<int64_t>(image_id, &id));
  MS_LOG(INFO) << "File ID " << image_id << ".";
  row->push_back(std::move(id));
  return Status::OK();
}

Status AlbumOp::LoadEmptyTensor(uint32_t col_num, TensorRow *row) {
  // hack to get the file name without extension, the 1 is to get rid of the backslash character
  std::shared_ptr<Tensor> empty_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({}), data_schema_->column(col_num).type(), &empty_tensor));
  row->push_back(std::move(empty_tensor));
  return Status::OK();
}

// Loads a tensor with float value, issue with float64, we don't have reverse look up to the type
// So we actually have to check what type we want to fill the tensor with.
// Float64 doesn't work with reinterpret cast here. Otherwise we limit the float in the schema to
// only be float32, seems like a weird limitation to impose
Status AlbumOp::LoadFloatTensor(const nlohmann::json &json_obj, uint32_t col_num, TensorRow *row) {
  std::shared_ptr<Tensor> float_tensor;
  if (data_schema_->column(col_num).type() == DataType(DataType::DE_FLOAT64)) {
    double data = json_obj;
    MS_LOG(INFO) << "double found: " << json_obj << ".";
    RETURN_IF_NOT_OK(Tensor::CreateScalar<double>(data, &float_tensor));
  } else if (data_schema_->column(col_num).type() == DataType(DataType::DE_FLOAT32)) {
    float data = json_obj;
    RETURN_IF_NOT_OK(Tensor::CreateScalar<float>(data, &float_tensor));
    MS_LOG(INFO) << "float found: " << json_obj << ".";
  }
  row->push_back(std::move(float_tensor));
  return Status::OK();
}

// Load 1 TensorRow (image,label) using 1 ImageColumns. 1 function call produces 1 TensorTow in a DataBuffer
// possible optimization: the helper functions of LoadTensorRow should be optimized
// to take a reference to a column descriptor?
Status AlbumOp::LoadTensorRow(const std::string &file, TensorRow *row) {
  // testing here is to just print out file path
  (*row) = {};
  MS_LOG(INFO) << "Image row file: " << file << ".";

  std::ifstream file_handle(folder_path_ + file);
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Json file " + folder_path_ + file + " can not open.");
  }
  std::string line;
  while (getline(file_handle, line)) {
    try {
      nlohmann::json js = nlohmann::json::parse(line);
      MS_LOG(INFO) << "This Line: " << line << ".";

      // note if take a schema here, then we have to iterate over all column descriptors in schema and check for key
      // get columns in schema:
      int32_t columns = data_schema_->NumColumns();

      // loop over each column descriptor, this can optimized by swtich cases
      for (int32_t i = 0; i < columns; i++) {
        // special case to handle
        if (data_schema_->column(i).name() == "id") {
          // id is internal, special case to load from file
          RETURN_IF_NOT_OK(LoadIDTensor(file, i, row));
          continue;
        }
        // find if key does not exist, insert placeholder nullptr if not found
        if (js.find(data_schema_->column(i).name()) == js.end()) {
          // iterator not found, push nullptr as placeholder
          MS_LOG(INFO) << "Pushing empty tensor for column: " << data_schema_->column(i).name() << ".";
          RETURN_IF_NOT_OK(LoadEmptyTensor(i, row));
          continue;
        }
        nlohmann::json column_value = js.at(data_schema_->column(i).name());
        MS_LOG(INFO) << "This column is: " << data_schema_->column(i).name() << ".";
        bool is_array = column_value.is_array();
        // load single string
        if (column_value.is_string() && data_schema_->column(i).type() == DataType(DataType::DE_STRING)) {
          RETURN_IF_NOT_OK(LoadStringTensor(column_value, i, row));
          continue;
        }
        // load string array
        if (is_array && data_schema_->column(i).type() == DataType(DataType::DE_STRING)) {
          RETURN_IF_NOT_OK(LoadStringArrayTensor(column_value, i, row));
          continue;
        }
        // load image file
        if (column_value.is_string() && data_schema_->column(i).type() != DataType(DataType::DE_STRING)) {
          std::string image_file_path = column_value;
          RETURN_IF_NOT_OK(LoadImageTensor(image_file_path, i, row));
          continue;
        }
        // load float array
        if (!is_array && (data_schema_->column(i).type() == DataType(DataType::DE_FLOAT32) ||
                          data_schema_->column(i).type() == DataType(DataType::DE_FLOAT64))) {
          RETURN_IF_NOT_OK(LoadFloatTensor(column_value, i, row));
          continue;
        }
        // int array
        if (is_array && (data_schema_->column(i).type() == DataType(DataType::DE_INT64) ||
                         data_schema_->column(i).type() == DataType(DataType::DE_INT32))) {
          RETURN_IF_NOT_OK(LoadIntArrayTensor(column_value, i, row));
          continue;
        } else {
          MS_LOG(WARNING) << "Value type for column: " << data_schema_->column(i).name() << " is not supported.";
          continue;
        }
      }
    } catch (const std::exception &err) {
      file_handle.close();
      RETURN_STATUS_UNEXPECTED("Parse Json file failed");
    }
  }
  file_handle.close();
  return Status::OK();
}

// Looping over LoadTensorRow to make 1 DataBuffer. 1 function call produces 1 buffer
Status AlbumOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  TensorRow trow;

  for (const int64_t &key : keys) {
    RETURN_IF_NOT_OK(this->LoadTensorRow(image_rows_[key], &trow));
    deq->push_back(std::move(trow));
  }
  (*db)->set_tensor_table(std::move(deq));
  return Status::OK();
}

void AlbumOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <AlbumOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nAlbum directory: " << folder_path_ << "\n\n";
  }
}

// Reset Sampler and wakeup Master thread (functor)
Status AlbumOp::Reset() {
  RETURN_IF_NOT_OK(sampler_->ResetSampler());
  row_cnt_ = 0;
  wp_.Set();  // wake up master thread after reset is done
  return Status::OK();
}

// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status AlbumOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

Status AlbumOp::LaunchThreadsAndInitOp() {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  // registers QueueList and individual Queues for interrupt services
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wp_.Register(tree_->AllTasks()));
  // launch main workers that load DataBuffers by reading all images
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&AlbumOp::WorkerEntry, this, std::placeholders::_1)));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->InitSampler());  // pass numRows to Sampler
  return Status::OK();
}

// Visitor accept method for NodePass
Status AlbumOp::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->RunOnNode(shared_from_base<AlbumOp>(), modified);
}

Status AlbumOp::ComputeColMap() {
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
}  // namespace dataset
}  // namespace mindspore
