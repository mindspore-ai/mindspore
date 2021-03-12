/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/csv_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
CsvOp::Builder::Builder()
    : builder_device_id_(0), builder_num_devices_(1), builder_num_samples_(0), builder_shuffle_files_(false) {
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  builder_num_workers_ = config_manager->num_parallel_workers();
  builder_op_connector_size_ = config_manager->op_connector_size();
  builder_rows_per_buffer_ = config_manager->rows_per_buffer();
  builder_worker_connector_size_ = config_manager->worker_connector_size();
}

Status CsvOp::Builder::ValidateInputs() const {
  std::string err;
  err += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                       std::to_string(builder_num_workers_) + ".\n"
                                   : "";
  err += (builder_device_id_ >= builder_num_devices_ || builder_num_devices_ < 1)
           ? "Invalid parameter, num_shard must be greater than shard_id and greater than 0, got num_shard: " +
               std::to_string(builder_num_devices_) + ", shard_id: " + std::to_string(builder_device_id_) + ".\n"
           : "";
  return err.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err);
}

Status CsvOp::Builder::Build(std::shared_ptr<CsvOp> *op) {
  RETURN_IF_NOT_OK(ValidateInputs());

  // Throttle the number of workers if we have more workers than files!
  if (static_cast<size_t>(builder_num_workers_) > builder_csv_files_list_.size()) {
    builder_num_workers_ = builder_csv_files_list_.size();
    MS_LOG(WARNING) << "CsvOp operator parallelism reduced to " << builder_num_workers_ << " workers.";
  }

  std::shared_ptr<CsvOp> csv_op = std::make_shared<CsvOp>(
    builder_csv_files_list_, builder_field_delim_, builder_column_default_list_, builder_column_name_list_,
    builder_num_workers_, builder_rows_per_buffer_, builder_num_samples_, builder_worker_connector_size_,
    builder_op_connector_size_, builder_shuffle_files_, builder_num_devices_, builder_device_id_);
  RETURN_IF_NOT_OK(csv_op->Init());
  *op = std::move(csv_op);

  return Status::OK();
}

CsvOp::CsvOp(const std::vector<std::string> &csv_files_list, char field_delim,
             const std::vector<std::shared_ptr<BaseRecord>> &column_default,
             const std::vector<std::string> &column_name, int32_t num_workers, int64_t rows_per_buffer,
             int64_t num_samples, int32_t worker_connector_size, int32_t op_connector_size, bool shuffle_files,
             int32_t num_device, int32_t device_id)
    : ParallelOp(num_workers, op_connector_size),
      csv_files_list_(std::move(csv_files_list)),
      field_delim_(field_delim),
      column_default_list_(column_default),
      column_name_list_(column_name),
      rows_per_buffer_(rows_per_buffer),
      num_rows_per_shard_(0),
      all_num_rows_(0),
      num_samples_(num_samples),
      filename_index_(std::make_unique<StringIndex>()),
      load_jagged_connector_(true),
      shuffle_files_(shuffle_files),
      finished_reading_dataset_(false),
      num_devices_(num_device),
      device_id_(device_id),
      load_io_block_queue_(true) {
  worker_connector_size_ = worker_connector_size;
}

Status CsvOp::Init() {
  RETURN_IF_NOT_OK(filename_index_->insert(csv_files_list_));

  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(csv_files_list_.size() / num_workers_) + 1);
  io_block_queues_.Init(num_workers_, safe_queue_size);

  RETURN_IF_NOT_OK(ParallelOp::CreateWorkerConnector(worker_connector_size_));
  jagged_buffer_connector_ = std::make_shared<JaggedConnector>(num_workers_, 1, worker_connector_size_);

  return Status::OK();
}

CsvOp::CsvParser::CsvParser(int32_t worker_id, std::shared_ptr<JaggedConnector> connector, int64_t rows_per_buffer,
                            char field_delim, std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default,
                            std::string file_path)
    : worker_id_(worker_id),
      buffer_connector_(connector),
      csv_rows_per_buffer_(rows_per_buffer),
      csv_field_delim_(field_delim),
      column_default_(column_default),
      file_path_(file_path),
      cur_state_(START_OF_FILE),
      pos_(0),
      cur_row_(0),
      cur_col_(0),
      total_rows_(0),
      start_offset_(0),
      end_offset_(std::numeric_limits<int64_t>::max()),
      err_message_("unknown") {
  cur_buffer_ = std::make_unique<DataBuffer>(0, DataBuffer::BufferFlags::kDeBFlagNone);
  InitCsvParser();
}

void CsvOp::CsvParser::Reset() {
  cur_state_ = START_OF_FILE;
  pos_ = 0;
  cur_row_ = 0;
  cur_col_ = 0;
}

CsvOp::CsvParser::Message CsvOp::CsvParser::GetMessage(int c) {
  if (c == csv_field_delim_) {
    return Message::MS_DELIM;
  } else if (c == '"') {
    return Message::MS_QUOTE;
  } else if (c == '\r' || c == '\n') {
    return Message::MS_END_OF_LINE;
  } else if (c == std::char_traits<char>::eof()) {
    return Message::MS_END_OF_FILE;
  } else {
    return Message::MS_NORMAL;
  }
}

int CsvOp::CsvParser::ProcessMessage(int c) {
  Message m = GetMessage(c);
  StateDiagram::iterator it = sd.find({cur_state_, m});
  if (it == sd.end()) {
    return -1;
  }
  int ret = it->second.second(*this, c);
  cur_state_ = it->second.first;
  return ret;
}

int CsvOp::CsvParser::PutChar(int c) {
  if (pos_ >= str_buf_.size()) {
    str_buf_.resize(str_buf_.size() * 2);
  }
  str_buf_[pos_] = c;
  pos_++;
  return 0;
}

int CsvOp::CsvParser::PutRecord(int c) {
  std::string s = std::string(str_buf_.begin(), str_buf_.begin() + pos_);
  std::shared_ptr<Tensor> t;
  if (cur_col_ >= column_default_.size()) {
    err_message_ = "Number of file columns does not match the default records";
    return -1;
  }
  switch (column_default_[cur_col_]->type) {
    case CsvOp::INT:
      Tensor::CreateScalar(std::stoi(s), &t);
      break;
    case CsvOp::FLOAT:
      Tensor::CreateScalar(std::stof(s), &t);
      break;
    default:
      Tensor::CreateScalar(s, &t);
      break;
  }
  if (cur_col_ >= (*tensor_table_)[cur_row_].size()) {
    err_message_ = "Number of file columns does not match the tensor table";
    return -1;
  }
  (*tensor_table_)[cur_row_][cur_col_] = std::move(t);
  pos_ = 0;
  cur_col_++;
  return 0;
}

int CsvOp::CsvParser::PutRow(int c) {
  if (total_rows_ < start_offset_) {
    total_rows_++;
    cur_col_ = 0;
    return 0;
  }

  if (total_rows_ >= end_offset_) {
    cur_col_ = 0;
    return 0;
  }

  int ret = PutRecord(c);
  if (ret < 0) {
    return ret;
  }

  if (cur_col_ != column_default_.size()) {
    err_message_ = "The number of columns does not match the definition.";
    return -1;
  }

  total_rows_++;
  cur_row_++;
  cur_col_ = 0;

  if (cur_row_ == csv_rows_per_buffer_) {
    cur_buffer_->set_tensor_table(std::move(tensor_table_));
    buffer_connector_->Add(worker_id_, std::move(cur_buffer_));

    cur_buffer_ = std::make_unique<DataBuffer>(0, DataBuffer::BufferFlags::kDeBFlagNone);
    tensor_table_ = std::make_unique<TensorQTable>();
    cur_row_ = 0;
  }
  return 0;
}

int CsvOp::CsvParser::AddRow(int c) {
  total_rows_++;
  return 0;
}

int CsvOp::CsvParser::EndFile(int c) {
  if (cur_col_ > 0) {
    int ret = PutRow(c);
    if (ret < 0) {
      return ret;
    }
  }

  if (cur_row_ > 0) {
    cur_buffer_->set_tensor_table(std::move(tensor_table_));
    buffer_connector_->Add(worker_id_, std::move(cur_buffer_));
  }
  return 0;
}

int CsvOp::CsvParser::CatchException(int c) {
  if (GetMessage(c) == Message::MS_QUOTE && cur_state_ == State::UNQUOTE) {
    err_message_ = "Invalid quote in unquote field.";
  } else if (GetMessage(c) == Message::MS_END_OF_FILE && cur_state_ == State::QUOTE) {
    err_message_ = "Reach the end of file in quote field.";
  } else if (GetMessage(c) == Message::MS_NORMAL && cur_state_ == State::SECOND_QUOTE) {
    err_message_ = "Receive unquote char in quote field.";
  }
  return -1;
}

int CsvOp::CsvParser::CountRows(int c) {
  Message m;
  if (c == '"') {
    m = Message::MS_QUOTE;
  } else if (c == '\r' || c == '\n' || c == std::char_traits<char>::eof()) {
    m = Message::MS_END_OF_LINE;
  } else {
    m = Message::MS_NORMAL;
  }
  StateDiagram::iterator it = sdl.find({cur_state_, m});
  if (it == sdl.end()) {
    return -1;
  }
  cur_state_ = it->second.first;
  return it->second.second(*this, c);
}

Status CsvOp::CsvParser::InitCsvParser() {
  str_buf_.resize(CSV_BUFFER_SIZE);

  // State diagram for counting rows
  sdl = {// START_OF_FILE
         // |---------------------------------------|
         // |    abc    |    "      |      \n       |
         // |---------------------------------------|
         // | UNQUOTE   |   QUOTE   | START_OF_FILE |
         // |---------------------------------------|
         // |  NullFunc |  NullFunc |    NullFunc   |
         // |---------------------------------------|
         {{State::START_OF_FILE, Message::MS_NORMAL}, {State::UNQUOTE, &CsvParser::NullFunc}},
         {{State::START_OF_FILE, Message::MS_QUOTE}, {State::QUOTE, &CsvParser::NullFunc}},
         {{State::START_OF_FILE, Message::MS_END_OF_LINE}, {State::START_OF_FILE, &CsvParser::NullFunc}},

         // UNQUOTE
         // |-------------------------------------|
         // |    abc    |    "      |    \n       |
         // |-------------------------------------|
         // | UNQUOTE   | QUOTE     | END_OF_LINE |
         // |-------------------------------------|
         // |  NullFunc | NullFunc  |  AddRow     |
         // |-------------------------------------|
         {{State::UNQUOTE, Message::MS_NORMAL}, {State::UNQUOTE, &CsvParser::NullFunc}},
         {{State::UNQUOTE, Message::MS_QUOTE}, {State::QUOTE, &CsvParser::NullFunc}},
         {{State::UNQUOTE, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::AddRow}},

         // QUOTE
         // |--------------------------------------|
         // |    abc    |      "       |    \n     |
         // |--------------------------------------|
         // | QUOTE     | SECOND_QUOTE |   QUOTE   |
         // |--------------------------------------|
         // |  NullFunc |  NullFunc    | NullFunc  |
         // |--------------------------------------|
         {{State::QUOTE, Message::MS_NORMAL}, {State::QUOTE, &CsvParser::NullFunc}},
         {{State::QUOTE, Message::MS_QUOTE}, {State::SECOND_QUOTE, &CsvParser::NullFunc}},
         {{State::QUOTE, Message::MS_END_OF_LINE}, {State::QUOTE, &CsvParser::NullFunc}},

         // SECOND_QUOTE
         // |-------------------------------------|
         // |    abc    |     "     |     \n      |
         // |-------------------------------------|
         // | UNQUOTE   | QUOTE     | END_OF_LINE |
         // |-------------------------------------|
         // |  NullFunc | NullFunc  | AddRow      |
         // |-------------------------------------|
         {{State::SECOND_QUOTE, Message::MS_NORMAL}, {State::UNQUOTE, &CsvParser::NullFunc}},
         {{State::SECOND_QUOTE, Message::MS_QUOTE}, {State::QUOTE, &CsvParser::NullFunc}},
         {{State::SECOND_QUOTE, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::AddRow}},

         // END_OF_LINE
         // |-------------------------------------|
         // |    abc    |     "     |     \n      |
         // |-------------------------------------|
         // | UNQUOTE   | QUOTE     | END_OF_LINE |
         // |-------------------------------------|
         // | NullFunc  |  NullFunc | NullFunc    |
         // |-------------------------------------|
         {{State::END_OF_LINE, Message::MS_NORMAL}, {State::UNQUOTE, &CsvParser::NullFunc}},
         {{State::END_OF_LINE, Message::MS_QUOTE}, {State::QUOTE, &CsvParser::NullFunc}},
         {{State::END_OF_LINE, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::NullFunc}}};

  // State diagram for CSV parser
  sd = {// START_OF_FILE
        // |-------------------------------------------------------------------|
        // |    abc    |    ,     |    "     |      \n        |       EOF      |
        // |-------------------------------------------------------------------|
        // | UNQUOTE   | DELIM    | QUOTE    | START_OF_FILE  | END_OF_FILE    |
        // |-------------------------------------------------------------------|
        // | lambda    | lambda   | lambda   | NullFunc       |  NullFunc      |
        // |-------------------------------------------------------------------|
        {{State::START_OF_FILE, Message::MS_NORMAL},
         {State::UNQUOTE,
          [this](CsvParser &, char c) -> int {
            TensorRow row(column_default_.size(), nullptr);
            std::vector<std::string> file_path(column_default_.size(), file_path_);
            row.setPath(file_path);
            this->tensor_table_ = std::make_unique<TensorQTable>();
            this->tensor_table_->push_back(row);
            this->str_buf_[0] = c;
            this->pos_ = 1;
            return 0;
          }}},
        {{State::START_OF_FILE, Message::MS_DELIM},
         {State::DELIM,
          [this](CsvParser &, char c) -> int {
            TensorRow row(column_default_.size(), nullptr);
            std::vector<std::string> file_path(column_default_.size(), file_path_);
            row.setPath(file_path);
            this->tensor_table_ = std::make_unique<TensorQTable>();
            this->tensor_table_->push_back(row);
            return this->PutRecord(c);
          }}},
        {{State::START_OF_FILE, Message::MS_QUOTE},
         {State::QUOTE,
          [this](CsvParser &, char c) -> int {
            TensorRow row(column_default_.size(), nullptr);
            std::vector<std::string> file_path(column_default_.size(), file_path_);
            row.setPath(file_path);
            this->tensor_table_ = std::make_unique<TensorQTable>();
            this->tensor_table_->push_back(row);
            this->pos_ = 0;
            return 0;
          }}},
        {{State::START_OF_FILE, Message::MS_END_OF_LINE}, {State::START_OF_FILE, &CsvParser::NullFunc}},
        {{State::START_OF_FILE, Message::MS_END_OF_FILE}, {State::END_OF_FILE, &CsvParser::NullFunc}},

        // UNQUOTE
        // |-------------------------------------------------------------------|
        // |    abc    |    ,       |     "     |      \n     |       EOF      |
        // |-------------------------------------------------------------------|
        // | UNQUOTE   | DELIM      | EXCEPTION | END_OF_LINE | END_OF_FILE    |
        // |-------------------------------------------------------------------|
        // | PutChar   | PutRecord  | exception | PutRow      | EndFile        |
        // |-------------------------------------------------------------------|
        {{State::UNQUOTE, Message::MS_NORMAL}, {State::UNQUOTE, &CsvParser::PutChar}},
        {{State::UNQUOTE, Message::MS_DELIM}, {State::DELIM, &CsvParser::PutRecord}},
        {{State::UNQUOTE, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::PutRow}},
        {{State::UNQUOTE, Message::MS_END_OF_FILE}, {State::END_OF_FILE, &CsvParser::EndFile}},
        // UNQUOTE-Exception
        {{State::UNQUOTE, Message::MS_QUOTE}, {State::EXCEPTION, &CsvParser::CatchException}},

        // DELIM
        // |-------------------------------------------------------------------|
        // |    abc    |    ,       |     "     |      \n     |       EOF      |
        // |-------------------------------------------------------------------|
        // | UNQUOTE   | DELIM      | QUOTE     | END_OF_LINE | END_OF_FILE    |
        // |-------------------------------------------------------------------|
        // | PutChar   | PutRecord  | lambda    | PutRow      | EndFile        |
        // |-------------------------------------------------------------------|
        {{State::DELIM, Message::MS_NORMAL}, {State::UNQUOTE, &CsvParser::PutChar}},
        {{State::DELIM, Message::MS_DELIM}, {State::DELIM, &CsvParser::PutRecord}},
        {{State::DELIM, Message::MS_QUOTE},
         {State::QUOTE,
          [this](CsvParser &, char c) -> int {
            this->pos_ = 0;
            return 0;
          }}},
        {{State::DELIM, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::PutRow}},
        {{State::DELIM, Message::MS_END_OF_FILE}, {State::END_OF_FILE, &CsvParser::EndFile}},

        // QUOTE
        // |-----------------------------------------------------------------|
        // |    abc    |    ,     |     "        |    \n    |       EOF      |
        // |-----------------------------------------------------------------|
        // | QUOTE     | QUOTE    | SECOND_QUOTE | QUOTE    | EXCEPTION      |
        // |-----------------------------------------------------------------|
        // | PutChar   | PutChar  |  NullFunc    | PutChar  | exception      |
        // |-----------------------------------------------------------------|
        {{State::QUOTE, Message::MS_NORMAL}, {State::QUOTE, &CsvParser::PutChar}},
        {{State::QUOTE, Message::MS_DELIM}, {State::QUOTE, &CsvParser::PutChar}},
        {{State::QUOTE, Message::MS_QUOTE}, {State::SECOND_QUOTE, &CsvParser::NullFunc}},
        {{State::QUOTE, Message::MS_END_OF_LINE}, {State::QUOTE, &CsvParser::PutChar}},
        // QUOTE-Exception
        {{State::QUOTE, Message::MS_END_OF_FILE}, {State::EXCEPTION, &CsvParser::CatchException}},

        // SECOND_QUOTE
        // |------------------------------------------------------------------|
        // |    abc    |     ,      |    "     |    \n       |       EOF      |
        // |------------------------------------------------------------------|
        // | EXCEPTION | DELIM      | QUOTE    | END_OF_LINE | END_OF_FILE    |
        // |------------------------------------------------------------------|
        // | exception | PutRecord | PutChar   | PutRow      | EndFile        |
        // |------------------------------------------------------------------|
        {{State::SECOND_QUOTE, Message::MS_QUOTE}, {State::QUOTE, &CsvParser::PutChar}},
        {{State::SECOND_QUOTE, Message::MS_DELIM}, {State::DELIM, &CsvParser::PutRecord}},
        {{State::SECOND_QUOTE, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::PutRow}},
        {{State::SECOND_QUOTE, Message::MS_END_OF_FILE}, {State::END_OF_FILE, &CsvParser::EndFile}},
        // SECOND_QUOTE-Exception
        {{State::SECOND_QUOTE, Message::MS_NORMAL}, {State::EXCEPTION, &CsvParser::CatchException}},

        // END_OF_LINE
        // |-------------------------------------------------------|
        // |   abc   |   ,    |   "    |      \n     |     EOF     |
        // |-------------------------------------------------------|
        // | UNQUOTE | DELIM  | QUOTE  | END_OF_LINE | END_OF_FILE |
        // |-------------------------------------------------------|
        // | lambda  | lambda | lambda |  NullFunc   | EndFile     |
        // |-------------------------------------------------------|
        {{State::END_OF_LINE, Message::MS_NORMAL},
         {State::UNQUOTE,
          [this](CsvParser &, char c) -> int {
            if (this->total_rows_ > this->start_offset_ && this->total_rows_ <= this->end_offset_) {
              TensorRow row(column_default_.size(), nullptr);
              std::vector<std::string> file_path(column_default_.size(), file_path_);
              row.setPath(file_path);
              this->tensor_table_->push_back(row);
            }
            this->str_buf_[0] = c;
            this->pos_ = 1;
            return 0;
          }}},
        {{State::END_OF_LINE, Message::MS_DELIM},
         {State::DELIM,
          [this](CsvParser &, char c) -> int {
            if (this->total_rows_ > this->start_offset_ && this->total_rows_ <= this->end_offset_) {
              TensorRow row(column_default_.size(), nullptr);
              std::vector<std::string> file_path(column_default_.size(), file_path_);
              row.setPath(file_path);
              this->tensor_table_->push_back(row);
            }
            return this->PutRecord(c);
          }}},
        {{State::END_OF_LINE, Message::MS_QUOTE},
         {State::QUOTE,
          [this](CsvParser &, char c) -> int {
            if (this->total_rows_ > this->start_offset_ && this->total_rows_ <= this->end_offset_) {
              TensorRow row(column_default_.size(), nullptr);
              std::vector<std::string> file_path(column_default_.size(), file_path_);
              row.setPath(file_path);
              this->tensor_table_->push_back(row);
            }
            return 0;
          }}},
        {{State::END_OF_LINE, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::NullFunc}},
        {{State::END_OF_LINE, Message::MS_END_OF_FILE}, {State::END_OF_FILE, &CsvParser::EndFile}}};
  return Status::OK();
}

Status CsvOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  load_jagged_connector_ = true;
  load_io_block_queue_ = true;

  RETURN_IF_NOT_OK(ParallelOp::Reset());
  NotifyToFillIOBlockQueue();
  return Status::OK();
}

Status CsvOp::LoadFile(const std::string &file, const int64_t start_offset, const int64_t end_offset,
                       const int32_t worker_id) {
  CsvParser csv_parser(worker_id, jagged_buffer_connector_, rows_per_buffer_, field_delim_, column_default_list_, file);
  csv_parser.SetStartOffset(start_offset);
  csv_parser.SetEndOffset(end_offset);
  std::ifstream ifs;
  ifs.open(file, std::ifstream::in);
  if (!ifs.is_open()) {
    RETURN_STATUS_UNEXPECTED("Error opening file: " + file);
  }
  if (column_name_list_.empty()) {
    std::string tmp;
    getline(ifs, tmp);
  }
  csv_parser.Reset();
  try {
    while (ifs.good()) {
      // when ifstream reaches the end of file, the function get() return std::char_traits<char>::eof()
      // which is a 32-bit -1, it's not equal to the 8-bit -1 on Euler OS. So instead of char, we use
      // int to receive its return value.
      int chr = ifs.get();
      if (csv_parser.ProcessMessage(chr) != 0) {
        RETURN_STATUS_UNEXPECTED("Invalid file, failed to parse file: " + file + ":" +
                                 std::to_string(csv_parser.GetTotalRows() + 1) +
                                 ". Error message: " + csv_parser.GetErrorMessage());
      }
    }
  } catch (std::invalid_argument &ia) {
    std::string err_row = std::to_string(csv_parser.GetTotalRows() + 1);
    RETURN_STATUS_UNEXPECTED("Invalid data, " + file + ":" + err_row + ", type does not match.");
  } catch (std::out_of_range &oor) {
    std::string err_row = std::to_string(csv_parser.GetTotalRows() + 1);
    RETURN_STATUS_UNEXPECTED("Invalid data, " + file + ":" + err_row + ", out of range.");
  }
  return Status::OK();
}

Status CsvOp::operator()() {
  RETURN_IF_NOT_OK(CalculateNumRowsPerShard());

  // Move register to the front of launching thread, this will fix the problem
  // when thread exit unnormally register will failed occasionally.
  RETURN_IF_NOT_OK(io_block_queue_wait_post_.Register(tree_->AllTasks()));

  // launch one thread, responsible for filling IoBlockQueue
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(1, std::bind(&CsvOp::WaitToFillIOBlockQueue, this), "", id()));

  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&CsvOp::WorkerEntry, this, std::placeholders::_1), "", id()));

  // must be called after launching workers.
  TaskManager::FindMe()->Post();
  NotifyToFillIOBlockQueue();

  while (!finished_reading_dataset_) {
    int64_t buffer_id = 0;
    int32_t workers_done = 0;
    int64_t rows_read = 0;
    load_io_block_queue_ = true;

    while (workers_done < num_workers_) {
      std::unique_ptr<DataBuffer> buffer;
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Pop(0, &buffer));
      if (buffer->eoe()) {
        workers_done++;
      } else if (num_samples_ == 0 || rows_read < num_samples_) {
        if ((num_samples_ > 0) && (rows_read + buffer->NumRows() > num_samples_)) {
          int64_t rowsToRemove = buffer->NumRows() - (num_samples_ - rows_read);
          RETURN_IF_NOT_OK(buffer->SliceOff(rowsToRemove));
        }
        rows_read += buffer->NumRows();
        buffer->set_id(buffer_id++);
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(buffer)));
      } else {
        // end of epoch
        load_jagged_connector_ = false;
        load_io_block_queue_ = false;
      }
    }

    std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoe_buffer)));

    if (IsLastIteration()) {
      finished_reading_dataset_ = true;
      NotifyToFillIOBlockQueue();
    } else {
      jagged_buffer_connector_->DoReset();
      buffer_id = 0;
      // Self-reset to start a new iteration
      RETURN_IF_NOT_OK(Reset());
    }
    UpdateRepeatAndEpochCounter();
  }
  std::unique_ptr<DataBuffer> eof_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eof_buffer)));

  RETURN_IF_NOT_OK(PostEndOfData());
  return Status::OK();
}

Status CsvOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::unique_ptr<FilenameBlock> io_block;
  RETURN_IF_NOT_OK(PopIoBlockQueue(worker_id, &io_block));
  while (!io_block->eof()) {
    if (!io_block->eoe()) {
      if (load_jagged_connector_) {
        std::string filename;
        RETURN_IF_NOT_OK(io_block->GetFilename(&filename, *filename_index_));
        int64_t start_offset = io_block->GetStartOffset();
        int64_t end_offset = io_block->GetEndOffset();
        RETURN_IF_NOT_OK(LoadFile(filename, start_offset, end_offset, worker_id));
      }
    } else {
      std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
      RETURN_IF_NOT_OK(jagged_buffer_connector_->Add(worker_id, std::move(eoe_buffer)));
    }

    RETURN_IF_NOT_OK(PopIoBlockQueue(worker_id, &io_block));
  }
  return Status::OK();
}

// A print method typically used for debugging
void CsvOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nRows per buffer: " << rows_per_buffer_ << "\nSample count: " << num_samples_
        << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nCsv files list:\n";
    for (int i = 0; i < csv_files_list_.size(); ++i) {
      out << " " << csv_files_list_[i];
    }
    out << "\n\n";
  }
}

// Pops an element from a queue in io_block_queues
Status CsvOp::PopIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> *out_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->PopFront(out_block));

  return Status::OK();
}

// Pushes an element to a queue in io_block_queues
Status CsvOp::PushIoBlockQueue(int32_t index, std::unique_ptr<FilenameBlock> &&io_block) {
  RETURN_IF_NOT_OK(io_block_queues_[index]->Add(std::move(io_block)));

  return Status::OK();
}

static void ShuffleKeys(std::vector<int64_t> *i_keys, uint32_t seed) {
  std::mt19937 rng(seed);
  std::shuffle(i_keys->begin(), i_keys->end(), rng);
}

Status CsvOp::WaitToFillIOBlockQueue() {
  // must be called first if called by worker spanwed by taskgroup
  TaskManager::FindMe()->Post();

  std::vector<int64_t> i_keys;
  if (shuffle_files_) {
    for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
      i_keys.push_back(it.key());
    }
  }
  uint32_t seed = 0;
  while (true) {
    RETURN_IF_NOT_OK(io_block_queue_wait_post_.Wait());
    io_block_queue_wait_post_.Clear();

    if (finished_reading_dataset_) {
      break;
    }

    if (shuffle_files_) {
      ShuffleKeys(&i_keys, num_devices_ == 1 ? GetSeed() : ++seed);
    }
    RETURN_IF_NOT_OK(FillIOBlockQueue(i_keys));
  }
  return Status::OK();
}

Status CsvOp::FillIOBlockQueue(const std::vector<int64_t> &i_keys) {
  int32_t queue_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  while (!finish) {
    std::vector<std::pair<std::string, int64_t>> file_index;
    if (!i_keys.empty()) {
      for (auto it = i_keys.begin(); it != i_keys.end(); ++it) {
        {
          if (!load_io_block_queue_) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        {
          if (!load_io_block_queue_) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>(it.value(), it.key()));
      }
    }
    for (auto file_info : file_index) {
      if (NeedPushFileToBlockQueue(file_info.first, &start_offset, &end_offset, pre_count)) {
        auto ioBlock =
          std::make_unique<FilenameBlock>(file_info.second, start_offset, end_offset, IOBlock::kDeIoBlockNone);
        RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
        queue_index = (queue_index + 1) % num_workers_;
      }

      pre_count += filename_numrows_[file_info.first];
    }

    if (pre_count < (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_) {
      finish = false;
    } else {
      finish = true;
    }
  }

  RETURN_IF_NOT_OK(PostEndOfEpoch(queue_index));
  return Status::OK();
}

void CsvOp::NotifyToFillIOBlockQueue() { io_block_queue_wait_post_.Set(); }

bool CsvOp::NeedPushFileToBlockQueue(const std::string &file_name, int64_t *start_offset, int64_t *end_offset,
                                     const int64_t &pre_count) {
  *start_offset = 0;
  *end_offset = 0;
  bool push = false;
  int64_t start_index = device_id_ * num_rows_per_shard_;
  if (device_id_ + 1 < 0) {
    MS_LOG(ERROR) << "Device id is invalid";
    return false;
  }

  int64_t end_index = (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_;
  if (pre_count <= start_index && pre_count + filename_numrows_[file_name] > start_index) {
    *start_offset = start_index - pre_count;
    push = true;
    if (pre_count < end_index && pre_count + filename_numrows_[file_name] >= end_index) {
      *end_offset = end_index - pre_count;
    } else {
      *end_offset = filename_numrows_[file_name];
    }
  }

  if (pre_count >= start_index && pre_count < end_index) {
    *start_offset = 0;
    push = true;
    if (pre_count + filename_numrows_[file_name] >= end_index) {
      *end_offset = end_index - pre_count;
    } else {
      *end_offset = filename_numrows_[file_name];
    }
  }

  return push;
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume. When the worker
// pops this control indicator, it will wait until the next epoch starts and then resume execution.
Status CsvOp::PostEndOfEpoch(int32_t queue_index) {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eoe = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEoe);
    RETURN_IF_NOT_OK(PushIoBlockQueue((queue_index + i) % num_workers_, std::move(eoe)));
  }

  return Status::OK();
}

Status CsvOp::CalculateNumRowsPerShard() {
  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    int64_t count = CountTotalRows(it.value());
    filename_numrows_[it.value()] = count;
    all_num_rows_ += count;
  }
  if (all_num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API CsvDataset. Please check file path or CSV format.");
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(all_num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

int64_t CsvOp::CountTotalRows(const std::string &file) {
  CsvParser csv_parser(0, jagged_buffer_connector_, rows_per_buffer_, field_delim_, column_default_list_, file);
  std::ifstream ifs;
  ifs.open(file, std::ifstream::in);
  if (!ifs.is_open()) {
    return 0;
  }
  if (column_name_list_.empty()) {
    std::string tmp;
    getline(ifs, tmp);
  }
  csv_parser.Reset();
  while (ifs.good()) {
    int chr = ifs.get();
    if (csv_parser.CountRows(chr) != 0) {
      break;
    }
  }

  return csv_parser.GetTotalRows();
}

// Pushes a control indicator onto the IOBlockQueue for each worker to consume.
// When the worker pops this control indicator, it will shut itself down gracefully.
Status CsvOp::PostEndOfData() {
  for (int i = 0; i < num_workers_; ++i) {
    std::unique_ptr<FilenameBlock> eof = std::make_unique<FilenameBlock>(IOBlock::kDeIoBlockFlagEof);
    RETURN_IF_NOT_OK(PushIoBlockQueue(i, std::move(eof)));
  }

  return Status::OK();
}

Status CsvOp::CountAllFileRows(const std::vector<std::string> &files, bool csv_header, int64_t *count) {
  std::shared_ptr<CsvOp> op;
  *count = 0;
  if (csv_header) {
    RETURN_IF_NOT_OK(Builder().SetCsvFilesList(files).Build(&op));
  } else {
    RETURN_IF_NOT_OK(Builder().SetCsvFilesList(files).SetColumName({""}).Build(&op));
  }
  for (auto file : files) {
    *count += op->CountTotalRows(file);
  }
  return Status::OK();
}

std::vector<std::string> CsvOp::split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

Status CsvOp::ComputeColMap() {
  // Set the column name mapping (base class field)
  if (column_name_id_map_.empty()) {
    if (!ColumnNameValidate()) {
      RETURN_STATUS_UNEXPECTED("Fail to validate column name for input CSV file list");
    }

    for (auto &csv_file : csv_files_list_) {
      Status rc = ColMapAnalyse(csv_file);

      /* Process exception if ERROR in column name solving*/
      if (!rc.IsOk()) {
        MS_LOG(ERROR) << "Fail to analyse column name map, invalid file: " + csv_file;
        RETURN_STATUS_UNEXPECTED("Fail to analyse column name map, invalid file: " + csv_file);
      }
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }

  if (column_default_list_.size() < column_name_id_map_.size()) {
    for (int32_t i = column_default_list_.size(); i < column_name_id_map_.size(); i++) {
      column_default_list_.push_back(std::make_shared<CsvOp::Record<std::string>>(CsvOp::STRING, ""));
    }
  }

  if (column_default_list_.size() != column_name_id_map_.size()) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid parameter, the number of column names does not match the column defaults, column_default_list: " +
      std::to_string(column_default_list_.size()) +
      ", column_name_id_map: " + std::to_string(column_name_id_map_.size()));
  }

  return Status::OK();
}

Status CsvOp::ColMapAnalyse(const std::string &csv_file_name) {
  if (column_name_list_.empty()) {
    // Actually we only deal with the first file, because the column name set in other files must remain the same
    if (!check_flag_) {
      std::string line;
      std::ifstream handle(csv_file_name);

      getline(handle, line);
      std::vector<std::string> col_names = split(line, field_delim_);

      for (int32_t i = 0; i < col_names.size(); i++) {
        // consider the case of CRLF on windows
        col_names[i].erase(col_names[i].find_last_not_of('\r') + 1);

        if (column_name_id_map_.find(col_names[i]) == column_name_id_map_.end()) {
          column_name_id_map_[col_names[i]] = i;
        } else {
          MS_LOG(ERROR) << "Invalid parameter, duplicate column names are not allowed: " + col_names[i] +
                             ", The corresponding data files: " + csv_file_name;

          RETURN_STATUS_UNEXPECTED("Invalid parameter, duplicate column names are not allowed: " + col_names[i] +
                                   ", The corresponding data files: " + csv_file_name);
        }
      }
      check_flag_ = true;
    }
  } else {
    if (!check_flag_) {  // Case the first CSV file, validate the column names
      for (int32_t i = 0; i < column_name_list_.size(); ++i) {
        if (column_name_id_map_.find(column_name_list_[i]) == column_name_id_map_.end()) {
          column_name_id_map_[column_name_list_[i]] = i;
        } else {
          MS_LOG(ERROR) << "Invalid parameter, duplicate column names are not allowed: " + column_name_list_[i] +
                             ", The corresponding data files: " + csv_file_name;

          RETURN_STATUS_UNEXPECTED("Invalid parameter, duplicate column names are not allowed: " +
                                   column_name_list_[i] + ", The corresponding data files: " + csv_file_name);
        }
      }
      check_flag_ = true;
    }
  }
  return Status::OK();
}

bool CsvOp::ColumnNameValidate() {
  /* Case 1: Users specify the column_names */
  if (!column_name_list_.empty()) {
    return true;
  }

  /* Case 2: Inferring the column_names from the first row of CSV files
  \\ record: the column name set in first CSV file.
  \\ match_file: First file same */
  std::vector<std::string> record;
  std::string match_file;

  for (auto &csv_file : csv_files_list_) {
    std::string line;
    std::ifstream handle(csv_file);

    // Parse the csv_file into column name set
    getline(handle, line);
    std::vector<std::string> col_names = split(line, field_delim_);

    /* Analyse the column name and draw a conclusion*/
    if (record.empty()) {  // Case the first file
      record = col_names;
      match_file = csv_file;
    } else {  // Case the other files
      if (col_names != record) {
        MS_LOG(ERROR)
          << "Every corresponding column name must be identical, either element or permutation. Invalid files are: " +
               match_file + " and " + csv_file;
        return false;
      }
    }
  }
  return true;
}

}  // namespace dataset
}  // namespace mindspore
