/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "utils/file_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {

CsvOp::CsvOp(const std::vector<std::string> &csv_files_list, char field_delim,
             const std::vector<std::shared_ptr<BaseRecord>> &column_default,
             const std::vector<std::string> &column_name, int32_t num_workers, int64_t num_samples,
             int32_t worker_connector_size, int32_t op_connector_size, bool shuffle_files, int32_t num_devices,
             int32_t device_id)
    : NonMappableLeafOp(std::min(num_workers, static_cast<int32_t>(csv_files_list.size())), worker_connector_size,
                        num_samples, op_connector_size, shuffle_files, num_devices, device_id),
      csv_files_list_(std::move(csv_files_list)),
      field_delim_(field_delim),
      column_default_list_(column_default),
      column_name_list_(column_name) {}

Status CsvOp::Init() {
  RETURN_IF_NOT_OK(filename_index_->insert(csv_files_list_));

  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(csv_files_list_.size() / num_workers_) + 1);
  io_block_queues_.Init(num_workers_, safe_queue_size);

  jagged_rows_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);

  return Status::OK();
}

CsvOp::CsvParser::CsvParser(int32_t worker_id, JaggedConnector *connector, char field_delim,
                            std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default, std::string file_path)
    : worker_id_(worker_id),
      rows_connector_(connector),
      csv_field_delim_(field_delim),
      column_default_(std::move(column_default)),
      cur_state_(START_OF_FILE),
      pos_(0),
      cur_col_(0),
      total_rows_(0),
      start_offset_(0),
      end_offset_(std::numeric_limits<int64_t>::max()),
      err_message_("unknown"),
      file_path_(std::move(file_path)) {}

void CsvOp::CsvParser::Reset() {
  cur_state_ = START_OF_FILE;
  pos_ = 0;
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
    std::stringstream ss;
    ss << "Invalid columns, the size of column_names should be less than the size of 'column_defaults', "
       << "but got the size of column_names: " << cur_col_
       << ", the size of column_defaults : " << column_default_.size() << ".";
    err_message_ = ss.str();
    return -1;
  }
  Status rc;
  switch (column_default_[cur_col_]->type) {
    case CsvOp::INT:
      rc = Tensor::CreateScalar(std::stoi(s), &t);
      if (rc.IsError()) {
        err_message_ = rc.ToString();
        return -1;
      }
      break;
    case CsvOp::FLOAT:
      rc = Tensor::CreateScalar(std::stof(s), &t);
      if (rc.IsError()) {
        err_message_ = rc.ToString();
        return -1;
      }
      break;
    default:
      rc = Tensor::CreateScalar(s, &t);
      if (rc.IsError()) {
        err_message_ = rc.ToString();
        return -1;
      }
      break;
  }
  if (cur_col_ >= cur_row_.size()) {
    std::stringstream ss;
    ss << "Invalid columns, the size of column_names should be greater than or equal to the size of columns of "
       << "loading data, but got the size of column_names: " << cur_col_
       << ", the size of columns in original loaded dataset: " << column_default_.size() << ".";
    err_message_ = ss.str();
    return -1;
  }
  cur_row_[cur_col_] = std::move(t);
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
    std::stringstream ss;
    ss << "Invalid columns, the size of column_names should be less than the size of 'column_defaults', "
       << "but got the size of column_names: " << cur_col_
       << ", the size of 'column_defaults': " << column_default_.size() << ".";
    err_message_ = ss.str();
    return -1;
  }

  total_rows_++;
  cur_col_ = 0;

  Status s = rows_connector_->Add(worker_id_, std::move(cur_row_));
  if (s.IsError()) {
    err_message_ = s.ToString();
    // if error type is interrupted, return error code -2
    if (s.StatusCode() == kMDInterrupted) {
      constexpr int error_code = -2;
      return error_code;
    }
    return -1;
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
  return 0;
}

int CsvOp::CsvParser::CatchException(int c) {
  if (GetMessage(c) == Message::MS_QUOTE && cur_state_ == State::UNQUOTE) {
    err_message_ = "Invalid csv file, unexpected quote in unquote field from " + file_path_ + ".";
  } else if (GetMessage(c) == Message::MS_END_OF_FILE && cur_state_ == State::QUOTE) {
    err_message_ = "Invalid csv file, reach the end of file in quote field, check " + file_path_ + ".";
  } else if (GetMessage(c) == Message::MS_NORMAL && cur_state_ == State::SECOND_QUOTE) {
    err_message_ = "Invalid csv file, receive unquote char in quote field, check " + file_path_ + ".";
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
  InitSDL();
  InitSD();
  return Status::OK();
}

void CsvOp::CsvParser::InitSDL() {
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
}

void CsvOp::CsvParser::InitSD() {
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
            this->cur_row_ = std::move(row);
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
            this->cur_row_ = std::move(row);
            return this->PutRecord(c);
          }}},
        {{State::START_OF_FILE, Message::MS_QUOTE},
         {State::QUOTE,
          [this](CsvParser &, char c) -> int {
            TensorRow row(column_default_.size(), nullptr);
            std::vector<std::string> file_path(column_default_.size(), file_path_);
            row.setPath(file_path);
            this->cur_row_ = std::move(row);
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
              this->cur_row_ = std::move(row);
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
              this->cur_row_ = std::move(row);
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
              this->cur_row_ = std::move(row);
            }
            return 0;
          }}},
        {{State::END_OF_LINE, Message::MS_END_OF_LINE}, {State::END_OF_LINE, &CsvParser::NullFunc}},
        {{State::END_OF_LINE, Message::MS_END_OF_FILE}, {State::END_OF_FILE, &CsvParser::EndFile}}};
}

Status CsvOp::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  CsvParser csv_parser(worker_id, jagged_rows_connector_.get(), field_delim_, column_default_list_, file);
  RETURN_IF_NOT_OK(csv_parser.InitCsvParser());
  csv_parser.SetStartOffset(start_offset);
  csv_parser.SetEndOffset(end_offset);

  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << file << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + file + " does not exist.");
  }

  std::ifstream ifs;
  ifs.open(realpath.value(), std::ifstream::in);
  if (!ifs.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + file + ", the file is damaged or permission denied.");
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
      int err = csv_parser.ProcessMessage(chr);
      if (err != 0) {
        // if error code is -2, the returned error is interrupted
        if (err == -2) {
          return Status(kMDInterrupted);
        }
        RETURN_STATUS_UNEXPECTED("Invalid file, failed to parse csv file: " + file + " at line " +
                                 std::to_string(csv_parser.GetTotalRows() + 1) +
                                 ". Error message: " + csv_parser.GetErrorMessage());
      }
    }
  } catch (std::invalid_argument &ia) {
    std::string err_row = std::to_string(csv_parser.GetTotalRows() + 1);
    RETURN_STATUS_UNEXPECTED("Invalid csv, csv file: " + file + " parse failed at line " + err_row +
                             ", type does not match.");
  } catch (std::out_of_range &oor) {
    std::string err_row = std::to_string(csv_parser.GetTotalRows() + 1);
    RETURN_STATUS_UNEXPECTED("Invalid csv, " + file + " parse failed at line " + err_row + " : value out of range.");
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
    out << "\nSample count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\n"
        << DatasetName(true) << " files list:\n";
    for (int i = 0; i < csv_files_list_.size(); ++i) {
      out << " " << csv_files_list_[i];
    }
    out << "\n\n";
  }
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
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
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

Status CsvOp::CalculateNumRowsPerShard() {
  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    int64_t count = CountTotalRows(it.value());
    filename_numrows_[it.value()] = count;
    num_rows_ += count;
  }
  if (num_rows_ == 0) {
    std::stringstream ss;
    for (int i = 0; i < csv_files_list_.size(); ++i) {
      ss << " " << csv_files_list_[i];
    }
    std::string file_list = ss.str();
    RETURN_STATUS_UNEXPECTED("Invalid data, " + DatasetName(true) +
                             "Dataset API can't read the data file (interface mismatch or no data found). Check " +
                             DatasetName() + " file path: " + file_list + ".");
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

int64_t CsvOp::CountTotalRows(const std::string &file) {
  CsvParser csv_parser(0, jagged_rows_connector_.get(), field_delim_, column_default_list_, file);
  Status rc = csv_parser.InitCsvParser();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "[Internal ERROR], failed to initialize " + DatasetName(true) + " Parser. Error description:"
                  << rc;
    return 0;
  }

  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, csv file: " << file << " does not exist.";
    return 0;
  }

  std::ifstream ifs;
  ifs.open(realpath.value(), std::ifstream::in);
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

Status CsvOp::CountAllFileRows(const std::vector<std::string> &files, bool csv_header, int64_t *count) {
  int32_t num_workers = GlobalContext::config_manager()->num_parallel_workers();
  int32_t op_connector_size = GlobalContext::config_manager()->op_connector_size();
  int32_t worker_connector_size = GlobalContext::config_manager()->worker_connector_size();
  const int32_t device_id = 0;
  const int32_t num_devices = 1;
  const int64_t num_samples = 0;
  bool shuffle_files = false;
  std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_list;
  std::vector<std::string> column_name_list;
  char field_delim = ',';
  std::shared_ptr<CsvOp> op;
  *count = 0;
  if (!csv_header) {
    (void)column_name_list.emplace_back("");
  }
  op = std::make_shared<CsvOp>(files, field_delim, column_list, column_name_list, num_workers, num_samples,
                               worker_connector_size, op_connector_size, shuffle_files, num_devices, device_id);
  RETURN_IF_NOT_OK(op->Init());
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
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to obtain column name from input " + DatasetName() +
                               " file list.");
    }

    for (auto &csv_file : csv_files_list_) {
      Status rc = ColMapAnalyse(csv_file);

      /* Process exception if ERROR in column name solving */
      if (!rc.IsOk()) {
        MS_LOG(ERROR) << "Invalid file, failed to get column name list from csv file: " + csv_file;
        RETURN_STATUS_UNEXPECTED("Invalid file, failed to get column name list from csv file: " + csv_file);
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
      "Invalid parameter, the size of column_names should be equal to the size of 'column_defaults', but got "
      " size of 'column_defaults': " +
      std::to_string(column_default_list_.size()) +
      ", size of column_names: " + std::to_string(column_name_id_map_.size()));
  }

  return Status::OK();
}

Status CsvOp::ColMapAnalyse(const std::string &csv_file_name) {
  if (column_name_list_.empty()) {
    // Actually we only deal with the first file, because the column name set in other files must remain the same
    if (!check_flag_) {
      auto realpath = FileUtils::GetRealPath(csv_file_name.c_str());
      if (!realpath.has_value()) {
        std::string err_msg = "Invalid file path, csv file: " + csv_file_name + " does not exist.";
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_UNEXPECTED(err_msg);
      }

      std::string line;
      std::ifstream handle(realpath.value());

      getline(handle, line);
      std::vector<std::string> col_names = split(line, field_delim_);

      for (int32_t i = 0; i < col_names.size(); i++) {
        // consider the case of CRLF on windows
        col_names[i].erase(col_names[i].find_last_not_of('\r') + 1);

        if (column_name_id_map_.find(col_names[i]) == column_name_id_map_.end()) {
          column_name_id_map_[col_names[i]] = i;
        } else {
          MS_LOG(ERROR) << "Invalid parameter, duplicate column " << col_names[i] << " for csv file: " << csv_file_name;
          RETURN_STATUS_UNEXPECTED("Invalid parameter, duplicate column " + col_names[i] +
                                   " for csv file: " + csv_file_name);
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
          MS_LOG(ERROR) << "Invalid parameter, duplicate column " << column_name_list_[i]
                        << " for csv file: " << csv_file_name << ".";
          RETURN_STATUS_UNEXPECTED("Invalid parameter, duplicate column " + column_name_list_[i] +
                                   " for csv file: " + csv_file_name + ".");
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
    auto realpath = FileUtils::GetRealPath(csv_file.c_str());
    if (!realpath.has_value()) {
      MS_LOG(ERROR) << "Invalid file path, csv file: " << csv_file << " does not exist.";
      return false;
    }

    std::string line;
    std::ifstream handle(realpath.value());

    // Parse the csv_file into column name set
    getline(handle, line);
    std::vector<std::string> col_names = split(line, field_delim_);

    /* Analyse the column name and draw a conclusion */
    if (record.empty()) {  // Case the first file
      record = col_names;
      match_file = csv_file;
    } else {  // Case the other files
      if (col_names != record) {
        MS_LOG(ERROR) << "Invalid parameter, every column name should be equal the record from csv, but got column: "
                      << col_names << ", csv record: " << record << ". Check " + match_file + " and " + csv_file + ".";
        return false;
      }
    }
  }
  return true;
}

}  // namespace dataset
}  // namespace mindspore
