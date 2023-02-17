/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef BUILDING_DLL
#define BUILDING_DLL
#endif

#include "include/api/status.h"
#ifndef ENABLE_ANDROID
#include <thread>
#endif
#include <map>
#include <sstream>

namespace mindspore {
struct Status::Data {
  enum StatusCode status_code = kSuccess;
  std::string status_msg;
  int line_of_code = -1;
  std::string file_name;
  std::string err_description;
};

static std::map<enum StatusCode, std::string> status_info_map = {
  {kSuccess, "No error occurs."},
  // Core
  {kCoreFailed, "Common error code."},
  // MD
  {kMDOutOfMemory, "Out of memory"},
  {kMDShapeMisMatch, "Shape is incorrect"},
  {kMDInterrupted, "Interrupted system call"},
  {kMDNoSpace, "No space left on device"},
  {kMDPyFuncException, "Exception thrown from user defined Python function in dataset"},
  {kMDDuplicateKey, "Duplicate key"},
  {kMDPythonInterpreterFailure, ""},
  {kMDTDTPushFailure, "Unexpected error"},
  {kMDFileNotExist, "Unexpected error"},
  {kMDProfilingError, "Error encountered while profiling"},
  {kMDBoundingBoxOutOfBounds, "Unexpected error"},
  {kMDBoundingBoxInvalidShape, "Unexpected error"},
  {kMDSyntaxError, "Syntax error"},
  {kMDTimeOut, "Unexpected error"},
  {kMDBuddySpaceFull, "BuddySpace full"},
  {kMDNetWorkError, "Network error"},
  {kMDNotImplementedYet, "Unexpected error"},
  {kMDUnexpectedError, "Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'"},
  // ME
  {kMEFailed, "Common error code."},
  {kMEInvalidInput, "Invalid input."},
  // MC
  {kMCFailed, "Common error code."},
  {kMCDeviceError, "Device error."},
  {kMCInvalidInput, "Invalid input."},
  {kMCInvalidArgs, "Invalid arguments."},
  // Lite
  {kLiteError, "Common error code."},
  {kLiteNullptr, "NULL pointer returned."},
  {kLiteParamInvalid, "Invalid parameter."},
  {kLiteNoChange, "No change."},
  {kLiteSuccessExit, "No error but exit."},
  {kLiteMemoryFailed, "Fail to create memory."},
  {kLiteNotSupport, "Fail to support."},
  {kLiteThreadPoolError, "Thread pool error."},
  {kLiteOutOfTensorRange, "Failed to check range."},
  {kLiteInputTensorError, "Failed to check input tensor."},
  {kLiteReentrantError, "Exist executor running."},
  {kLiteGraphFileError, "Failed to verify graph file."},
  {kLiteNotFindOp, "Failed to find operator."},
  {kLiteInvalidOpName, "Invalid operator name."},
  {kLiteInvalidOpAttr, "Invalid operator attr."},
  {kLiteOpExecuteFailure, "Failed to execution operator."},
  {kLiteFormatError, "Failed to checking tensor format."},
  {kLiteInferError, "Failed to infer shape."},
  {kLiteInferInvalid, "Invalid infer shape before runtime."},
  {kLiteInputParamInvalid, "Invalid input param by user."}};

Status::Status() : data_(std::make_shared<Data>()) {}

Status::Status(enum StatusCode status_code, const std::vector<char> &status_msg) : data_(std::make_shared<Data>()) {
  if (data_ == nullptr) {
    return;
  }

  data_->err_description = CharToString(status_msg);
  data_->status_msg = CharToString(status_msg);
  data_->status_code = status_code;
}

Status::Status(enum StatusCode code, int line_of_code, const char *file_name, const std::vector<char> &extra)
    : data_(std::make_shared<Data>()) {
  if (data_ == nullptr) {
    return;
  }
  data_->status_code = code;
  data_->line_of_code = line_of_code;
  if (file_name != nullptr) {
    data_->file_name = file_name;
  }
  data_->err_description = CharToString(extra);

  std::ostringstream ss;
#ifndef ENABLE_ANDROID
#ifdef DEBUG
  ss << "Thread ID " << std::this_thread::get_id() << " " << CodeAsString(code) << ". ";
#else
  ss << CodeAsString(code) << ". ";
#endif
  if (!data_->err_description.empty()) {
    ss << data_->err_description;
  }
  ss << "\n";
#endif

  ss << "Line of code : " << line_of_code << "\n";
  if (file_name != nullptr) {
    ss << "File         : " << file_name << "\n";
  }
  data_->status_msg = ss.str();
}

enum StatusCode Status::StatusCode() const {
  if (data_ == nullptr) {
    return kSuccess;
  }
  return data_->status_code;
}

std::vector<char> Status::ToCString() const {
  if (data_ == nullptr) {
    return std::vector<char>();
  }
  if (!data_->status_msg.empty()) {
    return StringToChar(data_->status_msg);
  }
  return CodeAsCString(data_->status_code);
}

int Status::GetLineOfCode() const {
  if (data_ == nullptr) {
    return -1;
  }
  return data_->line_of_code;
}

std::vector<char> Status::GetFileNameChar() const {
  if (data_ == nullptr) {
    return std::vector<char>();
  }
  return StringToChar(data_->file_name);
}

std::vector<char> Status::GetErrDescriptionChar() const {
  if (data_ == nullptr) {
    return std::vector<char>();
  }
  if (data_->err_description.empty()) {
    return ToCString();
  } else {
    return StringToChar(data_->err_description);
  }
}

std::vector<char> Status::CodeAsCString(enum StatusCode c) {
  auto iter = status_info_map.find(c);
  return StringToChar(iter == status_info_map.end() ? "Unknown error" : iter->second);
}

std::ostream &operator<<(std::ostream &os, const Status &s) {
  os << s.ToString();
  return os;
}

std::vector<char> Status::SetErrDescription(const std::vector<char> &err_description) {
  if (data_ == nullptr) {
    return std::vector<char>();
  }
  data_->err_description = CharToString(err_description);
  std::ostringstream ss;
#ifndef ENABLE_ANDROID
#ifdef DEBUG
  ss << "Thread ID " << std::this_thread::get_id() << " " << CodeAsString(data_->status_code) << ". ";
#else
  ss << CodeAsString(data_->status_code) << ". ";
#endif
  if (!data_->err_description.empty()) {
    ss << data_->err_description;
  }
  ss << "\n";
#endif

  if (data_->line_of_code > 0 && !data_->file_name.empty()) {
    ss << "Line of code : " << data_->line_of_code << "\n";
    ss << "File         : " << data_->file_name << "\n";
  }
  data_->status_msg = ss.str();
  return StringToChar(data_->status_msg);
}

void Status::SetStatusMsgChar(const std::vector<char> &status_msg) {
  if (data_ == nullptr) {
    return;
  }
  data_->status_msg = CharToString(status_msg);
}

bool Status::operator==(const Status &other) const {
  if (data_ == nullptr && other.data_ == nullptr) {
    return true;
  }

  if (data_ == nullptr || other.data_ == nullptr) {
    return false;
  }

  return data_->status_code == other.data_->status_code;
}

bool Status::operator==(enum StatusCode other_code) const { return StatusCode() == other_code; }
bool Status::operator!=(const Status &other) const { return !operator==(other); }
bool Status::operator!=(enum StatusCode other_code) const { return !operator==(other_code); }

Status::operator bool() const { return (StatusCode() == kSuccess); }
Status::operator int() const { return static_cast<int>(StatusCode()); }

Status Status::OK() { return StatusCode::kSuccess; }
bool Status::IsOk() const { return (StatusCode() == StatusCode::kSuccess); }
bool Status::IsError() const { return !IsOk(); }
}  // namespace mindspore
