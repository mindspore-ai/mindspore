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

#include "include/api/status.h"
#ifndef ENABLE_ANDROID
#include <thread>
#endif
#include <map>
#include <sstream>

namespace mindspore {
Status::Status(enum StatusCode code, int line_of_code, const char *file_name, const std::string &extra) {
  status_code_ = code;
  line_of_code_ = line_of_code;
  file_name_ = std::string(file_name);
  err_description_ = extra;
  std::ostringstream ss;
#ifndef ENABLE_ANDROID
  ss << "Thread ID " << std::this_thread::get_id() << " " << CodeAsString(code) << ". ";
  if (!extra.empty()) {
    ss << extra;
  }
  ss << "\n";
#endif

  ss << "Line of code : " << line_of_code << "\n";
  if (file_name != nullptr) {
    ss << "File         : " << file_name << "\n";
  }
  status_msg_ = ss.str();
}

std::string Status::CodeAsString(enum StatusCode c) {
  static std::map<enum StatusCode, std::string> info_map = {{kSuccess, "No error occurs."},
                                                            // Core
                                                            {kCoreFailed, "Common error code."},
                                                            // MD
                                                            {kMDOutOfMemory, "Out of memory"},
                                                            {kMDShapeMisMatch, "Shape is incorrect"},
                                                            {kMDInterrupted, "Interrupted system call"},
                                                            {kMDNoSpace, "No space left on device"},
                                                            {kMDPyFuncException, "Exception thrown from PyFunc"},
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
                                                            {kMDUnexpectedError, "Unexpected error"},
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
  auto iter = info_map.find(c);
  return iter == info_map.end() ? "Unknown error" : iter->second;
}

std::ostream &operator<<(std::ostream &os, const Status &s) {
  os << s.ToString();
  return os;
}

const std::string &Status::SetErrDescription(const std::string &err_description) {
  err_description_ = err_description;
  std::ostringstream ss;
#ifndef ENABLE_ANDROID
  ss << "Thread ID " << std::this_thread::get_id() << " " << CodeAsString(status_code_) << ". ";
  if (!err_description_.empty()) {
    ss << err_description_;
  }
  ss << "\n";
#endif

  if (line_of_code_ > 0 && !file_name_.empty()) {
    ss << "Line of code : " << line_of_code_ << "\n";
    ss << "File         : " << file_name_ << "\n";
  }
  status_msg_ = ss.str();
  return status_msg_;
}
}  // namespace mindspore
