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
#ifndef MINDSPORE_INCLUDE_API_STATUS_H
#define MINDSPORE_INCLUDE_API_STATUS_H

#include <string>
#include <ostream>
#include <climits>

namespace mindspore {
enum CompCode : uint32_t {
  kCore = 0x00000000u,
  kMD = 0x10000000u,
  kME = 0x20000000u,
  kMC = 0x30000000u,
  kLite = 0xF0000000u,
};

enum StatusCode : uint32_t {
  kSuccess = 0,
  // Core
  kCoreFailed = kCore | 0x1,

  // MD
  kMDOutOfMemory = kMD | 1,
  kMDShapeMisMatch = kMD | 2,
  kMDInterrupted = kMD | 3,
  kMDNoSpace = kMD | 4,
  kMDPyFuncException = kMD | 5,
  kMDDuplicateKey = kMD | 6,
  kMDPythonInterpreterFailure = kMD | 7,
  kMDTDTPushFailure = kMD | 8,
  kMDFileNotExist = kMD | 9,
  kMDProfilingError = kMD | 10,
  kMDBoundingBoxOutOfBounds = kMD | 11,
  kMDBoundingBoxInvalidShape = kMD | 12,
  kMDSyntaxError = kMD | 13,
  kMDTimeOut = kMD | 14,
  kMDBuddySpaceFull = kMD | 15,
  kMDNetWorkError = kMD | 16,
  kMDNotImplementedYet = kMD | 17,
  // Make this error code the last one. Add new error code above it.
  kMDUnexpectedError = kMD | 127,

  // ME
  kMEFailed = kME | 0x1,
  kMEInvalidInput = kME | 0x2,

  // MC
  kMCFailed = kMC | 0x1,
  kMCDeviceError = kMC | 0x2,
  kMCInvalidInput = kMC | 0x3,
  kMCInvalidArgs = kMC | 0x4,

  // Lite  // Common error code, range: [-1, -100）
  kLiteError = kLite | (0x0FFFFFFF & -1),           /**< Common error code. */
  kLiteNullptr = kLite | (0x0FFFFFFF & -2),         /**< NULL pointer returned.*/
  kLiteParamInvalid = kLite | (0x0FFFFFFF & -3),    /**< Invalid parameter.*/
  kLiteNoChange = kLite | (0x0FFFFFFF & -4),        /**< No change. */
  kLiteSuccessExit = kLite | (0x0FFFFFFF & -5),     /**< No error but exit. */
  kLiteMemoryFailed = kLite | (0x0FFFFFFF & -6),    /**< Fail to create memory. */
  kLiteNotSupport = kLite | (0x0FFFFFFF & -7),      /**< Fail to support. */
  kLiteThreadPoolError = kLite | (0x0FFFFFFF & -8), /**< Error occur in thread pool. */

  // Executor error code, range: [-100,-200)
  kLiteOutOfTensorRange = kLite | (0x0FFFFFFF & -100), /**< Failed to check range. */
  kLiteInputTensorError = kLite | (0x0FFFFFFF & -101), /**< Failed to check input tensor. */
  kLiteReentrantError = kLite | (0x0FFFFFFF & -102),   /**< Exist executor running. */

  // Graph error code, range: [-200,-300)
  kLiteGraphFileError = kLite | (0x0FFFFFFF & -200), /**< Failed to verify graph file. */

  // Node error code, range: [-300,-400)
  kLiteNotFindOp = kLite | (0x0FFFFFFF & -300),        /**< Failed to find operator. */
  kLiteInvalidOpName = kLite | (0x0FFFFFFF & -301),    /**< Invalid operator name. */
  kLiteInvalidOpAttr = kLite | (0x0FFFFFFF & -302),    /**< Invalid operator attr. */
  kLiteOpExecuteFailure = kLite | (0x0FFFFFFF & -303), /**< Failed to execution operator. */

  // Tensor error code, range: [-400,-500)
  kLiteFormatError = kLite | (0x0FFFFFFF & -400), /**< Failed to checking tensor format. */

  // InferShape error code, range: [-500,-600)
  kLiteInferError = kLite | (0x0FFFFFFF & -500),   /**< Failed to infer shape. */
  kLiteInferInvalid = kLite | (0x0FFFFFFF & -501), /**< Invalid infer shape before runtime. */

  // User input param error code, range: [-600, 700)
  kLiteInputParamInvalid = kLite | (0x0FFFFFFF & -600), /**< Invalid input param by user. */
};

class Status {
 public:
  Status() : status_code_(kSuccess), line_of_code_(-1) {}
  Status(enum StatusCode status_code, const std::string &status_msg = "")  // NOLINT(runtime/explicit)
      : status_code_(status_code), status_msg_(status_msg), line_of_code_(-1) {}
  Status(const StatusCode code, int line_of_code, const char *file_name, const std::string &extra = "");

  ~Status() = default;

  enum StatusCode StatusCode() const { return status_code_; }
  const std::string &ToString() const { return status_msg_; }

  int GetLineOfCode() const { return line_of_code_; }
  const std::string &GetErrDescription() const { return status_msg_; }
  const std::string &SetErrDescription(const std::string &err_description);

  friend std::ostream &operator<<(std::ostream &os, const Status &s);

  bool operator==(const Status &other) const { return status_code_ == other.status_code_; }
  bool operator==(enum StatusCode other_code) const { return status_code_ == other_code; }
  bool operator!=(const Status &other) const { return status_code_ != other.status_code_; }
  bool operator!=(enum StatusCode other_code) const { return status_code_ != other_code; }

  explicit operator bool() const { return (status_code_ == kSuccess); }
  explicit operator int() const { return static_cast<int>(status_code_); }

  static Status OK() { return Status(StatusCode::kSuccess); }

  bool IsOk() const { return (StatusCode() == StatusCode::kSuccess); }

  bool IsError() const { return !IsOk(); }

  static std::string CodeAsString(enum StatusCode c);

 private:
  enum StatusCode status_code_;
  std::string status_msg_;
  int line_of_code_;
  std::string file_name_;
  std::string err_description_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_STATUS_H
