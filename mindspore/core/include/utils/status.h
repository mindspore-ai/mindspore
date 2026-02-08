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
#ifndef MINDSPORE_INCLUDE_API_STATUS_H
#define MINDSPORE_INCLUDE_API_STATUS_H

#include <memory>
#include <string>
#include <vector>
#include <ostream>
#include <climits>
#include "utils/dual_abi_helper.h"
#include "mindapi/base/macros.h"
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

  // Lite  // Common error code, range: [-1, -100)
  kLiteError = kLite | (0x0FFFFFFF & -1),            /**< Common error code. */
  kLiteNullptr = kLite | (0x0FFFFFFF & -2),          /**< NULL pointer returned.*/
  kLiteParamInvalid = kLite | (0x0FFFFFFF & -3),     /**< Invalid parameter.*/
  kLiteNoChange = kLite | (0x0FFFFFFF & -4),         /**< No change. */
  kLiteSuccessExit = kLite | (0x0FFFFFFF & -5),      /**< No error but exit. */
  kLiteMemoryFailed = kLite | (0x0FFFFFFF & -6),     /**< Fail to create memory. */
  kLiteNotSupport = kLite | (0x0FFFFFFF & -7),       /**< Fail to support. */
  kLiteThreadPoolError = kLite | (0x0FFFFFFF & -8),  /**< Error occur in thread pool. */
  kLiteUninitializedObj = kLite | (0x0FFFFFFF & -9), /**< Object is not initialized. */
  kLiteFileError = kLite | (0x0FFFFFFF & -10),       /**< Invalid file. */
  kLiteServiceDeny = kLite | (0x0FFFFFFF & -11),     /**< Denial of service. */
  kLiteModelRebuild = kLite | (0x0FFFFFFF & -12),    /**< Model has been built. */

  // Executor error code, range: [-100,-200)
  kLiteOutOfTensorRange = kLite | (0x0FFFFFFF & -100),           /**< Failed to check range. */
  kLiteInputTensorError = kLite | (0x0FFFFFFF & -101),           /**< Failed to check input tensor. */
  kLiteReentrantError = kLite | (0x0FFFFFFF & -102),             /**< Exist executor running. */
  kLiteLLMWaitProcessTimeOut = kLite | (0x0FFFFFFF & -103),      /**< Wait to be processed time out. */
  kLiteLLMKVCacheNotExist = kLite | (0x0FFFFFFF & -104),         /**< KV Cache not exist. */
  kLiteLLMRepeatRequest = kLite | (0x0FFFFFFF & -105),           /**< repeat request. */
  kLiteLLMRequestAlreadyCompleted = kLite | (0x0FFFFFFF & -106), /**< request already complete!. */
  kLiteLLMEngineFinalized = kLite | (0x0FFFFFFF & -107),         /**< llm engine finalized. */
  kLiteLLMNotYetLink = kLite | (0x0FFFFFFF & -108),              /**< decoder cluster is no link with prompt. */
  kLiteLLMAlreadyLink = kLite | (0x0FFFFFFF & -109),  /**< decoder cluster is already linked with prompt cluster! */
  kLiteLLMLinkFailed = kLite | (0x0FFFFFFF & -110),   /**< decoder cluster link with prompt cluster failed! */
  kLiteLLMUnlinkFailed = kLite | (0x0FFFFFFF & -111), /**< decoder cluster unlink with prompt cluster failed */
  kLiteLLMNofiryPromptUnlinkFailed =
    kLite | (0x0FFFFFFF & -112), /**< decoder cluster notify prompt cluster do unlink failed */
  kLiteLLMClusterNumExceedLimit = kLite | (0x0FFFFFFF & -113), /**< cluster num exceed limit */
  kLiteLLMProcessingLink = kLite | (0x0FFFFFFF & -114),        /**< link is current processing. */
  kLiteLLMOutOfMemory = kLite | (0x0FFFFFFF & -115),           /**< device out of memory. */
  kLiteLLMPrefixAlreadyExist = kLite | (0x0FFFFFFF & -116),    /**< Prefix has already existed. */
  kLiteLLMPrefixNotExist = kLite | (0x0FFFFFFF & -117),        /**< Prefix does not exist. */
  kLiteLLMSeqLenOverLimit = kLite | (0x0FFFFFFF & -118),       /**< Sequence length exceed limit. */
  kLiteLLMNoFreeBlock = kLite | (0x0FFFFFFF & -119),           /**< No free block. */
  kLiteLLMBlockOutOfMemory = kLite | (0x0FFFFFFF & -120),      /**< Block is out of memory. */

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

class MS_CORE_API Status {
 public:
  /// \brief Constructor of Status.
  Status();
  /// \brief Constructor of Status.
  ///
  /// \param[in] status_code Status code.
  ///
  /// \param[in] status_msg Status message.
  inline Status(enum StatusCode status_code, const std::string &status_msg = "");  // NOLINT(runtime/explicit)
  /// \brief Constructor of Status.
  inline Status(const StatusCode code, int line_of_code, const char *file_name, const std::string &extra = "");
  /// \brief Destructor of Status.
  ~Status() = default;
  /// \brief Status code of status.
  ///
  /// \return Enum of status code.
  enum StatusCode StatusCode() const;
  /// \brief Exchange status to string.
  ///
  /// \return Status code exchanged to string.
  inline std::string ToString() const;
  /// \brief Get line of status code.
  ///
  /// \return Line of code to get.
  int GetLineOfCode() const;
  /// \brief Get file name of status.
  ///
  /// \return File name to get.
  inline std::string GetFileName() const;
  /// \brief Get error description of status.
  ///
  /// \return Error description to get.
  inline std::string GetErrDescription() const;
  /// \brief Get error description of status.
  ///
  /// \param[in] err_description Error description to be set.
  inline std::string SetErrDescription(const std::string &err_description);
  /// \brief Status message to be set.
  ///
  /// \param[in] status_msg Status message to be set.
  inline void SetStatusMsg(const std::string &status_msg);
  /// \brief Operator <<.
  MS_CORE_API friend std::ostream &operator<<(std::ostream &os, const Status &s);
  /// \brief Operator ==.
  bool operator==(const Status &other) const;
  /// \brief Operator ==.
  bool operator==(enum StatusCode other_code) const;
  /// \brief Operator !=.
  bool operator!=(const Status &other) const;
  /// \brief Operator !=.
  bool operator!=(enum StatusCode other_code) const;
  /// \brief Operator bool().
  explicit operator bool() const;
  /// \brief Operator int().
  explicit operator int() const;
  /// \brief Getting back the status of OK.
  ///
  /// \return Status Code of ok.
  static Status OK();
  /// \brief Getting back if it is ok.
  ///
  /// \return True if it is ok.
  bool IsOk() const;
  /// \brief Getting back if it is error.
  ///
  /// \return True if it is error.
  bool IsError() const;
  /// \brief Getting back the code as string.
  ///
  /// \return The code name as string type.
  static inline std::string CodeAsString(enum StatusCode c);

 private:
  // api without std::string
  Status(enum StatusCode status_code, const std::vector<char> &status_msg);
  Status(enum StatusCode code, int line_of_code, const char *file_name, const std::vector<char> &extra);
  std::vector<char> ToCString() const;
  std::vector<char> GetFileNameChar() const;
  std::vector<char> GetErrDescriptionChar() const;
  std::vector<char> SetErrDescription(const std::vector<char> &err_description);
  void SetStatusMsgChar(const std::vector<char> &status_msg);
  static std::vector<char> CodeAsCString(enum StatusCode c);

  struct Data;
  std::shared_ptr<Data> data_;
};

Status::Status(enum StatusCode status_code, const std::string &status_msg)
    : Status(status_code, StringToChar(status_msg)) {}
Status::Status(const enum StatusCode code, int line_of_code, const char *file_name, const std::string &extra)
    : Status(code, line_of_code, file_name, StringToChar(extra)) {}
std::string Status::ToString() const { return CharToString(ToCString()); }
std::string Status::GetFileName() const { return CharToString(GetFileNameChar()); }
std::string Status::GetErrDescription() const { return CharToString(GetErrDescriptionChar()); }
std::string Status::SetErrDescription(const std::string &err_description) {
  return CharToString(SetErrDescription(StringToChar(err_description)));
}
void Status::SetStatusMsg(const std::string &status_msg) { SetStatusMsgChar(StringToChar(status_msg)); }
std::string Status::CodeAsString(enum StatusCode c) { return CharToString(CodeAsCString(c)); }
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_STATUS_H
