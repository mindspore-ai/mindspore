/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "common/util/error_manager/error_manager.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <cstdarg>
#include <securec.h>

#include "mmpa/mmpa_api.h"
#include "toolchain/slog.h"

#define GE_MODULE_NAME static_cast<int32_t>(GE)

const std::string kParamCheckErrorSuffix = "8888";

namespace {
#ifdef __GNUC__
const error_message::char_t *const kErrorCodePath = "../conf/error_manager/error_code.json";
const error_message::char_t *const kSeparator = "/";
#else
const error_message::char_t *const kErrorCodePath = "..\\conf\\error_manager\\error_code.json";
const error_message::char_t *const kSeparator = "\\";
#endif

const error_message::char_t *const kErrorList = "error_info_list";
const error_message::char_t *const kErrCode = "ErrCode";
const error_message::char_t *const kErrMessage = "ErrMessage";
const error_message::char_t *const kArgList = "Arglist";
const uint64_t kLength = 2UL;
}  // namespace

///
/// @brief Obtain ErrorManager instance
/// @return ErrorManager instance
///
ErrorManager &ErrorManager::GetInstance() {
  static ErrorManager instance;
  return instance;
}

///
/// @brief init
/// @param [in] path: current so path
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::Init(const std::string path) {
  return 0;
}

///
/// @brief init
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::Init() {
  return 0;
}

int32_t ErrorManager::ReportInterErrMessage(const std::string error_code, const std::string &error_msg) {
  return 0;
}

///
/// @brief report error message
/// @param [in] error_code: error code
/// @param [in] args_map: parameter map
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReportErrMessage(const std::string error_code,
                                       const std::map<std::string, std::string> &args_map) {
  return 0;
}

std::string ErrorManager::GetErrorMessage() {
  return "";
}

std::string ErrorManager::GetWarningMessage() {
  return "";
}

///
/// @brief output error message
/// @param [in] handle: print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::OutputErrMessage(int32_t handle) {
  return 0;
}

///
/// @brief output message
/// @param [in] handle: print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::OutputMessage(int32_t handle) {
  return 0;
}

///
/// @brief parse json file
/// @param [in] path: json path
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ParseJsonFile(const std::string path) {
  return 0;
}

///
/// @brief read json file
/// @param [in] file_path: json path
/// @param [in] handle:  print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReadJsonFile(const std::string &file_path, void *const handle) {
  return 0;
}

///
/// @brief report error message
/// @param [in] error_code: error code
/// @param [in] vector parameter key, vector parameter value
/// @return int 0(success) -1(fail)
///
void ErrorManager::ATCReportErrMessage(const std::string error_code, const std::vector<std::string> &key,
                                       const std::vector<std::string> &value) {}

///
/// @brief report graph compile failed message such as error code and op_name in mustune case
/// @param [in] msg: failed message map, key is error code, value is op_name
/// @param [out] classified_msg: classified_msg message map, key is error code, value is op_name vector
///
void ErrorManager::ClassifyCompileFailedMsg(const std::map<std::string, std::string> &msg,
                                            std::map<std::string,
                                            std::vector<std::string>> &classified_msg) {}

///
/// @brief report graph compile failed message such as error code and op_name in mustune case
/// @param [in] root_graph_name: root graph name
/// @param [in] msg: failed message map, key is error code, value is op_name
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReportMstuneCompileFailedMsg(const std::string &root_graph_name,
                                                   const std::map<std::string, std::string> &msg) {
  return 0;
}

///
/// @brief get graph compile failed message in mustune case
/// @param [in] graph_name: graph name
/// @param [out] msg_map: failed message map, key is error code, value is op_name list
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::GetMstuneCompileFailedMsg(const std::string &graph_name, std::map<std::string,
                                            std::vector<std::string>> &msg_map) {
  return 0;
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetErrorMsgContainerByWorkId(uint64_t work_id) {
  auto iter = error_message_per_work_id_.find(work_id);
  if (iter == error_message_per_work_id_.end()) {
    (void)error_message_per_work_id_.emplace(work_id, std::vector<ErrorItem>());
    iter = error_message_per_work_id_.find(work_id);
  }
  return iter->second;
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetWarningMsgContainerByWorkId(uint64_t work_id) {
  auto iter = warning_messages_per_work_id_.find(work_id);
  if (iter == warning_messages_per_work_id_.end()) {
    (void)warning_messages_per_work_id_.emplace(work_id, std::vector<ErrorItem>());
    iter = warning_messages_per_work_id_.find(work_id);
  }
  return iter->second;
}

void ErrorManager::GenWorkStreamIdDefault() {}

void ErrorManager::GenWorkStreamIdBySessionGraph(const uint64_t session_id, const uint64_t graph_id) {}

void ErrorManager::ClearErrorMsgContainerByWorkId(const uint64_t work_stream_id) {}

void ErrorManager::ClearWarningMsgContainerByWorkId(const uint64_t work_stream_id) {}

void ErrorManager::SetErrorContext(error_message::Context error_context) {}

error_message::Context &ErrorManager::GetErrorManagerContext() {
  static error_message::Context context;
  return context;
}

void ErrorManager::SetStage(const std::string &first_stage, const std::string &second_stage) {}

void ErrorManager::SetStage(const error_message::char_t *first_stage, const size_t first_len,
                            const error_message::char_t *second_stage, const size_t second_len) {}

bool ErrorManager::IsInnerErrorCode(const std::string &error_code) const { return true; }
