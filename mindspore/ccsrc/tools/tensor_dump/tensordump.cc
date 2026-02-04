/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "tools/tensor_dump/tensordump.h"

#include <algorithm>
#include <atomic>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "tools/data_dump/utils.h"
#include "tools/silent_detect/silent_detect_config_parser.h"
#include "tools/silent_detect/silent_detector.h"
#include "utils/distributed_meta.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"

namespace {
template <typename T>
std::string ReplacePlaceholder(const std::string &file_name, const std::string &placeholder, T value) {
  std::string result = file_name;
  const std::string full_placeholder = "{" + placeholder + "}";
  std::ostringstream oss;
  oss << placeholder << value;
  const std::string value_str = oss.str();
  size_t pos = 0;
  while ((pos = result.find(full_placeholder, pos)) != std::string::npos) {
    result.replace(pos, full_placeholder.length(), value_str);
    pos += value_str.length();
  }
  return result;
}
}  // namespace
namespace mindspore {
namespace datadump {

std::string TensorDumpManager::TensorNameToArrayName(std::string tensor_path, std::string data_type, const int mode) {
  const std::string npy_suffix = ".npy";
  const std::string separator = "_";

  tensor_path = ReplacePlaceholder(tensor_path, "step", GetStep(mode));
  tensor_path = ReplacePlaceholder(tensor_path, "rank", GetRankID());
  std::transform(data_type.begin(), data_type.end(), data_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  std::optional<std::string> parent_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(tensor_path, &parent_path, &file_name);
  if (!parent_path.has_value()) {
    parent_path = ".";
  }
  if (!file_name.has_value()) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, failed to extract file name from the arg of 'file', file is "
                  << tensor_path;
    return {};
  }
  std::optional<std::string> realpath = FileUtils::CreateNotExistDirs(*parent_path, true);
  if (EndsWith(*file_name, npy_suffix)) {
    file_name = file_name->substr(0, file_name->length() - npy_suffix.length());
  }
  size_t name_id = TensorDumpManager::GetInstance().FetchAddID();
  std::optional<std::string> new_file_name =
    *file_name + separator + data_type + separator + std::to_string(name_id) + npy_suffix;
  std::optional<std::string> new_file_path;
  FileUtils::ConcatDirAndFileName(&realpath, &new_file_name, &new_file_path);
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, mode is " << (mode ? "PYNATIVE" : "GRAPH")
                            << ", dump file path is " << *new_file_path;
  return *new_file_path;
}

void TensorDumpManager::SetDumpStep(const std::vector<size_t> &steps) {
  valid_steps_ = {steps.begin(), steps.end()};
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, call set dump step, " << steps << " can be dump.";
}

bool TensorDumpManager::NeedDump(const int mode) const {
  MS_EXCEPTION_IF_CHECK_FAIL(mode == kCallFromPython || mode == kCallFromCXX, "Invalid mode");
  auto step = step_.at(mode);
  return valid_steps_.empty() || valid_steps_.find(step) != valid_steps_.end();
}

size_t TensorDumpManager::GetStep(const int mode) const {
  MS_EXCEPTION_IF_CHECK_FAIL(mode == kCallFromPython || mode == kCallFromCXX, "Invalid mode");
  return step_.at(mode);
}

void TensorDumpManager::UpdateStep(const int mode) {
  MS_EXCEPTION_IF_CHECK_FAIL(mode == kCallFromPython || mode == kCallFromCXX, "Invalid mode");
  step_.at(mode) += 1;
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, call from " << (mode ? "python(pynative)" : "c++(graph/jit)")
                            << ", after update step, current step is  " << step_.at(mode);
}

size_t TensorDumpManager::FetchAddID() { return id_.fetch_add(1, std::memory_order_relaxed); }

std::string TensorDumpManager::ProcessFileName(const std::string &filename, const std::string &dtype, const int mode) {
  constexpr std::string_view step_flag{"<tensordump-update-step>"};
  if (filename == step_flag) {
    UpdateStep(mode);
    return {};
  }
  if (!NeedDump(mode)) {
    return {};
  }
  if (StartsWith(filename, silentdetect::kSilentDetectFeatureFlag)) {
    return filename;
  }
  return TensorNameToArrayName(filename, dtype, mode);
}

TensorDumpManager::task_type TensorDumpManager::GetTaskType(const std::string &tensor_name, const int mode) {
  constexpr std::string_view step_flag{"<tensordump-update-step>"};
  if (datadump::StartsWith(tensor_name, silentdetect::kSilentDetectFeatureFlag)) {
    return task_type::silentdetect;
  }
  if (tensor_name == step_flag) {
    return task_type::update_step;
  }
  if (!datadump::TensorDumpManager::GetInstance().NeedDump(mode)) {
    return task_type::skip;
  }
  return task_type::dump;
}

void TensorDumpManager::ExecTask(task_type type, const std::string &tensor_name, tensor::TensorPtr tensor,
                                 const int mode) {
  switch (type) {
    case task_type::dump: {
      std::string data_type = TypeIdToType(tensor->data_type())->ToString();
      auto file_name = ProcessFileName(tensor_name, data_type);
      SaveTensor2NPY(file_name, tensor);
      break;
    }
    case task_type::update_step:
      UpdateStep(mode);
      break;
    case task_type::skip:
      break;
    case task_type::silentdetect: {
      const std::string sc_flag{silentdetect::kSilentDetectFeatureFlag};
      MS_EXCEPTION_IF_CHECK_FAIL(tensor_name.find(sc_flag) == 0,
                                 "silentdetect task tensor name must start with " + sc_flag);
      auto name = tensor_name.substr(sc_flag.length());
      silentdetect::SilentDetect(name, tensor);
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "unknown task type: " << type;
    }
  }
}

void TensorDumpManager::Exec(const std::string &tensor_name, tensor::TensorPtr tensor, const int mode) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto type = GetTaskType(tensor_name, mode);
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, call from " << (mode ? "python(pynative)" : "c++(graph/jit)")
                            << ", task type is " << type;
  ExecTask(type, tensor_name, tensor, mode);
}

}  // namespace datadump
}  // namespace mindspore
