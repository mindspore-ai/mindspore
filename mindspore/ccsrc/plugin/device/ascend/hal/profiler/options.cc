/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <climits>
#include <initializer_list>
#include <memory>
#include <vector>

#include "plugin/device/ascend/hal/profiler/options.h"
#include "utils/ms_context.h"
#include "nlohmann/detail/iterators/iter_impl.hpp"
#include "nlohmann/json.hpp"
#include "profiler/device/profiling.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

constexpr char kOutputPath[] = "output";

namespace mindspore {
namespace profiler {
namespace ascend {
std::string GetOutputPath() {
  auto ascend_profiler = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  const std::string options_str = ascend_profiler->GetProfilingOptions();
  nlohmann::json options_json;
  try {
    options_json = nlohmann::json::parse(options_str);
  } catch (nlohmann::json::parse_error &e) {
    MS_LOG(EXCEPTION) << "Parse profiling option json failed, error:" << e.what();
  }
  auto iter = options_json.find(kOutputPath);
  if (iter != options_json.end() && iter->is_string()) {
    char real_path[PATH_MAX] = {0};
    if ((*iter).size() >= PATH_MAX) {
      MS_LOG(ERROR) << "Path is invalid for profiling.";
      return "";
    }
#if defined(_WIN32) || defined(_WIN64)
    if (_fullpath(real_path, common::SafeCStr(*iter), PATH_MAX) == nullptr) {
      MS_LOG(ERROR) << "Path is invalid for memory profiling.";
      return "";
    }
#else
    if (realpath(common::SafeCStr(*iter), real_path) == nullptr) {
      MS_LOG(ERROR) << "Path is invalid for profiling.";
      return "";
    }
#endif
    return real_path;
  }

  MS_LOG(ERROR) << "Output path is not found when save profiling data";
  return "";
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
