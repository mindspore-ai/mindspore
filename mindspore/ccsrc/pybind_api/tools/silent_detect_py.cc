/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>

#include "tools/silent_detect/checksum/checksum_mgr.h"
#include "tools/silent_detect/silent_detect_config_parser.h"
#include "mindspore/ccsrc/pybind_api/tools/tools_api.h"

namespace py = pybind11;
namespace mindspore {
namespace silentdetect {
using checksum::CheckSumMgr;
void RegSilentDetect(py::module *m) {
  m->def(
     "sdc_detect_start", []() { CheckSumMgr::GetInstance().CheckSumStart(); }, "Start SDC detect")
    .def(
      "sdc_detect_stop", []() { CheckSumMgr::GetInstance().CheckSumStop(); }, "Stop SDC detect")
    .def(
      "get_sdc_detect_result", []() { return CheckSumMgr::GetInstance().GetCheckSumResult(); }, "Get SDC detect result")
    .def(
      "is_silent_detect_enable", []() { return SilentDetectConfigParser::GetInstance().IsEnable(); },
      "Is silent detect enable")
    .def(
      "is_silent_detect_with_checksum", []() { return SilentDetectConfigParser::GetInstance().IsWithChecksum(); },
      "Is silent detect with check sum")
    .def(
      "get_silent_detect_config",
      [](const std::string &name) { return SilentDetectConfigParser::GetInstance().GetConfig(name); },
      "Get silent detect config")
    .def(
      "get_silent_detect_feature_name",
      [](const std::string &name) { return SilentDetectConfigParser::GetInstance().GetSilentDetectFeatureName(name); },
      "Get silent detect feature name");
}
}  // namespace silentdetect
}  // namespace mindspore
