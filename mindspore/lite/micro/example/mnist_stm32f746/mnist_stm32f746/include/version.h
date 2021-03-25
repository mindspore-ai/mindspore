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

#ifndef MINDSPORE_LITE_INCLUDE_VERSION_H_
#define MINDSPORE_LITE_INCLUDE_VERSION_H_

#include "include/lite_utils.h"

namespace mindspore {
namespace lite {
const int ms_version_major = 1;
const int ms_version_minor = 2;
const int ms_version_revision = 0;

/// \brief Global method to get a version string.
///
/// \return The version string of MindSpore Lite.
inline String Version() {
  return "MindSpore Lite " + to_string(ms_version_major) + "." + to_string(ms_version_minor) + "." +
         to_string(ms_version_revision);
}
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_VERSION_H_
