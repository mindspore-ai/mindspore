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

#include <string>

namespace mindspore {
namespace lite {
/// \brief Global method to get a version string.
///
/// \return The version string of MindSpore Lite.
#ifndef MS_VERSION_MAJOR
#define MS_VERSION_MAJOR 0
#endif
#ifndef MS_VERSION_MINOR
#define MS_VERSION_MINOR 7
#endif
#ifndef MS_VERSION_REVISION
#define MS_VERSION_REVISION 0
#endif
std::string Version() {
  return "MindSpore Lite " + std::to_string(MS_VERSION_MAJOR) + "." + std::to_string(MS_VERSION_MINOR) + "." +
         std::to_string(MS_VERSION_REVISION);
}
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_VERSION_H
