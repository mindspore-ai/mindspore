/**
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

#ifndef MINDSPORE_CCSRC_UTILS_VISIBLE_H_
#define MINDSPORE_CCSRC_UTILS_VISIBLE_H_

namespace mindspore {
// refer to https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define MS_EXPORT __attribute__((dllexport))
#else
#define MS_EXPORT __declspec(dllexport)  // Note: actually gcc seems to also supports this syntax.
#endif
#else
#ifdef __GNUC__
#define MS_EXPORT __attribute__((dllimport))
#else
#define MS_EXPORT __declspec(dllimport)  // Note: actually gcc seems to also supports this syntax.
#endif
#endif
#define MS_LOCAL
#else
#define MS_EXPORT __attribute__((visibility("default")))
#define MS_LOCAL __attribute__((visibility("hidden")))
#endif

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_VISIBLE_H_
