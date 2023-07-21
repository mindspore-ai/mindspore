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

#ifndef MINDSPORE_CCSRC_C_API_BASE_MACROS_H_
#define MINDSPORE_CCSRC_C_API_BASE_MACROS_H_

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef BUILDING_C_API_DLL
#define MIND_C_API __declspec(dllexport)
#else
#define MIND_C_API __declspec(dllimport)
#endif
#else
#define MIND_C_API __attribute__((visibility("default")))
#endif

#define MAX_DIMS 8

#endif  // MINDSPORE_CCSRC_C_API_BASE_MACROS_H_
