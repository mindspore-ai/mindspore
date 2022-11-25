/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_MACROS_H_
#define MINDSPORE_CORE_MINDAPI_BASE_MACROS_H_

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef BUILDING_CORE_DLL
#define MIND_API __declspec(dllexport)
#define MS_EXPORT __declspec(dllexport)
#define MS_CORE_API __declspec(dllexport)
#define GVAR_DEF(type, name, value) MS_CORE_API inline const type name = value;
#else
#define MIND_API __declspec(dllimport)
#define MS_EXPORT __declspec(dllimport)
#define MS_CORE_API __declspec(dllimport)
#define GVAR_DEF(type, name, value) MS_CORE_API extern const type name;
#endif
#else
#define MIND_API __attribute__((visibility("default")))
#define MS_CORE_API __attribute__((visibility("default")))
#define MS_EXPORT __attribute__((visibility("default")))
#define GVAR_DEF(type, name, value) MS_CORE_API inline const type name = value;
#endif

#ifdef _MSC_VER
#define NO_RETURN
#define ALWAYS_INLINE
#else
#define NO_RETURN __attribute__((noreturn))
#define ALWAYS_INLINE __attribute__((__always_inline__))
#endif

#endif  // MINDSPORE_CORE_MINDAPI_BASE_MACROS_H_
