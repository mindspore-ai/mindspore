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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_VISIBLE_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_VISIBLE_H_

#include <utility>

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef COMMON_DLL
#define COMMON_EXPORT __declspec(dllexport)
#else
#define COMMON_EXPORT __declspec(dllimport)
#endif
#define COMMON_LOCAL
#else
#define COMMON_EXPORT __attribute__((visibility("default")))
#define COMMON_LOCAL __attribute__((visibility("hidden")))
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef PROFILER_DLL
#define PROFILER_EXPORT __declspec(dllexport)
#else
#define PROFILER_EXPORT __declspec(dllimport)
#endif
#define PROFILER_LOCAL
#else
#define PROFILER_EXPORT __attribute__((visibility("default")))
#define PROFILER_LOCAL __attribute__((visibility("hidden")))
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef PYNATIVE_DLL
#define PYNATIVE_EXPORT __declspec(dllexport)
#else
#define PYNATIVE_EXPORT __declspec(dllimport)
#endif
#else
#define PYNATIVE_EXPORT __attribute__((visibility("default")))
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef PYNATIVE_UTILS_DLL
#define PYNATIVE_UTILS_EXPORT __declspec(dllexport)
#else
#define PYNATIVE_UTILS_EXPORT __declspec(dllimport)
#endif
#else
#define PYNATIVE_UTILS_EXPORT __attribute__((visibility("default")))
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef FRONTEND_DLL
#define FRONTEND_EXPORT __declspec(dllexport)
#else
#define FRONTEND_EXPORT __declspec(dllimport)
#endif
#else
#define FRONTEND_EXPORT __attribute__((visibility("default")))
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef PYBIND_DLL
#define PYBIND_EXPORT __declspec(dllexport)
#else
#define PYBIND_EXPORT __declspec(dllimport)
#endif
#else
#define PYBIND_EXPORT __attribute__((visibility("default")))
#endif

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_VISIBLE_H_
