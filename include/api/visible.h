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
#ifndef MINDSPORE_INCLUDE_API_VISIBLE_H
#define MINDSPORE_INCLUDE_API_VISIBLE_H

#ifndef MS_API
#ifdef _WIN32
#ifdef BUILDING_DLL
#define MS_API __declspec(dllexport)
#else
#define MS_API __declspec(dllimport)
#endif
#else
#define MS_API __attribute__((visibility("default")))
#endif  // _WIN32
#endif

#ifdef _MSC_VER
#ifdef BUILDING_DATASET_DLL
#define DATASET_API __declspec(dllexport)
#else
#define DATASET_API __declspec(dllimport)
#endif
#else
#define DATASET_API __attribute__((visibility("default")))
#endif  // _MSC_VER

#endif  // MINDSPORE_INCLUDE_API_VISIBLE_H
