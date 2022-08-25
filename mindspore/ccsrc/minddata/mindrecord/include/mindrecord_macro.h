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
#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_MINDRECORD_MACRO_H
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_MINDRECORD_MACRO_H
#ifdef _MSC_VER
#ifdef BUILDING_MINDRECORD_DLL
#define MINDRECORD_API __declspec(dllexport)
#else
#define MINDRECORD_API __declspec(dllimport)
#endif
#else
#define MINDRECORD_API __attribute__((visibility("default")))
#endif  // _MSC_VER

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_MINDRECORD_MACRO_H
