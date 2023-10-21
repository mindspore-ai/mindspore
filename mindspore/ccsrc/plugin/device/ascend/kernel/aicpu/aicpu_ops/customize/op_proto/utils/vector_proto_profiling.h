/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file vector_proto_profiling.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTIL_VECTOR_PROTO_PROFILING_H__H_
#define CUSTOMIZE_OP_PROTO_UTIL_VECTOR_PROTO_PROFILING_H__H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>

#include "op_log.h"

const bool vector_prof_switch = false;

#define PROFILING_PROTO_INIT(op_type)                                                                         \
  auto profiling_op_name = op_type;                                                                           \
  std::unique_ptr<std::vector<std::chrono::time_point<std::chrono::steady_clock>>> time_vector_ptr = nullptr; \
  if (vector_prof_switch) {                                                                                   \
    time_vector_ptr = std::unique_ptr<std::vector<std::chrono::time_point<std::chrono::steady_clock>>>(       \
      new std::vector<std::chrono::time_point<std::chrono::steady_clock>>);                                   \
    time_vector_ptr->reserve(5);                                                                              \
    time_vector_ptr->push_back(std::chrono::steady_clock::now());                                             \
  }

#define PROFILING_PROTO_AFTER_GET_SHAPE_REG()                     \
  if (time_vector_ptr != nullptr) {                               \
    time_vector_ptr->push_back(std::chrono::steady_clock::now()); \
  }

#define PROFILING_PROTO_AFTER_INFER_SHAPE_REG()                   \
  if (time_vector_ptr != nullptr) {                               \
    time_vector_ptr->push_back(std::chrono::steady_clock::now()); \
  }

#define PROFILING_PROTO_END()                                                                                      \
  if (time_vector_ptr != nullptr) {                                                                                \
    time_vector_ptr->push_back(std::chrono::steady_clock::now());                                                  \
    const vector<string> profiline_cost_name_list = {"GET_SHAPE", "INFER_SHAPE", "SET_SHAPE"};                     \
    if (time_vector_ptr->size() == profiline_cost_name_list.size() + 1) {                                          \
      for (size_t i = 1; i < time_vector_ptr->size(); i++) {                                                       \
        auto profiling_cast =                                                                                      \
          std::chrono::duration_cast<std::chrono::microseconds>((*time_vector_ptr)[i] - (*time_vector_ptr)[i - 1]) \
            .count();                                                                                              \
        OP_EVENT(profiling_op_name, "[INFER_PROF][%s]: %d(us)", profiline_cost_name_list[i - 1].c_str(),           \
                 static_cast<int>(profiling_cast));                                                                \
      }                                                                                                            \
    }                                                                                                              \
  }
#endif  // CUSTOMIZE_OP_PROTO_UTIL_VECTOR_PROTO_PROFILING_H__H_
