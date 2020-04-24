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

#ifndef MINDSPORE_CCSRC_PIPELINE_PIPELINE_GE_H_
#define MINDSPORE_CCSRC_PIPELINE_PIPELINE_GE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <mutex>

#include "pybind11/pybind11.h"
#include "pipeline/base.h"
#include "operator/ops.h"

namespace mindspore {
namespace pipeline {
namespace py = pybind11;

void SetGeOption(const std::map<std::string, std::string> &options);

void RunGEInitGraph(const py::dict &init_params, const std::string &phase);

py::object ExecDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::tuple &args,
                       const std::string &phase = "train");

FuncGraphPtr BuildDFGraph(const std::map<std::string, ExecutorInfoPtr> &info, const py::dict &init_params,
                          const std::string &phase, const py::object &broadcast_params = {});

// init and exec dataset sub graph for GE backend
bool InitExecDatasetGe(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, const std::string &phase);

void ExportDFGraph(const std::string &file_name, const std::string &phase);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PIPELINE_GE_H_
