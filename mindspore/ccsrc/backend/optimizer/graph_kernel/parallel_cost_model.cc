/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/graph_kernel/parallel_cost_model.h"

#include <algorithm>

#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "pipeline/jit/parse/python_adapter.h"

namespace mindspore {
namespace opt {
std::string CommonDimInfo::ToString() {
  std::ostringstream buffer;
  buffer << "Dim(" << dim_info_ << ")";
  return buffer.str();
}

int ParallelCostModel::GetNodeCalAmount(const AnfNodePtr &node) {
  nlohmann::json json_desc;
  AnfNodePtrList nodes = {node};
  DumpOption dump_option;
  if (!AnfToJsonDesc(nodes, dump_option, &json_desc)) {
    MS_LOG(EXCEPTION) << "Collect json desc failed.";
  }

  auto json_desc_str = json_desc.dump();
  auto ret = parse::python_adapter::CallPyFn(kGraphKernelModule, kGraphKernelGetNodeCalAmount, json_desc_str);
  if (py::isinstance<py::none>(ret)) {
    MS_LOG(EXCEPTION) << "CallPyFn: [" << kGraphKernelSplitFunc << "] return invalid result. input json:\n"
                      << json_desc_str;
  }
  return py::cast<int>(ret);
}

std::tuple<std::vector<DimInfoPtr>, int> ParallelCostModel::CalFuseInfo(const AnfNodePtrList &nodes) {
  nlohmann::json json_desc;
  std::vector<AnfNodePtrList> graphs;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(graphs),
                 [](const AnfNodePtr &node) -> AnfNodePtrList { return {node}; });
  DumpOption dump_option;
  if (!AnfToJsonDesc(graphs, dump_option, &json_desc)) {
    MS_LOG(EXCEPTION) << "Collect json desc failed.";
  }

  auto json_desc_str = json_desc.dump();
  auto ret = parse::python_adapter::CallPyFn(kGraphKernelModule, kGraphKernelEstimateOps, json_desc_str);
  if (py::isinstance<py::none>(ret)) {
    MS_LOG(EXCEPTION) << "CallPyFn: [" << kGraphKernelSplitFunc << "] return invalid result. input json:\n"
                      << json_desc_str;
  }

  py::tuple ret_tuple = py::cast<py::tuple>(ret);
  if (!py::isinstance<py::tuple>(ret_tuple) || ret_tuple.size() != 2) {
    MS_LOG(EXCEPTION) << "Parallel cost model should return a tuple with two elements!";
  }

  std::vector<DimInfoPtr> dim_infos;
  py::list dim_list = py::cast<py::list>(ret_tuple[0]);
  for (size_t i = 0; i < dim_list.size(); ++i) {
    dim_infos.push_back(std::make_shared<CommonDimInfo>(py::cast<int>(dim_list[i])));
  }
  int benefit = py::cast<int>(ret_tuple[1]);

  return std::make_tuple(dim_infos, benefit);
}

ParallelCostModelPtr ParellelCostModelWarehouse::GetParallelCostModel(const std::string &target) {
  if (target != kGPUDevice) {
    MS_LOG(EXCEPTION) << "Parallel cost model only support " << kGPUDevice << " now.";
  }
  return cost_model_;
}
}  // namespace opt
}  // namespace mindspore
