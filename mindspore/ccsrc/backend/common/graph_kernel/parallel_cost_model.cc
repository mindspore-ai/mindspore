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

#include "backend/common/graph_kernel/parallel_cost_model.h"

#include <algorithm>
#include "kernel/akg/akg_kernel_json_generator.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore::graphkernel {
std::string CommonDimInfo::ToString() {
  std::ostringstream buffer;
  buffer << "Dim(" << dim_info_ << ")";
  return buffer.str();
}

int64_t ParallelCostModel::GetNodeCalAmount(const AnfNodePtr &node) const {
  nlohmann::json json_desc;
  AnfNodePtrList nodes = {node};
  DumpOption dump_option;
  if (!AnfToJsonDesc(nodes, dump_option, &json_desc)) {
    MS_LOG(EXCEPTION) << "Collect json desc for node [" << node->fullname_with_scope() << "] failed";
  }

  auto json_desc_str = json_desc.dump();
  auto ret = python_adapter::CallPyFn(kGraphKernelModule, kGraphKernelGetNodeCalAmount, json_desc_str);
  auto bottleneck = py::cast<int64_t>(ret);
  if (bottleneck == -1) {
    MS_LOG(EXCEPTION) << "CallPyFn: [" << kGraphKernelGetNodeCalAmount << "] return invalid result. input json:\n"
                      << json_desc_str;
  }
  return bottleneck;
}

std::tuple<std::vector<DimInfoPtr>, int, FusionInfoPtr> ParallelCostModel::CalFuseInfo(
  const AnfNodePtrList &nodes) const {
  nlohmann::json json_desc;
  std::vector<AnfNodePtrList> graphs;
  (void)std::transform(nodes.begin(), nodes.end(), std::back_inserter(graphs),
                       [](const AnfNodePtr &node) -> AnfNodePtrList { return {node}; });
  DumpOption dump_option;
  if (!AnfToJsonDesc(graphs, dump_option, &json_desc)) {
    MS_LOG(EXCEPTION) << "Collect json desc failed.";
  }

  auto json_desc_str = json_desc.dump();
  auto ret = python_adapter::CallPyFn(kGraphKernelModule, kGraphKernelEstimateOps, json_desc_str);
  if (py::isinstance<py::none>(ret)) {
    MS_LOG(EXCEPTION) << "CallPyFn: [" << kGraphKernelEstimateOps << "] return invalid result. input json:\n"
                      << json_desc_str;
  }

  py::tuple ret_tuple = py::cast<py::tuple>(ret);
  if (!py::isinstance<py::tuple>(ret_tuple) || ret_tuple.size() != 4) {
    MS_LOG(EXCEPTION) << "Parallel cost model should return a tuple with 4 elements!";
  }

  std::vector<DimInfoPtr> dim_infos;
  py::list dim_list = py::cast<py::list>(ret_tuple[0]);
  for (size_t i = 0; i < dim_list.size(); ++i) {
    dim_infos.push_back(std::make_shared<CommonDimInfo>(py::cast<int>(dim_list[i])));
  }
  int benefit = py::cast<int>(ret_tuple[1]);
  auto fusion_info = ProcessFusionInfo(ret_tuple[2], ret_tuple[3]);

  return std::make_tuple(dim_infos, benefit, fusion_info);
}

FusionInfoPtr ParallelCostModel::ProcessFusionInfo(const py::object &fusion_type, const py::object &type_info) const {
  if (!py::isinstance<py::str>(fusion_type)) {
    MS_LOG(EXCEPTION) << "Fusion type for parallel should be of type str!";
  }

  std::string fusion_type_name = py::cast<std::string>(fusion_type);

  FusionInfoPtr fusion_info;
  if (fusion_type_name == "block_fusion") {
    fusion_info = std::make_shared<BlockFusionInfo>();
  } else if (fusion_type_name == "block_pipeline_fusion") {
    if (!py::isinstance<py::list>(type_info)) {
      MS_LOG(EXCEPTION) << "Fusion type info for block pipe fusion type should be of type list!";
    }
    std::vector<std::vector<int>> pipeline_ids;
    py::list pipeline_ids_list = py::cast<py::list>(type_info);
    for (size_t i = 0; i < pipeline_ids_list.size(); ++i) {
      std::vector<int> part_ids;
      py::list inner_ids_list = py::cast<py::list>(pipeline_ids_list[i]);
      for (size_t j = 0; j < inner_ids_list.size(); ++j) {
        part_ids.push_back(py::cast<int>(inner_ids_list[j]));
      }
      pipeline_ids.push_back(part_ids);
    }

    fusion_info = std::make_shared<BlockPipelineFusionInfo>(pipeline_ids);
  } else {
    MS_LOG(EXCEPTION) << "parallel fusion type: " << fusion_type_name
                      << " is not in supported list [block_fusion, block_pipeline_fusion]";
  }
  return fusion_info;
}

ParallelCostModelPtr ParellelCostModelWarehouse::GetParallelCostModel(const std::string &target) const {
  if (target != kGPUDevice && target != kAscendDevice) {
    MS_LOG(EXCEPTION) << "Parallel cost model supports " << kGPUDevice << " and " << kAscendDevice << ", not "
                      << target;
  }
  return cost_model_;
}
}  // namespace mindspore::graphkernel
