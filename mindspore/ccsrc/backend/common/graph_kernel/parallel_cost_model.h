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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_COST_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_COST_MODEL_H_

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "base/base.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/graph_kernel/parallel_cost_model.h"
#include "backend/common/session/kernel_graph.h"
#include "include/common/utils/python_adapter.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
class DimInfo {
 public:
  DimInfo() = default;
  virtual ~DimInfo() {}
  virtual std::string ToString() = 0;
};

class CommonDimInfo : public DimInfo {
 public:
  explicit CommonDimInfo(size_t dim) : dim_info_(dim) {}
  ~CommonDimInfo() {}
  void set_dim_info(size_t d) { dim_info_ = d; }
  size_t dim_info() const { return dim_info_; }
  std::string ToString() override;

 private:
  size_t dim_info_;
};

using DimInfoPtr = std::shared_ptr<DimInfo>;
using CommonDimInfoPtr = std::shared_ptr<CommonDimInfo>;

class FusionInfo {
 public:
  FusionInfo() = default;
  explicit FusionInfo(const std::string &type) : fusion_type_(type) {}
  virtual ~FusionInfo() = default;
  std::string FusionType() const { return fusion_type_; }
  virtual bool ExistTypeInfo() { return false; }

 private:
  std::string fusion_type_{"none"};
};

class BlockFusionInfo : public FusionInfo {
 public:
  BlockFusionInfo() : FusionInfo("block_fusion") {}
  ~BlockFusionInfo() = default;
  bool ExistTypeInfo() override { return false; }
};

class BlockPipelineFusionInfo : public FusionInfo {
 public:
  explicit BlockPipelineFusionInfo(const std::vector<std::vector<int>> &ids)
      : FusionInfo("block_pipeline_fusion"), pipeline_ids_(ids) {}
  ~BlockPipelineFusionInfo() = default;
  bool ExistTypeInfo() override { return true; }
  std::vector<std::vector<int>> PipelineIds() { return pipeline_ids_; }

 private:
  std::vector<std::vector<int>> pipeline_ids_;
};

using FusionInfoPtr = std::shared_ptr<FusionInfo>;
using BlockFusionInfoPtr = std::shared_ptr<BlockFusionInfo>;
using BlockPipelineFusionInfoPtr = std::shared_ptr<BlockPipelineFusionInfo>;

class ParallelCostModel {
 public:
  ParallelCostModel() {}
  ~ParallelCostModel() {}
  int64_t GetNodeCalAmount(const AnfNodePtr &node) const;
  std::tuple<std::vector<DimInfoPtr>, int, FusionInfoPtr> CalFuseInfo(const AnfNodePtrList &nodes) const;

 private:
  FusionInfoPtr ProcessFusionInfo(const py::object &fusion_type, const py::object &type_info) const;
};

using ParallelCostModelPtr = std::shared_ptr<ParallelCostModel>;

class ParellelCostModelWarehouse {
 public:
  static ParellelCostModelWarehouse &Instance() {
    static ParellelCostModelWarehouse instance = ParellelCostModelWarehouse();
    return instance;
  }
  ParallelCostModelPtr GetParallelCostModel(const std::string &target) const;

 private:
  ParellelCostModelWarehouse() { cost_model_ = std::make_shared<ParallelCostModel>(); }
  ~ParellelCostModelWarehouse() = default;
  ParallelCostModelPtr cost_model_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_COST_MODEL_H_
