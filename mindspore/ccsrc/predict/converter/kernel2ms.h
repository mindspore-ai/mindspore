/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_KERNEL_TO_MS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_KERNEL_TO_MS_H_

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <utility>
#include "session/kernel_graph.h"
#include "predict/converter/executor_tensor.h"
#include "predict/schema/inner/ms_generated.h"
#include "predict/converter/attr_utils/convert_util.h"

static constexpr size_t kTupleGetItemIndex = 2;
namespace mindspore {
namespace executor {
using KernelGraphPtr = std::shared_ptr<mindspore::session::KernelGraph>;
enum ConvertMode { kConvertCpuMode, kConvertAscendMode, kConvertUnused };
enum TargetMode { kCPUTarget, kGPUTarget, kUnknowTarget };
class Kernel2Ms {
 public:
  static Kernel2Ms &GetInstance();

  Kernel2Ms(const Kernel2Ms &) = delete;

  Kernel2Ms &operator=(const Kernel2Ms &) = delete;

  bool KernelGraph2MsGraph(const KernelGraphPtr &kernel_graph_ptr);

  bool KernelInput2MS(const std::vector<TensorPtr> &input_tensors);

  ConvertMode convert_mode() const { return convert_mode_; }

  void set_convert_mode(ConvertMode convert_mode) { convert_mode_ = convert_mode; }

  TargetMode device_target() const { return device_target_; }

  void set_device_target(TargetMode device_target) { device_target_ = device_target; }

  bool SaveDeviceModel(const std::shared_ptr<GraphDefT> &new_ms_graph_ptr, const std::string &save_path_name);

 private:
  Kernel2Ms() : graph_index_(0) {}

  void ReleaseContextRes();

  ~Kernel2Ms() = default;

  bool SetAllTensors(const TensorCachePtr &tensor_cache, SubGraphDefT *sub_graph_def_t);

  bool SetOpInputIdx(const CNodePtr &c_node_ptr, const TensorCachePtr &tensor_cache, NodeDef *ms_node);

  bool SetOpOutputIdx(const CNodePtr &c_node_ptr, const TensorPtr &output_tensor, const TensorCachePtr &tensor_cache,
                      int ref_count, size_t order_index, NodeDef *ms_node);

  bool SetGraphOutputIdx(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache,
                         SubGraphDefT *sub_graph_def_t, AllOutputTensors *all_output_tensors);

  void TransformGraphIndx();

  void GetRealInpoutsPtr(const AnfNodePtr &node, std::vector<AnfNodePtr> *real_inputs,
                         std::vector<size_t> *real_output_idx);

  bool InitGraphIndx(const KernelGraphPtr &kernel_graph_ptr);

  bool InitGraphInputsIndx(const KernelGraphPtr &kernel_graph_ptr);

  bool InitGraphValueNodesIndx(const KernelGraphPtr &kernel_graph_ptr);

  bool InitGraphOpsIndx(const KernelGraphPtr &kernel_graph_ptr);

  bool InitGraphOutputsIndx(const KernelGraphPtr &kernel_graph_ptr);

  bool SetGraphInputTensors(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache,
                            SubGraphDefT *sub_graph_def_t);

  bool SetGraphValueTensors(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache);

  bool SetGraphOpTensors(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache,
                         SubGraphDefT *sub_graph_def_t);
  std::vector<uint32_t> GetAllInputWeightIdxs() const { return input_weight_idxs_; }
  std::vector<uint32_t> GetAllInputIdxs() const { return all_input_idxs_; }

  bool CheckInputSizes(const std::vector<TensorPtr> &input_tensors, const std::vector<uint32_t> &all_input_idxs);

  bool SetMemResue() const;
  SubGraphPtr sub_ms_graph_;
  AllOutputTensors all_output_tensors_;
  std::vector<NodeDef *> tmp_op_nodes_;
  std::unordered_map<MsKernelKey, int> node_indexs_;
  std::unordered_map<int, MsKernelKey> index_nodes_;
  int graph_index_ = 0;
  TensorCachePtr tensor_cache_ptr_ = nullptr;
  ConvertMode convert_mode_ = kConvertCpuMode;
  TargetMode device_target_ = kCPUTarget;
  std::vector<uint32_t> input_weight_idxs_;
  std::vector<uint32_t> all_input_idxs_;
};
using Kernel2MsPtr = std::shared_ptr<Kernel2Ms>;
}  // namespace executor
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_KERNEL_TO_MS_H_
