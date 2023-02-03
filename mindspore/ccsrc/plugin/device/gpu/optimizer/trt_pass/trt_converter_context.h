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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_CONVERTER_CONTEXT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_CONVERTER_CONTEXT_H_

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <map>
#include <NvInfer.h>
#include "utils/hash_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/optimizer/trt_pass/layer_input.h"

namespace mindspore {
namespace opt {
// The const number 4GB in bytes.
constexpr size_t kFourGBytes = 4UL << 30;

// Class transform ANF graph to Tensor-RT network.
// It converts the operators in ANF graph to Tensor-RT layer according to the topological order.
// During the conversion, the cache keep the map between ANF node outputs and Tensor-RT layer outputs.
//  Before starting the operator conversion, it first caches the weights and constant node int the Anf graph.
//  During performing operator transformation, it obtains the inputs of the operator from the cache.
//  After conversion is completed, it store the outputs of the operator to the cache.
class TrtConverterContext : public std::enable_shared_from_this<TrtConverterContext> {
 public:
  explicit TrtConverterContext(FuncGraphPtr fg)
      : func_graph_(fg),
        batch_size_(1),
        workspace_size_(kFourGBytes),
        builder_(nullptr),
        network_(nullptr),
        config_(nullptr),
        engine_(nullptr) {}
  ~TrtConverterContext() = default;

  // Create Tensor-RT object and cache the ANF graph inputs and constant node.
  bool Init();

  // Parser KernelGraph to trt graph
  bool Parser();

  // Serialize trt models.
  bool Serialize(std::string *model);

  // Get trt graph inputs without weights. The inputs keep same order as binding name.
  std::vector<AnfNodePtr> GetGraphInputs() const;

  // Get trt graph outputs. All outputs are flatten to vector with concret shape.
  std::tuple<std::map<size_t, size_t>, std::vector<session::KernelWithIndex>> GetGraphOutputs() const;

  // Store trt layer outputs to the cache.
  bool StoreLayerOutput(const AnfNodePtr &node, const std::vector<nvinfer1::ITensor *> &inputs);

  // Get trt layer inputs from the cache.
  bool LoadLayerInput(const AnfNodePtr &node, std::vector<LayerInput> *inputs);

  // Create and keep temporary weight, as constant folding demanding new weight excluded in graph,
  // which should release until building finish.
  std::shared_ptr<tensor::Tensor> CreateTempWeight(const TypeId &type, const ShapeVector &shape);

  std::shared_ptr<nvinfer1::INetworkDefinition> network() const { return network_; }

 private:
  bool InitInputTable();
  bool InitValueNodeTable();
  LayerInput *LoadInputOnDemand(const AnfNodePtr &node);

  FuncGraphPtr func_graph_;
  uint32_t batch_size_;
  size_t workspace_size_;
  std::shared_ptr<nvinfer1::IBuilder> builder_;
  std::shared_ptr<nvinfer1::INetworkDefinition> network_;
  std::shared_ptr<nvinfer1::IBuilderConfig> config_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;

  // Cache (AnfNode + output_index : ILayer output).
  mindspore::HashMap<AnfNodePtr, mindspore::HashMap<size_t, LayerInput>> output_map_;
  std::vector<std::shared_ptr<tensor::Tensor>> temp_weights_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_CONVERTER_HELPER_H_
