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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_TENSORRT_TENSORRT_SUBGRAPH_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_TENSORRT_TENSORRT_SUBGRAPH_H_
#include <experimental/optional>
#include <utility>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "include/api/kernel.h"
#include "src/litert/delegate/tensorrt/tensorrt_runtime.h"
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"
#include "src/litert/delegate/tensorrt/tensorrt_serializer.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"
#include "src/litert/delegate/parameter_cache/embedding_cache_manager.h"
#include "include/api/context.h"

namespace mindspore::lite {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
struct CacheTensorInfo {
  std::vector<mindspore::MSTensor> network_input_tensor_;
  bool front_op_can_cache_;
};

class TensorRTSubGraph : public kernel::Kernel {
 public:
  TensorRTSubGraph(std::vector<TensorRTOp *> ops, const std::vector<mindspore::MSTensor> &inputs,
                   const std::vector<mindspore::MSTensor> &outputs, const mindspore::Context *ctx,
                   std::shared_ptr<GPUDeviceInfo> device_info, TensorRTRuntime *runtime, bool support_resize,
                   bool support_hw_resize, const std::unordered_map<std::string, std::vector<nvinfer1::Dims>> &min_dims,
                   const std::unordered_map<std::string, std::vector<nvinfer1::Dims>> &opt_dims,
                   const std::unordered_map<std::string, std::vector<nvinfer1::Dims>> &max_dims)
      : kernel::Kernel(inputs, outputs, nullptr, ctx),
        all_ops_(std::move(ops)),
        device_info_(device_info),
        runtime_(runtime),
        min_dims_(min_dims),
        opt_dims_(opt_dims),
        max_dims_(max_dims) {
    trt_specific_weight_nodes_ = {
      schema::PrimitiveType_Conv2DFusion,   schema::PrimitiveType_ReduceFusion, schema::PrimitiveType_Transpose,
      schema::PrimitiveType_Gather,         schema::PrimitiveType_Reshape,      schema::PrimitiveType_MatMulFusion,
      schema::PrimitiveType_ScaleFusion,    schema::PrimitiveType_PadFusion,    schema::PrimitiveType_BroadcastTo,
      schema::PrimitiveType_FullConnection, schema::PrimitiveType_Cast,         schema::PrimitiveType_ExpandDims,
      schema::PrimitiveType_Resize,         schema::PrimitiveType_LSTM,         schema::PrimitiveType_LayerNormFusion,
      schema::PrimitiveType_TopKFusion,     schema::PrimitiveType_TileFusion,   schema::PrimitiveType_QuantDTypeCast,
    };
    if (!support_resize) {
      input_batchsize_index_ = -1;
      input_hw_index_ = -1;
    }
    if (!support_hw_resize) {
      input_hw_index_ = -1;
    }
  }

  ~TensorRTSubGraph() override;

  int Prepare() override;

  int Execute() override;

  int ReSize();

  int BuildTensorRTGraph();

  int Init(cudaStream_t stream);

  void SetCacheManager(const std::shared_ptr<cache::EmbeddingCacheManager> &cache_mgr) { cache_mgr_ = cache_mgr; }

  void SetSerializePath(const std::string &path) { serialize_file_path_ = std::move(path); }

 private:
  int GetInputIndexByName(const std::string &name);

  int BuildEngine();

  int SetDeviceConfig(cudaStream_t stream);

  bool IsInt8Mode();

  bool SupportFP16();

  nvinfer1::ITensor *SetTensorRTNetworkInput(const mindspore::MSTensor &in_tensor, size_t index);

  ITensorHelper FindTensorRTInputs(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor);

  int MarkOutputs();

  bool FixSizeOutputNeedTranspose(ITensorHelper output_helper, const mindspore::MSTensor &out_tensor);
  bool DynamicSizeOutputNeedTranspose(ITensorHelper output_helper, const mindspore::MSTensor &out_tensor);

  bool IsCached(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor);

  void FindCacheTensorInfo(TensorRTOp *cur_op, mindspore::MSTensor device_cache_tensor);

  bool CanOpCache(TensorRTOp *cur_op);

  int HandleCacheTensor(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor);

  nvinfer1::Dims ParseInputDimsProfile(const mindspore::MSTensor &in_tensor);
  nvinfer1::Dims SetInputDimsProfile(const mindspore::MSTensor &in_tensor);
  int ParseInputsProfile();

  size_t MaxVolumnProfileIndex() const;
  std::experimental::optional<int> SelectProfile() const;

  bool ValidInputResizeDims(const nvinfer1::Dims &construct_dims, const std::vector<int64_t> &resize_input_shape);
  std::experimental::optional<bool> IsValidProfileDims() const;

  std::vector<TensorRTOp *> all_ops_{};
  // subgraph input nodes.
  std::vector<TensorRTOp *> in_ops_{};
  // subgraph output nodes.
  std::vector<TensorRTOp *> out_ops_{};

  void **tensor_bindings_{nullptr};

  std::shared_ptr<GPUDeviceInfo> device_info_{nullptr};

  TensorRTRuntime *runtime_{nullptr};  // all subgraph in one delegate share a runtime_

  std::set<mindspore::schema::PrimitiveType> trt_specific_weight_nodes_;

  // save in/out tensor name for subgraph isolate.
  std::vector<std::string> trt_in_tensor_name_;
  std::vector<std::string> trt_out_tensor_name_;

  std::vector<mindspore::MSTensor> cache_const_inputs_;
  std::map<std::string, CacheTensorInfo> network_cache_tensor_info_;

  nvinfer1::INetworkDefinition *network_{nullptr};
  nvinfer1::IBuilderConfig *config_{nullptr};
  nvinfer1::ICudaEngine *engine_{nullptr};
  nvinfer1::IExecutionContext *trt_context_{nullptr};
  std::vector<nvinfer1::IOptimizationProfile *> profiles_{};

  TensorRTContext *ctx_;

  // -1 means don't support resize
  int input_batchsize_index_{0};
  int output_batchsize_index_{0};
  int input_hw_index_{0};
  bool using_input_ranges_{false};

  std::map<std::string, std::vector<mindspore::MSTensor>> model_input_to_cache_tensors_;

  std::shared_ptr<cache::EmbeddingCacheManager> cache_mgr_{nullptr};

  std::shared_ptr<TensorRTSerializer> serializer_{nullptr};

  std::string serialize_file_path_;
  cudaStream_t stream_{nullptr};

  std::unordered_map<std::string, std::vector<nvinfer1::Dims>> min_dims_;
  std::unordered_map<std::string, std::vector<nvinfer1::Dims>> opt_dims_;
  std::unordered_map<std::string, std::vector<nvinfer1::Dims>> max_dims_;
  size_t profile_index_{0};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_TENSORRT_TENSORRT_SUBGRAPH_H_
