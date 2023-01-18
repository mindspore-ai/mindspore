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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_SUBGRAPH_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_SUBGRAPH_H_
#include <utility>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "include/api/kernel.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_runtime.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_serializer.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/parameter_cache/embedding_cache_manager.h"
#include "include/api/context.h"
#include "common/config_infos.h"

namespace mindspore::lite {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
struct CacheTensorInfo {
  std::vector<mindspore::MSTensor> network_input_tensor_;
  bool front_op_can_cache_;
};

class TensorRTSubGraph {
 public:
  TensorRTSubGraph(std::vector<TensorRTOp *> ops, const std::vector<TensorInfo> &inputs,
                   const std::vector<TensorInfo> &outputs, const mindspore::Context *ctx,
                   std::shared_ptr<GPUDeviceInfo> device_info, TensorRTRuntime *runtime, bool support_resize,
                   bool support_hw_resize, const ProfileConfigs &trt_profile_config);
  ~TensorRTSubGraph();

  int Prepare();

  int Execute(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs);

  int Resize(const std::vector<tensor::Tensor> &inputs, const std::vector<ShapeVector> &new_shapes);

  int BuildTensorRTGraph();

  int Init(cudaStream_t stream, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle);

  void SetSerializePath(const std::string &path) { serialize_file_path_ = std::move(path); }

  std::vector<TensorInfo> &inputs() { return inputs_; }

  std::vector<TensorInfo> &outputs() { return outputs_; }

 private:
  int GetInputIndexByName(const std::string &name);
  int BuildEngine();

  int SetDeviceConfig(cudaStream_t stream, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle);

  bool IsInt8Mode();

  bool SupportFP16();

  nvinfer1::ITensor *SetTensorRTNetworkInput(const TensorInfo &in_tensor, int index);

  ITensorHelper FindTensorRTInputs(TensorRTOp *cur_op, const TensorInfo &in_tensor);

  int MarkOutputs();

  bool IsCached(TensorRTOp *cur_op, const TensorInfo &in_tensor);

  void FindCacheTensorInfo(TensorRTOp *cur_op, TensorInfo device_cache_tensor);

  bool CanOpCache(TensorRTOp *cur_op);

  int HandleCacheTensor(TensorRTOp *cur_op, const TensorInfo &in_tensor);

  nvinfer1::Dims ParseInputDimsProfile(const TensorInfo &in_tensor, int index);
  nvinfer1::Dims SetInputDimsProfile(const TensorInfo &in_tensor, int index);
  int ParseInputsProfile();

  int PreExecute(const std::vector<tensor::Tensor> &inputs, const std::vector<tensor::Tensor> &outputs,
                 bool sync = true);
  int PostExecute(std::vector<tensor::Tensor> *outputs, bool sync = true);

  int OnNewInputShapes(const std::vector<ShapeVector> &inputs);

  size_t MaxVolumnProfileIndex() const;
  int SelectProfile(const std::vector<ShapeVector> &new_shapes) const;
  int GetProfileBindingIndex(const std::string &name, size_t profile_index);
  bool ValidInputResizeDims(const nvinfer1::Dims &construct_dims, const std::vector<int64_t> &resize_input_shape);
  bool IsValidProfileDims() const;

  std::string name_;
  std::vector<TensorInfo> inputs_;
  std::vector<TensorInfo> outputs_;

  std::vector<TensorRTOp *> all_ops_{};
  // subgraph input nodes.
  std::vector<TensorRTOp *> in_ops_{};
  // subgraph output nodes.
  std::vector<TensorRTOp *> out_ops_{};

  void **tensor_bindings_{nullptr};

  std::shared_ptr<GPUDeviceInfo> device_info_{nullptr};

  TensorRTRuntime *runtime_{nullptr};  // all subgraph in one delegate share a runtime_

  std::set<std::string> trt_specific_weight_handled_inner_;

  // save in/out tensor name for subgraph isolate.
  std::vector<std::string> trt_in_tensor_name_;
  std::vector<std::string> trt_out_tensor_name_;

  nvinfer1::INetworkDefinition *network_{nullptr};
  nvinfer1::IBuilderConfig *config_{nullptr};
  nvinfer1::ICudaEngine *engine_{nullptr};
  nvinfer1::IExecutionContext *trt_context_{nullptr};

  TensorRTContext *ctx_;

  // -1 means don't support resize
  int input_batchsize_index_{0};
  int output_batchsize_index_{0};
  int input_hw_index_{0};

  std::map<std::string, std::vector<mindspore::MSTensor>> model_input_to_cache_tensors_;

  std::shared_ptr<TensorRTSerializer> serializer_{nullptr};

  std::string serialize_file_path_;
  cudaStream_t stream_{nullptr};

  std::vector<nvinfer1::IOptimizationProfile *> profiles_{};
  bool using_input_ranges_{false};
  ProfileConfigs trt_profile_config_;
  size_t profile_index_{0};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_SUBGRAPH_H_
