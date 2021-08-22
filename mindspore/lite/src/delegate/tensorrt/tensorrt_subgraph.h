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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_SUB_GTAPH_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_SUB_GTAPH_
#include <utility>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include "include/api/kernel.h"
#include "src/delegate/tensorrt/tensorrt_runtime.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "include/api/context.h"

namespace mindspore::lite {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
class TensorRTSubGraph : public kernel::Kernel {
 public:
  TensorRTSubGraph(std::vector<TensorRTOp *> ops, const std::vector<mindspore::MSTensor> &inputs,
                   const std::vector<mindspore::MSTensor> &outputs, const mindspore::Context *ctx,
                   std::shared_ptr<GPUDeviceInfo> device_info)
      : kernel::Kernel(inputs, outputs, nullptr, ctx), all_ops_(std::move(ops)), device_info_(device_info) {
    trt_specific_weight_nodes_ = {
      schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_ReduceFusion, schema::PrimitiveType_Transpose,
      schema::PrimitiveType_Gather,       schema::PrimitiveType_Reshape,      schema::PrimitiveType_PowFusion,
      schema::PrimitiveType_DivFusion,    schema::PrimitiveType_MatMul,       schema::PrimitiveType_ScaleFusion,
      schema::PrimitiveType_MulFusion,    schema::PrimitiveType_StridedSlice, schema::PrimitiveType_PadFusion};
  }

  ~TensorRTSubGraph() override;

  int Prepare() override;

  int Execute() override;

  int ReSize() override {
    MS_LOG(ERROR) << "TensorRT does not support the resize function temporarily.";
    return lite::RET_ERROR;
  }

  int BuildTensorRTGraph();

  int Init();

 private:
  int BuildEngine();

  int SetDeviceConfig();

  bool SupportFP16();

  static nvinfer1::ITensor *FindTensorRTInputs(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor);

  TensorRTRuntime *runtime_{nullptr};

  std::vector<TensorRTOp *> all_ops_{};
  // subgraph input nodes.
  std::vector<TensorRTOp *> in_ops_{};
  // subgraph output nodes.
  std::vector<TensorRTOp *> out_ops_{};

  void **tensor_bindings_{nullptr};
  std::shared_ptr<GPUDeviceInfo> device_info_{nullptr};

  std::set<mindspore::schema::PrimitiveType> trt_specific_weight_nodes_;

  // save in/out tensor name for subgraph isolate.
  std::vector<std::string> trt_in_tensor_name_;
  std::vector<std::string> trt_out_tensor_name_;

  nvinfer1::INetworkDefinition *network_{nullptr};
  nvinfer1::IBuilderConfig *config_{nullptr};
  nvinfer1::ICudaEngine *engine_{nullptr};
  nvinfer1::IExecutionContext *trt_context_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_SUB_GTAPH_
