/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/op/nmswithmask_tensorrt.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/nms_with_mask_impl.cuh"
#include "ops/nms_with_mask.h"

namespace mindspore::lite {
int NMSwithmaskTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                   const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int NMSwithmaskTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  auto in_tensor = input(ctx, 0).trt_tensor_;
  if (in_tensors_[0].DataType() == DataType::kNumberTypeFloat16) {
    in_tensor = TRTTensorCast(ctx, in_tensor, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in");
  }
  auto input1_dims = in_tensor->getDimensions();
  if (input1_dims.nbDims != INPUT_SIZE2 || input1_dims.d[1] != INPUT_SIZE5) {
    MS_LOG(ERROR) << "input tensor is invalid";
    return RET_ERROR;
  }
  auto num_input = input1_dims.d[0];
  auto nms_with_mask_op = AsOps<ops::BaseOperator>();
  if (nms_with_mask_op == nullptr) {
    MS_LOG(ERROR) << "Failed to as operator ConstantOfShape: " << op_name_;
    return RET_ERROR;
  }
  auto iou_value = GetValue<float>(nms_with_mask_op->GetAttr(kAttrIouThreshold));
  auto plugin = std::make_shared<NMSwithmaskPlugin>(op_name_, num_input, iou_value);
  CHECK_NULL_RETURN(plugin);
  nvinfer1::ITensor *inputTensors[] = {in_tensor};
  nvinfer1::IPluginV2Layer *nmswithmask_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  CHECK_NULL_RETURN(nmswithmask_layer);
  this->layer_ = nmswithmask_layer;
  for (int i = 0; i < INPUT_SIZE3; i++) {
    nvinfer1::ITensor *op_out_tensor = nmswithmask_layer->getOutput(i);
    CHECK_NULL_RETURN(op_out_tensor);
    ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                        out_tensors_[i].Name());
  }

  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(NMSwithmaskPluginCreater);
template class TensorRTPluginCreater<NMSwithmaskPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int NMSwithmaskPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                               const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                               void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaNMSwithmask(inputDesc, inputs, outputs, stream);
}

int NMSwithmaskPlugin::RunCudaNMSwithmask(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                          void *const *outputs, cudaStream_t stream) {
  //  T *output = outputs[0];
  //  int *sel_idx = outputs[1];
  //  int *sel_boxes = outputs[2];
  int box_size_ = INPUT_SIZE5;
  void *data_buff = nullptr;
  cudaMalloc(&data_buff, NmsRoundUpPower2(num_input_) * sizeof(float));
  void *index_buff = nullptr;
  cudaMalloc(&index_buff, NmsRoundUpPower2(num_input_) * sizeof(int));
  void *row_mask = nullptr;
  cudaMalloc(&row_mask, num_input_ * num_input_ * sizeof(bool));

  CalSort(static_cast<const int>(num_input_), static_cast<const float *>(inputs[0]), static_cast<float *>(outputs[0]),
          static_cast<int *>(index_buff), static_cast<float *>(data_buff), box_size_, device_id_, stream);
  CalPreprocess(static_cast<const int>(num_input_), static_cast<int *>(outputs[1]),
                static_cast<int *>(outputs[INPUT_SIZE2]), static_cast<const float *>(inputs[0]),
                static_cast<float *>(outputs[0]), static_cast<int *>(index_buff), box_size_,
                static_cast<bool *>(row_mask), device_id_, stream);
  CalNms(static_cast<const int>(num_input_), iou_value_, static_cast<float *>(outputs[0]),
         static_cast<int *>(outputs[INPUT_SIZE2]), box_size_, static_cast<bool *>(row_mask), device_id_, stream);
  cudaFree(data_buff);
  cudaFree(index_buff);
  cudaFree(row_mask);
  return RET_OK;
}
nvinfer1::DimsExprs NMSwithmaskPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                           int nbInputDims,
                                                           nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  if (index == 0) {
    dims = inputs[0];
  }
  if (index == 1) {
    dims.d[0] = inputs[0].d[0];
    dims.nbDims = 1;
  }
  if (index == INPUT_SIZE2) {
    dims.d[0] = inputs[0].d[0];
    dims.nbDims = 1;
  }
  return dims;
}

nvinfer1::DataType NMSwithmaskPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                        int nbInputs) const noexcept {
  nvinfer1::DataType datatype;
  if (index == 0) {
    datatype = nvinfer1::DataType::kFLOAT;
  }
  if (index == 1) {
    datatype = nvinfer1::DataType::kINT32;
  }
  if (index == INPUT_SIZE2) {
    datatype = nvinfer1::DataType::kINT32;
  }
  return datatype;
}

nvinfer1::IPluginV2DynamicExt *NMSwithmaskPlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) NMSwithmaskPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "malloc nms with mask plugin failed";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}
REGISTER_TENSORRT_CREATOR(ops::kNameNMSWithMask, NMSwithmaskTensorRT)
}  // namespace mindspore::lite
