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

#include "src/extendrt/delegate/tensorrt/op/bounding_box_decode_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/boundingbox_decode_impl.cuh"
#include "ops/bounding_box_decode.h"

namespace mindspore::lite {
int BoundingBoxDecodeTensorRT::IsSupport(const BaseOperatorPtr &base_operator,
                                         const std::vector<TensorInfo> &in_tensors,
                                         const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int BoundingBoxDecodeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto op = AsOps<ops::BoundingBoxDecode>();
  auto wh_ratio_clip_attr = op->GetAttr("wh_ratio_clip");
  float wh_ratio_clip = GetValue<float>(wh_ratio_clip_attr);

  auto max_shape_attr = op->GetAttr("max_shape");
  std::vector<int64_t> max_shape = GetValue<std::vector<int64_t>>(max_shape_attr);
  if (max_shape.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "max_shape size not equal 2 for " << op_name_;
  }
  std::vector<int> max_shape_32(INPUT_SIZE2);
  max_shape_32[0] = max_shape[0];
  max_shape_32[1] = max_shape[1];

  auto means_attr = op->GetAttr("means");
  std::vector<float> means = GetValue<std::vector<float>>(means_attr);
  auto stds_attr = op->GetAttr("stds");
  std::vector<float> stds = GetValue<std::vector<float>>(stds_attr);

  auto plugin = std::make_shared<BoundingBoxDecodePlugin>(op_name_, means, stds, max_shape_32, wh_ratio_clip);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create ActivationOptPlugin failed for " << op_name_;
    return RET_ERROR;
  }
  auto in_tensor1 = input(ctx, 0).trt_tensor_;
  auto in_tensor2 = input(ctx, 1).trt_tensor_;
  if (in_tensors_[0].DataType() == DataType::kNumberTypeFloat16) {
    in_tensor1 = TRTTensorCast(ctx, in_tensor1, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in_0");
  }
  if (in_tensors_[1].DataType() == DataType::kNumberTypeFloat16) {
    in_tensor2 = TRTTensorCast(ctx, in_tensor2, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in_1");
  }
  nvinfer1::ITensor *inputTensors[] = {in_tensor1, in_tensor2};
  nvinfer1::IPluginV2Layer *layer = ctx->network()->addPluginV2(inputTensors, INPUT_SIZE2, *plugin);
  this->layer_ = layer;
  nvinfer1::ITensor *op_out_tensor = layer->getOutput(0);
  if (op_out_tensor == nullptr) {
    MS_LOG(ERROR) << "addElementWise out tensor is nullptr.";
    return RET_ERROR;
  }
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(BoundingBoxDecodePluginCreater);
template class TensorRTPluginCreater<BoundingBoxDecodePlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int BoundingBoxDecodePlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                     const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                     void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaBoundingBoxDecode(inputDesc, inputs, outputs, stream);
}

int BoundingBoxDecodePlugin::RunCudaBoundingBoxDecode(const nvinfer1::PluginTensorDesc *inputDesc,
                                                      const void *const *inputs, void *const *outputs,
                                                      cudaStream_t stream) {
  BoundingBoxDecode<float>(GetDimsVolume(inputDesc[0].dims), static_cast<const float *>(inputs[0]),
                           static_cast<const float *>(inputs[1]), static_cast<float *>(outputs[0]), means_[0],
                           means_[1], means_[INPUT_SIZE2], means_[INPUT_SIZE3], stds_[0], stds_[1], stds_[INPUT_SIZE2],
                           stds_[INPUT_SIZE3], max_shape_[0], max_shape_[1], wh_ratio_clip_, device_id_, stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *BoundingBoxDecodePlugin::clone() const noexcept {
  auto *plugin = new BoundingBoxDecodePlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t BoundingBoxDecodePlugin::getSerializationSize() const noexcept {
  return sizeof(float) * (INPUT_SIZE4 + INPUT_SIZE4 + 1) + sizeof(int) * INPUT_SIZE2;
}

bool BoundingBoxDecodePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc,
                                                        int nbInputs, int nbOutputs) noexcept {
  return tensorsDesc[pos].type == nvinfer1::DataType::kFLOAT &&
         tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void BoundingBoxDecodePlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &means_[0], sizeof(float) * INPUT_SIZE4);
  SerializeValue(&buffer, &stds_[0], sizeof(float) * INPUT_SIZE4);
  SerializeValue(&buffer, &max_shape_[0], sizeof(int) * INPUT_SIZE2);
  SerializeValue(&buffer, &wh_ratio_clip_, sizeof(float));
}

REGISTER_TENSORRT_CREATOR(ops::kNameBoundingBoxDecode, BoundingBoxDecodeTensorRT)
}  // namespace mindspore::lite
