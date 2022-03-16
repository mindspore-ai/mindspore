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

#include "src/delegate/tensorrt/op/cast_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(CastPluginCreater);

int CastTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int CastTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ITensor *inputTensors[] = {tensorrt_in_tensors_[0].trt_tensor_};
  // cast to type tensor
  auto type_tensor = in_tensors_[1];
  if (type_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "unknown cast type of " << op_name_;
    return RET_ERROR;
  }
  if (type_tensor.DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(WARNING) << "unknown type_tensor data type of " << op_name_;
  }
  auto type_data = static_cast<const int *>(type_tensor.Data().get());
  DataType data_type = static_cast<DataType>(type_data[0]);
  MS_LOG(DEBUG) << op_name_ << " cast to data type(43 float): " << type_data[0];
  nvinfer1::DataType dest_datatype = ConvertDataType(data_type);
  auto plugin = std::make_shared<CastPlugin>(op_name_, tensorrt_in_tensors_[0].trt_tensor_->getType(), dest_datatype);
  nvinfer1::IPluginV2Layer *cast_layer = network->addPluginV2(inputTensors, 1, *plugin);
  if (cast_layer == nullptr) {
    MS_LOG(ERROR) << "create cast layer failed for: " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *cast_out = cast_layer->getOutput(0);
  cast_layer->setName(op_name_.c_str());
  cast_out->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(
    ITensorHelper{cast_out, tensorrt_in_tensors_[0].format_, tensorrt_in_tensors_[0].same_format_});
  this->layer_ = cast_layer;
  return RET_OK;
}

int CastPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                        const void *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) noexcept {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int element_cnt = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>());

  if (inputDesc->type == outputDesc->type) {
    int element_size = (outputDesc->type == nvinfer1::DataType::kFLOAT)
                         ? sizeof(float)
                         : ((outputDesc->type == nvinfer1::DataType::kINT32) ? sizeof(int) : 0);
    auto cuda_ret = cudaMemcpy(outputs[0], inputs[0], element_cnt * element_size, cudaMemcpyDeviceToDevice);
    if (cuda_ret != cudaSuccess) {
      MS_LOG(ERROR) << "copy mem failed for " << layer_name_;
      return RET_ERROR;
    }
    return RET_OK;
  }
  if (inputDesc->type == nvinfer1::DataType::kINT32 && dest_datatype_ == nvinfer1::DataType::kFLOAT) {
    auto input = static_cast<const int *>(inputs[0]);
    auto output = static_cast<float *>(outputs[0]);
    Cast(element_cnt, input, output, stream);
  } else if (inputDesc->type == nvinfer1::DataType::kFLOAT && dest_datatype_ == nvinfer1::DataType::kINT32) {
    auto input = static_cast<const float *>(inputs[0]);
    auto output = static_cast<int *>(outputs[0]);
    Cast(element_cnt, input, output, stream);
  } else {
    MS_LOG(ERROR) << "unsupported data type cast " << layer_name_;
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *CastPlugin::clone() const noexcept {
  auto *plugin = new CastPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DataType CastPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
  noexcept {
  return dest_datatype_;
}

size_t CastPlugin::getSerializationSize() const noexcept {
  // origin_datatype_ and dest_datatype_
  return sizeof(nvinfer1::DataType) * 2;
}

void CastPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &origin_datatype_, sizeof(nvinfer1::DataType));
  SerializeValue(&buffer, &dest_datatype_, sizeof(nvinfer1::DataType));
}

nvinfer1::IPluginV2 *CastPluginCreater::createPlugin(const char *name,
                                                     const nvinfer1::PluginFieldCollection *fc) noexcept {
  const nvinfer1::PluginField *fields = fc->fields;
  nvinfer1::DataType origin_datatype = static_cast<const nvinfer1::DataType *>(fields[0].data)[0];
  nvinfer1::DataType dest_datatype = static_cast<const nvinfer1::DataType *>(fields[1].data)[0];
  return new (std::nothrow) CastPlugin(name, origin_datatype, dest_datatype);
}

nvinfer1::IPluginV2 *CastPluginCreater::deserializePlugin(const char *name, const void *serialData,
                                                          size_t serialLength) noexcept {
  nvinfer1::DataType origin_datatype;
  DeserializeValue(&serialData, &serialLength, &origin_datatype, sizeof(nvinfer1::DataType));
  nvinfer1::DataType dest_datatype;
  DeserializeValue(&serialData, &serialLength, &dest_datatype, sizeof(nvinfer1::DataType));
  MS_LOG(DEBUG) << name << " is deserialize as origin_datatype: " << static_cast<int32_t>(origin_datatype)
                << ", dest_datatype: " << static_cast<int32_t>(dest_datatype);
  return new (std::nothrow) CastPlugin(name, origin_datatype, dest_datatype);
}
}  // namespace mindspore::lite
