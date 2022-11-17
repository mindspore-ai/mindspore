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

#include "src/extendrt/delegate/tensorrt/op/cast_plugin.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "src/extendrt/delegate/tensorrt/cuda_impl/cast.cuh"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(CastPluginCreater);
template class TensorRTPluginCreater<CastPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

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

nvinfer1::DataType CastPlugin::getOutputDataType(int, const nvinfer1::DataType *, int) const noexcept {
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
}  // namespace mindspore::lite
