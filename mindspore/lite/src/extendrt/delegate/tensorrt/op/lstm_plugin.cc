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

#include "src/extendrt/delegate/tensorrt/op/lstm_plugin.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/swish_impl.cuh"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(LSTMPluginCreater);
template class TensorRTPluginCreater<LSTMPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int LSTMPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                        const void *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) noexcept {
  cudnnRNNDataDescriptor_t xdesc;
  cudnnRNNDataDescriptor_t ydesc;

  cudnnTensorDescriptor_t hdesc;
  cudnnTensorDescriptor_t cdesc;

  cudnnRNNDataLayout_t layout{CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED};

  int *seq_length_array;
  int *dev_seq_lenghts;
  size_t weight_space_size;
  size_t reserve_space_size{0};
  size_t workspace_size;
  void *d_workspace;
  void *d_reserve_space{nullptr};

  // lstm type
  cudnnRNNMode_t cell_mode{CUDNN_LSTM};
  cudnnRNNBiasMode_t bias_mode{CUDNN_RNN_DOUBLE_BIAS};
  cudnnDirectionMode_t direction_mode{CUDNN_UNIDIRECTIONAL};
  cudnnRNNInputMode_t input_mode{CUDNN_LINEAR_INPUT};
  cudnnForwardMode_t fwd_mode{CUDNN_FWD_MODE_INFERENCE};

  cudnnRNNDescriptor_t rnn_desc;
  cudnnDropoutDescriptor_t dropout_desc;

  cudnnHandle_t cudnn_handle;
  cudnnDataType_t data_type{CUDNN_DATA_FLOAT};
  cudnnDataType_t math_precison{CUDNN_DATA_FLOAT};
  cudnnMathType_t math_type{CUDNN_DEFAULT_MATH};
  cudnnRNNAlgo_t rnn_algo{CUDNN_RNN_ALGO_STANDARD};
  CUDNN_CHECK(cudnnCreate(&cudnn_handle));
  CUDNN_CHECK(cudnnSetStream(cudnn_handle, reinterpret_cast<cudaStream_t>(stream)));
  CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&xdesc));
  CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&ydesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hdesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cdesc));
  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));
  CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc));
  seq_length_array = new int[batch_size_];
  for (int i = 0; i < batch_size_; ++i) {
    seq_length_array[i] = seq_len_;
  }
  cudaMalloc(&dev_seq_lenghts, sizeof(int) * batch_size_);
  cudaMemcpy(dev_seq_lenghts, seq_length_array, sizeof(int) * batch_size_, cudaMemcpyHostToDevice);

  CUDNN_CHECK(
    cudnnSetRNNDataDescriptor(xdesc, data_type, layout, seq_len_, batch_size_, input_size_, seq_length_array, nullptr));
  CUDNN_CHECK(cudnnSetRNNDataDescriptor(ydesc, data_type, layout, seq_len_, batch_size_, hidden_size_, seq_length_array,
                                        nullptr));
  constexpr int kDims = 3;
  int dim[kDims];
  int stride[kDims];

  dim[0] = num_layers_;
  dim[1] = batch_size_;
  dim[INPUT_SIZE2] = hidden_size_;

  stride[0] = dim[INPUT_SIZE2] * dim[1];
  stride[1] = dim[INPUT_SIZE2];
  stride[INPUT_SIZE2] = 1;

  CUDNN_CHECK(cudnnSetTensorNdDescriptor(hdesc, data_type, kDims, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(cdesc, data_type, kDims, dim, stride));

  CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, 0, nullptr, 0, 1));

  CUDNN_CHECK(cudnnSetRNNDescriptor_v8(rnn_desc, rnn_algo, cell_mode, bias_mode, direction_mode, input_mode, data_type,
                                       math_precison, math_type, input_size_, hidden_size_, hidden_size_, num_layers_,
                                       dropout_desc, 0));

  // Set up weights and bias parameters
  CUDNN_CHECK(cudnnGetRNNWeightSpaceSize(cudnn_handle, rnn_desc, &weight_space_size));

  // Set up work space and reserved memory
  CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(cudnn_handle, rnn_desc, fwd_mode, xdesc, &workspace_size, &reserve_space_size));

  if (workspace_size > 0) {
    cudaMalloc(reinterpret_cast<void **>(&d_workspace), workspace_size);
  }
  if (reserve_space_size > 0) {
    cudaMalloc(reinterpret_cast<void **>(&d_reserve_space), reserve_space_size);
  }
  auto x_addr = static_cast<const float *>(inputs[0]);
  auto hx_addr = static_cast<const float *>(inputs[1]);
  auto cx_addr = static_cast<const float *>(inputs[INPUT_SIZE2]);
  auto w_addr = static_cast<const float *>(inputs[INPUT_SIZE3]);

  auto y_addr = static_cast<float *>(outputs[0]);

  CUDNN_CHECK(cudnnRNNForward(cudnn_handle, rnn_desc, fwd_mode, dev_seq_lenghts, xdesc, x_addr, ydesc, y_addr, hdesc,
                              hx_addr, nullptr, cdesc, cx_addr, nullptr, weight_space_size, w_addr, workspace_size,
                              d_workspace, reserve_space_size, d_workspace));
  return RET_OK;
}

nvinfer1::DimsExprs LSTMPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                                    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  if (index == 0) {
    dims = inputs[0];
    dims.d[INPUT_SIZE2] = exprBuilder.constant(hidden_size_);
  }
  return dims;
}

nvinfer1::DataType LSTMPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
  noexcept {
  return nvinfer1::DataType::kFLOAT;
}

nvinfer1::IPluginV2DynamicExt *LSTMPlugin::clone() const noexcept {
  auto *plugin = new LSTMPlugin(layer_name_, num_layers_, batch_size_, seq_len_, input_size_, hidden_size_, dropout_,
                                has_bias_, bidirectional_, device_id_);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t LSTMPlugin::getSerializationSize() const noexcept {
  return sizeof(int) * INPUT_SIZE5 + sizeof(float) + sizeof(bool) * INPUT_SIZE2;
}

void LSTMPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &num_layers_, sizeof(int));
  SerializeValue(&buffer, &batch_size_, sizeof(int));
  SerializeValue(&buffer, &seq_len_, sizeof(int));
  SerializeValue(&buffer, &input_size_, sizeof(int));
  SerializeValue(&buffer, &hidden_size_, sizeof(int));
  SerializeValue(&buffer, &dropout_, sizeof(float));
  SerializeValue(&buffer, &has_bias_, sizeof(bool));
  SerializeValue(&buffer, &bidirectional_, sizeof(bool));
}
}  // namespace mindspore::lite
