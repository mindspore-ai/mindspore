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

#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxunpool2d_grad_impl.cuh"

template <typename T, typename S>
__global__ void MaxUnpool2DGradNCHW(const T *grad, const S *indices, const int64_t inputChannel,
                                    const int64_t inputHeight, const int64_t inputWidth, const int64_t outputChannel,
                                    const int64_t outputHeight, const int64_t outputWidth, const int64_t outer_size,
                                    T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (inputHeight * inputWidth * inputChannel);
    const int posc = pos / (inputWidth * inputHeight) % inputChannel;
    S maxind = indices[pos];
    output[pos] = grad[maxind + (posn * inputChannel + posc) * outputHeight * outputWidth];
  }
  return;
}

template <typename T, typename S>
__global__ void MaxUnpool2DGradNHWC(const T *grad, const S *indices, const int64_t inputHeight,
                                    const int64_t inputWidth, const int64_t inputChannel, const int64_t outputHeight,
                                    const int64_t outputWidth, const int64_t outputChannel, const int64_t outer_size,
                                    T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (inputHeight * inputWidth * inputChannel);
    const int posc = pos % inputChannel;
    S maxind = indices[pos];
    output[pos] = grad[(posn * outputHeight * outputWidth + maxind) * outputChannel + posc];
  }
  return;
}

template <typename T, typename S>
void CalMaxUnpool2DGrad(const T *grad, const S *indices, const std::vector<int64_t> backprop_input_shape,
                        const std::vector<int64_t> grad_shape, T *output, const int64_t outer_size,
                        const std::string data_format_, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  if (data_format_ == "NCHW") {
    MaxUnpool2DGradNCHW<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      grad, indices, backprop_input_shape[1], backprop_input_shape[2], backprop_input_shape[3], grad_shape[1],
      grad_shape[2], grad_shape[3], outer_size, output);
    return;
  } else {
    MaxUnpool2DGradNHWC<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      grad, indices, backprop_input_shape[1], backprop_input_shape[2], backprop_input_shape[3], grad_shape[1],
      grad_shape[2], grad_shape[3], outer_size, output);
  }
}

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint8_t, int32_t>(const uint8_t *grad, const int32_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   uint8_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint8_t, int64_t>(const uint8_t *grad, const int64_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   uint8_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint16_t, int32_t>(
  const uint16_t *grad, const int32_t *indices, const std::vector<int64_t> backprop_input_shape,
  const std::vector<int64_t> grad_shape, uint16_t *output, const int64_t outer_size, const std::string data_format_,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint16_t, int64_t>(
  const uint16_t *grad, const int64_t *indices, const std::vector<int64_t> backprop_input_shape,
  const std::vector<int64_t> grad_shape, uint16_t *output, const int64_t outer_size, const std::string data_format_,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint32_t, int32_t>(
  const uint32_t *grad, const int32_t *indices, const std::vector<int64_t> backprop_input_shape,
  const std::vector<int64_t> grad_shape, uint32_t *output, const int64_t outer_size, const std::string data_format_,
   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint32_t, int64_t>(
  const uint32_t *grad, const int64_t *indices, const std::vector<int64_t> backprop_input_shape,
  const std::vector<int64_t> grad_shape, uint32_t *output, const int64_t outer_size, const std::string data_format_,
   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint64_t, int32_t>(
  const uint64_t *grad, const int32_t *indices, const std::vector<int64_t> backprop_input_shape,
  const std::vector<int64_t> grad_shape, uint64_t *output, const int64_t outer_size, const std::string data_format_,
   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<uint64_t, int64_t>(
  const uint64_t *grad, const int64_t *indices, const std::vector<int64_t> backprop_input_shape,
  const std::vector<int64_t> grad_shape, uint64_t *output, const int64_t outer_size, const std::string data_format_,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int8_t, int32_t>(const int8_t *grad, const int32_t *indices,
                                                                  const std::vector<int64_t> backprop_input_shape,
                                                                  const std::vector<int64_t> grad_shape, int8_t *output,
                                                                  const int64_t outer_size,
                                                                  const std::string data_format_,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int8_t, int64_t>(const int8_t *grad, const int64_t *indices,
                                                                  const std::vector<int64_t> backprop_input_shape,
                                                                  const std::vector<int64_t> grad_shape, int8_t *output,
                                                                  const int64_t outer_size,
                                                                  const std::string data_format_,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int16_t, int32_t>(const int16_t *grad, const int32_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   int16_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int16_t, int64_t>(const int16_t *grad, const int64_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   int16_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int32_t, int32_t>(const int32_t *grad, const int32_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   int32_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int32_t, int64_t>(const int32_t *grad, const int64_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   int32_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int64_t, int32_t>(const int64_t *grad, const int32_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   int64_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<int64_t, int64_t>(const int64_t *grad, const int64_t *indices,
                                                                   const std::vector<int64_t> backprop_input_shape,
                                                                   const std::vector<int64_t> grad_shape,
                                                                   int64_t *output, const int64_t outer_size,
                                                                   const std::string data_format_,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<half, int32_t>(const half *grad, const int32_t *indices,
                                                                const std::vector<int64_t> backprop_input_shape,
                                                                const std::vector<int64_t> grad_shape, half *output,
                                                                const int64_t outer_size,
                                                                const std::string data_format_,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<half, int64_t>(const half *grad, const int64_t *indices,
                                                                const std::vector<int64_t> backprop_input_shape,
                                                                const std::vector<int64_t> grad_shape, half *output,
                                                                const int64_t outer_size,
                                                                const std::string data_format_,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<float, int32_t>(const float *grad, const int32_t *indices,
                                                                 const std::vector<int64_t> backprop_input_shape,
                                                                 const std::vector<int64_t> grad_shape, float *output,
                                                                 const int64_t outer_size,
                                                                 const std::string data_format_,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<float, int64_t>(const float *grad, const int64_t *indices,
                                                                 const std::vector<int64_t> backprop_input_shape,
                                                                 const std::vector<int64_t> grad_shape, float *output,
                                                                 const int64_t outer_size,
                                                                 const std::string data_format_,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<double, int32_t>(const double *grad, const int32_t *indices,
                                                                  const std::vector<int64_t> backprop_input_shape,
                                                                  const std::vector<int64_t> grad_shape, double *output,
                                                                  const int64_t outer_size,
                                                                  const std::string data_format_,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool2DGrad<double, int64_t>(const double *grad, const int64_t *indices,
                                                                  const std::vector<int64_t> backprop_input_shape,
                                                                  const std::vector<int64_t> grad_shape, double *output,
                                                                  const int64_t outer_size,
                                                                  const std::string data_format_,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
