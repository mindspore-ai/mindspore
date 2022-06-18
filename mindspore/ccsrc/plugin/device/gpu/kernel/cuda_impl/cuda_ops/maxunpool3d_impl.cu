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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxunpool3d_impl.cuh"
template <typename T>
__global__ void InitMaxUnpool3D(const int64_t outer_size, T *output) {
  T zero = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    output[pos] = zero;
  }
  return;
}
template <typename T, typename S>
__global__ void MaxUnpool3DNCDHW(const T *input, const S *indices, const int64_t inputChannel, const int64_t inputFlame,
                                 const int64_t inputHeight, const int64_t inputWidth, const int64_t outputChannel,
                                 const int64_t outputFlame, const int64_t outputHeight, const int64_t outputWidth,
                                 const int64_t thread_size, T *output) {
  int posn = blockIdx.z;
  int posc = blockIdx.y;
  output += (posn * inputChannel + posc) * outputFlame * outputHeight * outputWidth;
  input += (posn * inputChannel + posc) * inputFlame * inputHeight * inputWidth;
  indices += (posn * inputChannel + posc) * inputFlame * inputHeight * inputWidth;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < thread_size; pos += blockDim.x * gridDim.x) {
    S maxind = indices[pos];
    CUDA_KERNEL_ASSERT(maxind >= 0 && maxind < outputChannel * outputFlame * outputHeight * outputWidth);
    output[maxind] = input[pos];
  }

  return;
}

template <typename T, typename S>
__global__ void MaxUnpool3DNDHWC(const T *input, const S *indices, const int64_t inputFlame, const int64_t inputHeight,
                                 const int64_t inputWidth, const int64_t inputChannel, const int64_t outputFlame,
                                 const int64_t outputHeight, const int64_t outputWidth, const int64_t outputChannel,
                                 const int64_t thread_size, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < thread_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (inputHeight * inputWidth * inputFlame * inputChannel);
    const int posc = pos % inputChannel;
    S maxind = indices[pos];
    CUDA_KERNEL_ASSERT(maxind >= 0 && maxind < outputFlame * outputHeight * outputWidth * outputChannel);
    output[(posn * outputFlame * outputHeight * outputWidth + maxind) * outputChannel + posc] = input[pos];
  }

  return;
}

template <typename T, typename S>
void CalMaxUnpool3D(const T *input, const S *indices, const std::vector<int64_t> input_shape,
                    const std::vector<int64_t> output_shape, T *output, const int64_t outer_size,
                    const int64_t thread_size, const std::string data_format_, const uint32_t &device_id,
                    cudaStream_t cuda_stream) {
  InitMaxUnpool3D<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(outer_size, output);
  if (data_format_ == "NCDHW") {
    int outputPlaneSize = input_shape[2] * input_shape[3] * input_shape[4];
    dim3 grid((outputPlaneSize + 127) / 128, input_shape[1], input_shape[0]);
    dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);
    MaxUnpool3DNCDHW<<<grid, block, 0, cuda_stream>>>(
      input, indices, input_shape[1], input_shape[2], input_shape[3], input_shape[4], output_shape[1], output_shape[2],
      output_shape[3], output_shape[4], outputPlaneSize, output);
  } else {
    MaxUnpool3DNDHWC<<<CUDA_BLOCKS(device_id, thread_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, indices, input_shape[1], input_shape[2], input_shape[3], input_shape[4], output_shape[1], output_shape[2],
      output_shape[3], output_shape[4], thread_size, output);
  }

  return;
}

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint8_t, int32_t>(const uint8_t *input, const int32_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, uint8_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint8_t, int64_t>(const uint8_t *input, const int64_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, uint8_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint16_t, int32_t>(
  const uint16_t *input, const int32_t *indices, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, uint16_t *output, const int64_t outer_size, const int64_t thread_size,
  const std::string data_format_, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint16_t, int64_t>(
  const uint16_t *input, const int64_t *indices, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, uint16_t *output, const int64_t outer_size, const int64_t thread_size,
  const std::string data_format_, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint32_t, int32_t>(
  const uint32_t *input, const int32_t *indices, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, uint32_t *output, const int64_t outer_size, const int64_t thread_size,
  const std::string data_format_, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint32_t, int64_t>(
  const uint32_t *input, const int64_t *indices, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, uint32_t *output, const int64_t outer_size, const int64_t thread_size,
  const std::string data_format_, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint64_t, int32_t>(
  const uint64_t *input, const int32_t *indices, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, uint64_t *output, const int64_t outer_size, const int64_t thread_size,
  const std::string data_format_, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<uint64_t, int64_t>(
  const uint64_t *input, const int64_t *indices, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, uint64_t *output, const int64_t outer_size, const int64_t thread_size,
  const std::string data_format_, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int8_t, int32_t>(const int8_t *input, const int32_t *indices,
                                                              const std::vector<int64_t> input_shape,
                                                              const std::vector<int64_t> output_shape, int8_t *output,
                                                              const int64_t outer_size, const int64_t thread_size,
                                                              const std::string data_format_,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int8_t, int64_t>(const int8_t *input, const int64_t *indices,
                                                              const std::vector<int64_t> input_shape,
                                                              const std::vector<int64_t> output_shape, int8_t *output,
                                                              const int64_t outer_size, const int64_t thread_size,
                                                              const std::string data_format_,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int16_t, int32_t>(const int16_t *input, const int32_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, int16_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int16_t, int64_t>(const int16_t *input, const int64_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, int16_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int32_t, int32_t>(const int32_t *input, const int32_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, int32_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int32_t, int64_t>(const int32_t *input, const int64_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, int32_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int64_t, int32_t>(const int64_t *input, const int32_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, int64_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<int64_t, int64_t>(const int64_t *input, const int64_t *indices,
                                                               const std::vector<int64_t> input_shape,
                                                               const std::vector<int64_t> output_shape, int64_t *output,
                                                               const int64_t outer_size, const int64_t thread_size,
                                                               const std::string data_format_,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<half, int32_t>(const half *input, const int32_t *indices,
                                                            const std::vector<int64_t> input_shape,
                                                            const std::vector<int64_t> output_shape, half *output,
                                                            const int64_t outer_size, const int64_t thread_size,
                                                            const std::string data_format_,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<half, int64_t>(const half *input, const int64_t *indices,
                                                            const std::vector<int64_t> input_shape,
                                                            const std::vector<int64_t> output_shape, half *output,
                                                            const int64_t outer_size, const int64_t thread_size,
                                                            const std::string data_format_,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<float, int32_t>(const float *input, const int32_t *indices,
                                                             const std::vector<int64_t> input_shape,
                                                             const std::vector<int64_t> output_shape, float *output,
                                                             const int64_t outer_size, const int64_t thread_size,
                                                             const std::string data_format_,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<float, int64_t>(const float *input, const int64_t *indices,
                                                             const std::vector<int64_t> input_shape,
                                                             const std::vector<int64_t> output_shape, float *output,
                                                             const int64_t outer_size, const int64_t thread_size,
                                                             const std::string data_format_,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<double, int32_t>(const double *input, const int32_t *indices,
                                                              const std::vector<int64_t> input_shape,
                                                              const std::vector<int64_t> output_shape, double *output,
                                                              const int64_t outer_size, const int64_t thread_size,
                                                              const std::string data_format_,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxUnpool3D<double, int64_t>(const double *input, const int64_t *indices,
                                                              const std::vector<int64_t> input_shape,
                                                              const std::vector<int64_t> output_shape, double *output,
                                                              const int64_t outer_size, const int64_t thread_size,
                                                              const std::string data_format_,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
