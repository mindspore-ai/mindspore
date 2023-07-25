/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/convolution/convolution_ops_impl.cuh"
#include <iostream>
#include <algorithm>
#include "include/cuda_fp16.h"

size_t kThreadPerBlock = 1024;
#define DIVIDE_CEIL(a, b) a / b + ((a / b * b) < a)

template <typename T>
__global__ void __launch_bounds__(1024)
  Conv2dDepthWiseForwardNCHWKernel(ConvolutionCudaArgs cuda_args, const T *input, const T *filter, T *output) {
  const int in_channel = cuda_args.in_channel;
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / out_height / out_width;
    const int c = (pos / out_height / out_width) % in_channel;
    const int h = (pos / out_width) % out_height;
    const int w = pos % out_width;
    const T *filter_data = filter + c * filter_height * filter_width;
    T accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_in = -pad_height + h * stride_height + fh * dilation_height;
        const int w_in = -pad_width + w * stride_width + fw * dilation_width;
        if ((h_in >= 0) && (h_in < in_height) && (w_in >= 0) && (w_in < in_width)) {
          const int offset = ((n * in_channel + c) * in_height + h_in) * in_width + w_in;
          accumulation += (*filter_data) * input[offset];
        }
        ++filter_data;
      }
    }
    output[pos] = accumulation;
  }
}

template <>
__global__ void __launch_bounds__(1024) Conv2dDepthWiseForwardNCHWKernel<half>(ConvolutionCudaArgs cuda_args,
                                                                               const half *input, const half *filter,
                                                                               half *output) {
  const int in_channel = cuda_args.in_channel;
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / out_height / out_width;
    const int c = (pos / out_height / out_width) % in_channel;
    const int h = (pos / out_width) % out_height;
    const int w = pos % out_width;
    const half *filter_data = filter + c * filter_height * filter_width;
    float accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_in = -pad_height + h * stride_height + fh * dilation_height;
        const int w_in = -pad_width + w * stride_width + fw * dilation_width;
        if ((h_in >= 0) && (h_in < in_height) && (w_in >= 0) && (w_in < in_width)) {
          const int offset = ((n * in_channel + c) * in_height + h_in) * in_width + w_in;
          accumulation += __half2float(*filter_data) * __half2float(input[offset]);
        }
        ++filter_data;
      }
    }
    output[pos] = __float2half(accumulation);
  }
}

template <typename T>
__global__ void __launch_bounds__(1024)
  Conv2dDepthWiseForwardNHWCKernel(ConvolutionCudaArgs cuda_args, const T *input, const T *filter, T *output) {
  const int in_channel = cuda_args.in_channel;
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / out_height / out_width;
    const int h = (pos / in_channel / out_width) % out_height;
    const int w = (pos / in_channel) % out_width;
    const int c = pos % in_channel;
    const T *filter_data = filter + c * filter_height * filter_width;
    T accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_in = -pad_height + h * stride_height + fh * dilation_height;
        const int w_in = -pad_width + w * stride_width + fw * dilation_width;
        if ((h_in >= 0) && (h_in < in_height) && (w_in >= 0) && (w_in < in_width)) {
          const int offset = ((n * in_height + h_in) * in_width + w_in) * in_channel + c;
          accumulation += (*filter_data) * input[offset];
        }
        ++filter_data;
      }
    }
    output[pos] = accumulation;
  }
}

template <>
__global__ void __launch_bounds__(1024) Conv2dDepthWiseForwardNHWCKernel<half>(ConvolutionCudaArgs cuda_args,
                                                                               const half *input, const half *filter,
                                                                               half *output) {
  const int in_channel = cuda_args.in_channel;
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / out_height / out_width;
    const int h = (pos / in_channel / out_width) % out_height;
    const int w = (pos / in_channel) % out_width;
    const int c = pos % in_channel;
    const half *filter_data = filter + c * filter_height * filter_width;
    float accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_in = -pad_height + h * stride_height + fh * dilation_height;
        const int w_in = -pad_width + w * stride_width + fw * dilation_width;
        if ((h_in >= 0) && (h_in < in_height) && (w_in >= 0) && (w_in < in_width)) {
          const int offset = ((n * in_height + h_in) * in_width + w_in) * in_channel + c;
          accumulation += __half2float(*filter_data) * __half2float(input[offset]);
        }
        ++filter_data;
      }
    }
    output[pos] = __float2half(accumulation);
  }
}

template <typename T>
__global__ void __launch_bounds__(1024)
  Conv2dDepthWiseInputGradNCHWKernel(ConvolutionCudaArgs cuda_args, const T *dy, const T *filter, T *output) {
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int in_channel = cuda_args.in_channel;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / in_height / in_width;
    const int c = (pos / in_height / in_width) % in_channel;
    const int h = (pos / in_width) % in_height;
    const int w = pos % in_width;
    const T *filter_data = filter + c * filter_height * filter_width;
    T accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_out_s = h + pad_height - fh * dilation_height;
        const int w_out_s = w + pad_width - fw * dilation_width;
        if (((h_out_s % stride_height) == 0) && ((w_out_s % stride_width) == 0)) {
          const int h_out = h_out_s / stride_height;
          const int w_out = w_out_s / stride_width;
          if ((h_out >= 0) && (h_out < out_height) && (w_out >= 0) && (w_out < out_width)) {
            const int offset = ((n * in_channel + c) * out_height + h_out) * out_width + w_out;
            accumulation += (*filter_data) * dy[offset];
          }
        }
        ++filter_data;
      }
    }
    output[pos] = accumulation;
  }
}

template <>
__global__ void __launch_bounds__(1024) Conv2dDepthWiseInputGradNCHWKernel<half>(ConvolutionCudaArgs cuda_args,
                                                                                 const half *dy, const half *filter,
                                                                                 half *output) {
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int in_channel = cuda_args.in_channel;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / in_height / in_width;
    const int c = (pos / in_height / in_width) % in_channel;
    const int h = (pos / in_width) % in_height;
    const int w = pos % in_width;
    const half *filter_data = filter + c * filter_height * filter_width;
    float accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_out_s = h + pad_height - fh * dilation_height;
        const int w_out_s = w + pad_width - fw * dilation_width;
        if (((h_out_s % stride_height) == 0) && ((w_out_s % stride_width) == 0)) {
          const int h_out = h_out_s / stride_height;
          const int w_out = w_out_s / stride_width;
          if ((h_out >= 0) && (h_out < out_height) && (w_out >= 0) && (w_out < out_width)) {
            const int offset = ((n * in_channel + c) * out_height + h_out) * out_width + w_out;
            accumulation += __half2float(*filter_data) * __half2float(dy[offset]);
          }
        }
        ++filter_data;
      }
    }
    output[pos] = __float2half(accumulation);
  }
}

template <typename T>
__global__ void __launch_bounds__(1024)
  Conv2dDepthWiseInputGradNHWCKernel(ConvolutionCudaArgs cuda_args, const T *dy, const T *filter, T *output) {
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int in_channel = cuda_args.in_channel;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / in_height / in_width;
    const int h = (pos / in_channel / in_width) % in_height;
    const int w = (pos / in_channel) % in_width;
    const int c = pos % in_channel;
    const T *filter_data = filter + c * filter_height * filter_width;
    T accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_out_s = h + pad_height - fh * dilation_height;
        const int w_out_s = w + pad_width - fw * dilation_width;
        if (((h_out_s % stride_height) == 0) && ((w_out_s % stride_width) == 0)) {
          const int h_out = h_out_s / stride_height;
          const int w_out = w_out_s / stride_width;
          if ((h_out >= 0) && (h_out < out_height) && (w_out >= 0) && (w_out < out_width)) {
            const int offset = ((n * out_height + h_out) * out_width + w_out) * in_channel + c;
            accumulation += (*filter_data) * dy[offset];
          }
        }
        ++filter_data;
      }
    }
    output[pos] = accumulation;
  }
}

template <>
__global__ void __launch_bounds__(1024) Conv2dDepthWiseInputGradNHWCKernel<half>(ConvolutionCudaArgs cuda_args,
                                                                                 const half *dy, const half *filter,
                                                                                 half *output) {
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int in_channel = cuda_args.in_channel;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int dilation_height = cuda_args.dilation_height;
  const int dilation_width = cuda_args.dilation_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int n = pos / in_channel / in_height / in_width;
    const int h = (pos / in_channel / in_width) % in_height;
    const int w = (pos / in_channel) % in_width;
    const int c = pos % in_channel;
    const half *filter_data = filter + c * filter_height * filter_width;
    float accumulation = 0;
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        const int h_out_s = h + pad_height - fh * dilation_height;
        const int w_out_s = w + pad_width - fw * dilation_width;
        if (((h_out_s % stride_height) == 0) && ((w_out_s % stride_width) == 0)) {
          const int h_out = h_out_s / stride_height;
          const int w_out = w_out_s / stride_width;
          if ((h_out >= 0) && (h_out < out_height) && (w_out >= 0) && (w_out < out_width)) {
            const int offset = ((n * out_height + h_out) * out_width + w_out) * in_channel + c;
            accumulation += __half2float(*filter_data) * __half2float(dy[offset]);
          }
        }
        ++filter_data;
      }
    }
    output[pos] = __float2half(accumulation);
  }
}

template <typename T>
__global__ void __launch_bounds__(1024)
  Conv2dDepthWiseFilterGradNCHWKernel(ConvolutionCudaArgs cuda_args, const T *dy, const T *input_x, T *filter_diff) {
  const int in_channel = cuda_args.in_channel;
  const int batch_size = cuda_args.batch_size;
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int kw = pos % filter_width;
    const int kh = (pos / filter_width) % filter_height;
    const int c = pos / filter_width / filter_height;
    T gradient = 0;
    for (int n = 0; n < batch_size; n++) {
      const T *const dy_slice = dy + (n * in_channel + c) * out_height * out_width;
      const T *const input_x_slice = input_x + (n * in_channel + c) * in_height * in_width;

      const int ph_start = max(DIVIDE_CEIL((pad_height - kh), stride_height), 0);
      const int ph_end = min(DIVIDE_CEIL((in_height + pad_height - kh), stride_height), out_height);
      const int pw_start = max(DIVIDE_CEIL((pad_width - kw), stride_width), 0);
      const int pw_end = min(DIVIDE_CEIL((in_width + pad_width - kw), stride_width), out_width);
      for (int ph = ph_start; ph < ph_end; ph++) {
        for (int pw = pw_start; pw < pw_end; pw++) {
          const int h = ph * stride_height + kh - pad_height;
          const int w = pw * stride_width + kw - pad_width;
          gradient += dy_slice[ph * out_width + pw] * input_x_slice[h * in_width + w];
        }
      }
    }
    filter_diff[c * filter_height * filter_width + kh * filter_width + kw] = gradient;
  }
}

template <>
__global__ void __launch_bounds__(1024) Conv2dDepthWiseFilterGradNCHWKernel<half>(ConvolutionCudaArgs cuda_args,
                                                                                  const half *dy, const half *input_x,
                                                                                  half *filter_diff) {
  const int in_channel = cuda_args.in_channel;
  const int batch_size = cuda_args.batch_size;
  const int in_height = cuda_args.in_height;
  const int in_width = cuda_args.in_width;
  const int filter_height = cuda_args.filter_height;
  const int filter_width = cuda_args.filter_width;
  const int stride_height = cuda_args.stride_height;
  const int stride_width = cuda_args.stride_width;
  const int pad_height = cuda_args.pad_height;
  const int pad_width = cuda_args.pad_width;
  const int out_height = cuda_args.out_height;
  const int out_width = cuda_args.out_width;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < cuda_args.output_size; pos += blockDim.x * gridDim.x) {
    const int kw = pos % filter_width;
    const int kh = (pos / filter_width) % filter_height;
    const int c = pos / filter_width / filter_height;
    float gradient = 0;
    for (int n = 0; n < batch_size; n++) {
      const half *const dy_slice = dy + (n * in_channel + c) * out_height * out_width;
      const half *const input_x_slice = input_x + (n * in_channel + c) * in_height * in_width;

      const int ph_start = max(DIVIDE_CEIL((pad_height - kh), stride_height), 0);
      const int ph_end = min(DIVIDE_CEIL((in_height + pad_height - kh), stride_height), out_height);
      const int pw_start = max(DIVIDE_CEIL((pad_width - kw), stride_width), 0);
      const int pw_end = min(DIVIDE_CEIL((in_width + pad_width - kw), stride_width), out_width);
      for (int ph = ph_start; ph < ph_end; ph++) {
        for (int pw = pw_start; pw < pw_end; pw++) {
          const int h = ph * stride_height + kh - pad_height;
          const int w = pw * stride_width + kw - pad_width;
          gradient += __half2float(dy_slice[ph * out_width + pw]) * __half2float(input_x_slice[h * in_width + w]);
        }
      }
    }
    filter_diff[c * filter_height * filter_width + kh * filter_width + kw] = __float2half(gradient);
  }
}

template <typename T>
struct ConvolutionFunc<ConvolutionOpType::kConv2dDepthWiseForwardNCHW, T> {
  __host__ __forceinline__ ConvolutionFunc() {}
  __host__ __forceinline__ cudaError_t operator()(const ConvolutionCudaArgs &cuda_args, const T *input0_addr,
                                                  const T *input1_addr, T *output_addr,
                                                  cudaStream_t cuda_stream) const {
    size_t output_size = cuda_args.output_size;
    size_t thread_num = output_size > kThreadPerBlock ? kThreadPerBlock : output_size;
    Conv2dDepthWiseForwardNCHWKernel<<<CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, output_size, thread_num), thread_num, 0,
                                       cuda_stream>>>(cuda_args, input0_addr, input1_addr, output_addr);
    return GetCudaStatus();
  }
};

template <typename T>
struct ConvolutionFunc<ConvolutionOpType::kConv2dDepthWiseForwardNHWC, T> {
  __host__ __forceinline__ ConvolutionFunc() {}
  __host__ __forceinline__ cudaError_t operator()(const ConvolutionCudaArgs &cuda_args, const T *input0_addr,
                                                  const T *input1_addr, T *output_addr,
                                                  cudaStream_t cuda_stream) const {
    size_t output_size = cuda_args.output_size;
    size_t thread_num = output_size > kThreadPerBlock ? kThreadPerBlock : output_size;
    Conv2dDepthWiseForwardNHWCKernel<<<CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, output_size, thread_num), thread_num, 0,
                                       cuda_stream>>>(cuda_args, input0_addr, input1_addr, output_addr);
    return GetCudaStatus();
  }
};

template <typename T>
struct ConvolutionFunc<ConvolutionOpType::kConv2dDepthWiseInputGradNCHW, T> {
  __host__ __forceinline__ ConvolutionFunc() {}
  __host__ __forceinline__ cudaError_t operator()(const ConvolutionCudaArgs &cuda_args, const T *input0_addr,
                                                  const T *input1_addr, T *output_addr,
                                                  cudaStream_t cuda_stream) const {
    size_t output_size = cuda_args.output_size;
    size_t thread_num = output_size > kThreadPerBlock ? kThreadPerBlock : output_size;
    Conv2dDepthWiseInputGradNCHWKernel<<<CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, output_size, thread_num), thread_num, 0,
                                         cuda_stream>>>(cuda_args, input0_addr, input1_addr, output_addr);
    return GetCudaStatus();
  }
};

template <typename T>
struct ConvolutionFunc<ConvolutionOpType::kConv2dDepthWiseInputGradNHWC, T> {
  __host__ __forceinline__ ConvolutionFunc() {}
  __host__ __forceinline__ cudaError_t operator()(const ConvolutionCudaArgs &cuda_args, const T *input0_addr,
                                                  const T *input1_addr, T *output_addr,
                                                  cudaStream_t cuda_stream) const {
    size_t output_size = cuda_args.output_size;
    size_t thread_num = output_size > kThreadPerBlock ? kThreadPerBlock : output_size;
    Conv2dDepthWiseInputGradNHWCKernel<<<CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, output_size, thread_num), thread_num, 0,
                                         cuda_stream>>>(cuda_args, input0_addr, input1_addr, output_addr);
    return GetCudaStatus();
  }
};

template <typename T>
struct ConvolutionFunc<ConvolutionOpType::kConv2dDepthWiseFilterGradNCHW, T> {
  __host__ __forceinline__ ConvolutionFunc() {}
  __host__ __forceinline__ cudaError_t operator()(const ConvolutionCudaArgs &cuda_args, const T *input0_addr,
                                                  const T *input1_addr, T *output_addr,
                                                  cudaStream_t cuda_stream) const {
    size_t output_size = cuda_args.output_size;
    size_t thread_num = output_size > kThreadPerBlock ? kThreadPerBlock : output_size;
    Conv2dDepthWiseFilterGradNCHWKernel<<<CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, output_size, thread_num), thread_num, 0,
                                          cuda_stream>>>(cuda_args, input0_addr, input1_addr, output_addr);
    return GetCudaStatus();
  }
};

template <enum ConvolutionOpType OP, typename T>
CUDA_LIB_EXPORT cudaError_t ConvolutionOpCudaFunc(const ConvolutionCudaArgs &cuda_args, const T *input0_addr,
                                                  const T *input1_addr, T *output_addr, cudaStream_t cuda_stream) {
  ConvolutionFunc<OP, T> conv_func;
  return conv_func(cuda_args, input0_addr, input1_addr, output_addr, cuda_stream);
}

REGISTER_CONVOLUTION_OP_CUDA_FUNC_FLOAT_TYPE(ConvolutionOpType::kConv2dDepthWiseForwardNCHW);
REGISTER_CONVOLUTION_OP_CUDA_FUNC_FLOAT_TYPE(ConvolutionOpType::kConv2dDepthWiseForwardNHWC);
REGISTER_CONVOLUTION_OP_CUDA_FUNC_FLOAT_TYPE(ConvolutionOpType::kConv2dDepthWiseInputGradNCHW);
REGISTER_CONVOLUTION_OP_CUDA_FUNC_FLOAT_TYPE(ConvolutionOpType::kConv2dDepthWiseInputGradNHWC);
REGISTER_CONVOLUTION_OP_CUDA_FUNC_FLOAT_TYPE(ConvolutionOpType::kConv2dDepthWiseFilterGradNCHW);
