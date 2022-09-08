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

#include "triplet_margin_loss_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

__device__ __forceinline__ int64_t Index(const int64_t &index, const int64_t &dim) { return dim == 1 ? 0 : index; }


template <typename T>
__global__ void FillAndBroadcast(const int64_t size, const size_t shape_size,
                                const int64_t *tensor_shapes, const int64_t *dst_shape,
                                const T *anchor, const T *positive, const T *negative,
                                T *anchor_broadcast) {
  const T *pair_tensor[3] = {anchor, positive, negative};
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < 3*size; pos += blockDim.x * gridDim.x) {
    const size_t mode = pos/size;
    const int64_t *src_shape = tensor_shapes + shape_size * mode;
    size_t tmp_pos = pos % size;
    size_t pos_size = size / dst_shape[0];
    size_t dst_index_array[8];
    dst_index_array[0] = tmp_pos / pos_size;
    for (size_t i = 1; i < shape_size; i++) {
      tmp_pos -= dst_index_array[i - 1] * pos_size;
      pos_size = pos_size / dst_shape[i];
      dst_index_array[i] = tmp_pos / pos_size;
    }
    size_t src_size = 1;
    for (size_t i = 0; i < shape_size; i++) {
      src_size *= src_shape[i];
    }
    size_t src_pos = 0;
    for (size_t i = 0; i < shape_size; i++) {
      src_size /= src_shape[i];
      size_t length_by_index = Index(dst_index_array[i], src_shape[i]) * src_size;
      src_pos += length_by_index;
    }
    (anchor_broadcast + mode * size)[pos % size] = pair_tensor[mode][src_pos];
  }
  return;
}

template <typename T>
__global__ void PairwiseDistance(const T *anchor, const T *positive, const T *negative,
                                 const size_t *bound_list, const size_t bound, const size_t outer_size,
                                 const size_t inner_size, float *tem_output, const size_t n, const int64_t p,
                                 const float eps) {
  const T *pair_tensor[3] = {anchor, positive, negative};
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
       pos < n * outer_size * inner_size; pos += gridDim.x * blockDim.x) {
    size_t mode = pos / (outer_size * inner_size);
    size_t idx = pos % (outer_size * inner_size);
    float res = 0;
    size_t x = idx / inner_size % outer_size;
    size_t y = idx % inner_size;
    for (int i = 0; i < bound_list[mode]; i++) {
      size_t input_offset = x * bound * inner_size + i * inner_size + y;
      float base =
        abs(static_cast<T>(pair_tensor[mode / 2][input_offset] - pair_tensor[(mode + 3) / 2][input_offset]) + eps);
      float tem = pow(base, static_cast<float>(p));
      res += tem;
    }
    tem_output[pos] = pow(res, static_cast<float>(1.0 / p));
  }
  return;
}


__global__ void PairwiseDistancePzero(const size_t *bound_list, const size_t output_size,
                                                 float *tem_output, const size_t n) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < n * output_size; pos += gridDim.x * blockDim.x) {
    size_t mode = pos / output_size;
    tem_output[pos] = static_cast<float>(bound_list[mode]);
  }
  return;
}



__global__ void SwapTrue(float *tem_output, const size_t output_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += gridDim.x * blockDim.x) {
    tem_output[pos + output_size] = tem_output[pos + output_size] > tem_output[pos + 2 * output_size] ?
                                    tem_output[pos + 2 * output_size] :  tem_output[pos + output_size];
  }
  return;
}


__global__ void MaxReduction(float *tem_output, float *output, const size_t output_size, const float *margin) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += gridDim.x * blockDim.x) {
    output[pos] = max(static_cast<float>(margin[0]) + tem_output[pos] - tem_output[pos + output_size], 0.0);
  }
  return;
}


__global__ void AddTile(float *tmp_loss, size_t index) {
  tmp_loss[0] += tmp_loss[index];
}

__global__ void PartialSum(float *tmp_loss, size_t stride) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += blockDim.x * gridDim.x) {
    tmp_loss[i] += tmp_loss[i + stride];
  }
}

template <typename S>
__global__ void ReductionDivde(S *output, float *tem_output, const size_t k) {
  output[0] = tem_output[0] / k;
}

// half
template <>
__global__ void PairwiseDistance(const half *anchor, const half *positive, const half *negative,
                                 const size_t *bound_list, const size_t bound, const size_t outer_size,
                                 const size_t inner_size, float *tem_output, const size_t n, const int64_t p,
                                 const float eps) {
  const half *pair_tensor[3] = {anchor, positive, negative};
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
       pos < n * outer_size * inner_size; pos += gridDim.x * blockDim.x) {
    size_t mode = pos / (outer_size * inner_size);
    size_t idx = pos % (outer_size * inner_size);
    float res = 0;
    size_t x = idx / inner_size % outer_size;
    size_t y = idx % inner_size;
    for (int i = 0; i < bound_list[mode]; i++) {
      size_t input_offset = x * bound * inner_size + i * inner_size + y;
      float base = abs(__half2float(pair_tensor[mode / 2][input_offset]) -
                       __half2float(pair_tensor[(mode+3) / 2][input_offset]) + eps);
      float tem = pow(base, static_cast<float>(p));
      res += tem;
    }
    tem_output[pos] = pow(res, static_cast<float>(1.0 / p));
  }
  return;
}


// half
__global__ void MaxReduction(float *tem_output, half *output, const size_t output_size, const float *margin) {
  float lower_bound = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += gridDim.x * blockDim.x) {
    output[pos] = __float2half(max(margin[0] + tem_output[pos] - tem_output[pos + output_size], lower_bound));
  }
  return;
}


// half
template <>
__global__ void ReductionDivde(half *output, float *tem_output, const size_t k) {
  output[0] = __float2half((tem_output[0] / k));
}

// Complex
template <typename S>
__global__ void PairwiseDistance(const Complex<S> *anchor, const Complex<S> *positive,
                                                 const Complex<S> *negative, const size_t *bound_list,
                                                 const size_t bound, const size_t outer_size, const size_t inner_size,
                                                 float *tem_output, const size_t n, const int64_t p, const float eps) {
  const Complex<S> *pair_tensor[3] = {anchor, positive, negative};
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
       pos < n * outer_size * inner_size; pos += gridDim.x * blockDim.x) {
    size_t mode = pos / (outer_size * inner_size);
    size_t idx = pos % (outer_size * inner_size);
    S res = 0;
    size_t x = idx / inner_size % outer_size;
    size_t y = idx % inner_size;
    for (int i = 0; i < bound_list[mode]; i++) {
      size_t input_offset = x * bound * inner_size + i * inner_size + y;
      Complex<S> base_complex =
        pair_tensor[mode / 2][input_offset] - pair_tensor[(mode + 3) / 2][input_offset] + static_cast<S>(eps);
      S base = sqrt((base_complex.real() * base_complex.real() + base_complex.imag() * base_complex.imag()));
      S tem = pow(base, static_cast<S>(p));
      res += tem;
    }
    tem_output[pos] = pow(res, 1.0 / p);
  }
  return;
}

template <typename T, typename S>
void CalTripletMarginLoss(const T *anchor, const T *positive, const T *negative, T *anchor_broadcast,
                          T *positive_broadcast, T *negative_broadcast, S *output, float *tem_output,
                          const int64_t *tensor_shapes, const int64_t *dst_shape, const size_t outer_size,
                          const size_t inner_size, const size_t *bound_list, const size_t bound,
                          const size_t shape_size, float *margin, const int64_t p, const float eps,
                          const std::string reduction, const bool swap, const bool need_broadcast,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int64_t size = outer_size * inner_size * bound;
  size_t n;
  if (swap)
    n = 3;
  else
    n = 2;
  const size_t output_size = outer_size * inner_size;
  if (p == 0) {
    PairwiseDistancePzero<<<CUDA_BLOCKS(device_id, n * output_size), CUDA_THREADS(device_id), 0, cuda_stream>>>
                                       (bound_list, output_size, tem_output, n);
  } else if (need_broadcast) {
    FillAndBroadcast<<<CUDA_BLOCKS(device_id, 3 * size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
                            size, shape_size, tensor_shapes, dst_shape, anchor, positive, negative,
                            anchor_broadcast);
    PairwiseDistance<<<CUDA_BLOCKS(device_id, n * output_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
                            anchor_broadcast, positive_broadcast, negative_broadcast, bound_list, bound, outer_size,
                            inner_size, tem_output, n, p, eps);
  } else {
    PairwiseDistance<<<CUDA_BLOCKS(device_id, n * output_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
                        anchor, positive, negative, bound_list, bound, outer_size, inner_size, tem_output, n, p, eps);
  }

  if (swap) {
    SwapTrue<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(tem_output,
      output_size);
  }
  if (reduction == "none") {
    MaxReduction<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(tem_output,
      output, output_size, margin);
  } else {
    MaxReduction<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(tem_output,
      tem_output, output_size, margin);
    if (output_size % 2 == 1 && output_size != 1) {
      AddTile<<<1, 1, 0, cuda_stream>>>(tem_output, output_size - 1);
    }
    for (size_t stride = output_size / 2; stride > 0; stride >>= 1) {
      PartialSum<<<CUDA_BLOCKS(device_id, stride), CUDA_THREADS(device_id), 0, cuda_stream>>>(tem_output, stride);
      if (stride > 2 && stride % 2 == 1) {
        AddTile<<<1, 1, 0, cuda_stream>>>(tem_output, stride - 1);
      }
    }
    if (reduction == "mean") {
      ReductionDivde<<<1, 1, 0, cuda_stream>>>(output, tem_output, output_size);
    } else {
      ReductionDivde<<<1, 1, 0, cuda_stream>>>(output, tem_output, 1);
    }
  }

  return;
}


template CUDA_LIB_EXPORT void CalTripletMarginLoss<int8_t, float>(
    const int8_t *anchor, const int8_t *positive, const int8_t *negative,
    int8_t *anchor_broadcast, int8_t *positive_broadcast, int8_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<int16_t, float>(
    const int16_t *anchor, const int16_t *positive, const int16_t *negative,
    int16_t *anchor_broadcast, int16_t *positive_broadcast, int16_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<int32_t, float>(
    const int32_t *anchor, const int32_t *positive, const int32_t *negative,
    int32_t *anchor_broadcast, int32_t *positive_broadcast, int32_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<int64_t, float>(
    const int64_t *anchor, const int64_t *positive, const int64_t *negative,
    int64_t *anchor_broadcast, int64_t *positive_broadcast, int64_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<uint8_t, float>(
    const uint8_t *anchor, const uint8_t *positive, const uint8_t *negative,
    uint8_t *anchor_broadcast, uint8_t *positive_broadcast, uint8_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<uint16_t, float>(
    const uint16_t *anchor, const uint16_t *positive, const uint16_t *negative,
    uint16_t *anchor_broadcast, uint16_t *positive_broadcast, uint16_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<uint32_t, float>(
    const uint32_t *anchor, const uint32_t *positive, const uint32_t *negative,
    uint32_t *anchor_broadcast, uint32_t *positive_broadcast, uint32_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<uint64_t, float>(
    const uint64_t *anchor, const uint64_t *positive, const uint64_t *negative,
    uint64_t *anchor_broadcast, uint64_t *positive_broadcast, uint64_t *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<double, float>(
    const double *anchor, const double *positive, const double *negative,
    double *anchor_broadcast, double *positive_broadcast, double *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTripletMarginLoss<float, float>(
    const float *anchor, const float *positive, const float *negative,
    float *anchor_broadcast, float *positive_broadcast, float *negative_broadcast,
    float *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalTripletMarginLoss<half, half>(
    const half *anchor, const half *positive, const half *negative,
    half *anchor_broadcast, half *positive_broadcast, half *negative_broadcast,
    half *output, float *tem_output, const int64_t *tensor_shapes,
    const int64_t *dst_shape, const size_t outer_size, const size_t inner_size,
    const size_t *bound_list, const size_t bound, const size_t shape_size, float *margin,
    const int64_t p, const float eps, const std::string reduction,
    const bool swap, const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void
CalTripletMarginLoss<Complex<float>, float>(
    const Complex<float> *anchor, const Complex<float> *positive, const Complex<float> *negative,
    Complex<float> *anchor_broadcast, Complex<float> *positive_broadcast, Complex<float> *negative_broadcast,
     float *output, float *tem_output,
    const int64_t *tensor_shapes, const int64_t *dst_shape,
    const size_t outer_size, const size_t inner_size, const size_t *bound_list, const size_t bound,
    const size_t shape_size, float *margin, const int64_t p,
    const float eps, const std::string reduction, const bool swap,
    const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void
CalTripletMarginLoss<Complex<double>, float>(
    const Complex<double> *anchor, const Complex<double> *positive,
    const Complex<double> *negative,
    Complex<double> *anchor_broadcast, Complex<double> *positive_broadcast, Complex<double> *negative_broadcast,
    float *output, float *tem_output,
    const int64_t *tensor_shapes, const int64_t *dst_shape,
    const size_t outer_size, const size_t inner_size, const size_t *bound_list, const size_t bound,
    const size_t shape_size, float *margin, const int64_t p,
    const float eps, const std::string reduction, const bool swap,
    const bool need_broadcast, const uint32_t &device_id,
    cudaStream_t cuda_stream);
