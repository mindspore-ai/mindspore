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

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <algorithm>
#include <vector>
#include "unique_consecutive_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"

template <typename T>
struct BinaryEqual {
  int64_t col;
  const T *data;

  BinaryEqual(int64_t _col, const T *_data) : col(_col), data(_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      T lhs = data[i + a * col];
      T rhs = data[i + b * col];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
};

template <typename T>
struct BinaryNotEqual {
  int64_t col;
  const T *data;

  BinaryNotEqual(int64_t _col, const T *_data) : col(_col), data(_data) {}

  __device__ int64_t operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      T lhs = data[i + a * col];
      T rhs = data[i + b * col];
      if (lhs != rhs) {
        return 1;
      }
    }
    return 0;
  }
};

template <typename S>
struct IndexToAxis {
  int num_elements;
  int64_t axis;
  const size_t *input_shape;
  const S *range_data;
  int range_size;

  IndexToAxis(int _num_elements, int64_t _axis, const size_t *_input_shape, const S *_range_data, int _range_size)
      : num_elements(_num_elements),
        axis(_axis),
        input_shape(_input_shape),
        range_data(_range_data),
        range_size(_range_size) {}

  __device__ S operator()(S pos) const {
    size_t pos_size = num_elements / input_shape[0];
    size_t last_axis = pos / pos_size;
    for (size_t i = 1; i <= axis; ++i) {
      pos -= last_axis * pos_size;
      pos_size /= input_shape[i];
      last_axis = pos / pos_size;
    }
    for (size_t k = 0; k < range_size; k++) {
      if (last_axis == range_data[k]) {
        return 0;
      }
    }
    return 1;
  }
};

template <typename T, typename S>
std::vector<std::vector<int>> ComputeUniqueConsecutive(const T *input, int num_elements,
                                                       const std::vector<int64_t> &input_shape, S *range_data,
                                                       T *output, S *index, S *counts, cudaStream_t cuda_stream) {
  auto policy = thrust::cuda::par.on(cuda_stream);
  std::vector<std::vector<int>> out_shapes;
  // Copy input to output.
  thrust::copy(policy, thrust::device_pointer_cast(input), thrust::device_pointer_cast(input) + num_elements,
               thrust::device_pointer_cast(output));

  // Inverse indices.
  if (index == nullptr || num_elements == 0) {
    std::vector<int> idx_shape = {0};
    out_shapes.emplace_back(idx_shape);
  } else {
    thrust::adjacent_difference(policy, thrust::device_pointer_cast(output),
                                thrust::device_pointer_cast(output) + num_elements, thrust::device_pointer_cast(index),
                                thrust::not_equal_to<T>());
    thrust::fill(policy, thrust::device_pointer_cast(index), thrust::device_pointer_cast(index) + 1, 0);
    thrust::inclusive_scan(policy, thrust::device_pointer_cast(index),
                           thrust::device_pointer_cast(index) + num_elements, thrust::device_pointer_cast(index));
    std::vector<int> idx_shape(input_shape.begin(), input_shape.end());
    out_shapes.emplace_back(idx_shape);
  }

  // Unique and count.
  int output_size = num_elements;
  if (counts == nullptr || num_elements == 0) {
    output_size = thrust::unique(policy, thrust::device_pointer_cast(output),
                                 thrust::device_pointer_cast(output) + num_elements, thrust::equal_to<T>()) -
                  thrust::device_pointer_cast(output);
    std::vector<int> counts_shape = {output_size};
    out_shapes.emplace_back(counts_shape);
  } else {
    thrust::sequence(policy, thrust::device_pointer_cast(range_data),
                     thrust::device_pointer_cast(range_data) + num_elements + 1);
    output_size = thrust::unique_by_key(policy, thrust::device_pointer_cast(output),
                                        thrust::device_pointer_cast(output) + num_elements,
                                        thrust::device_pointer_cast(range_data), thrust::equal_to<T>())
                    .first -
                  thrust::device_pointer_cast(output);
    thrust::fill(policy, thrust::device_pointer_cast(range_data) + output_size,
                 thrust::device_pointer_cast(range_data) + output_size + 1, num_elements);
    thrust::adjacent_difference(policy, thrust::device_pointer_cast(range_data) + 1,
                                thrust::device_pointer_cast(range_data) + output_size + 1, counts);
    std::vector<int> counts_shape = {output_size};
    out_shapes.emplace_back(counts_shape);
  }
  std::vector<int> output_shape = {output_size};
  out_shapes.insert(out_shapes.begin(), output_shape);
  return out_shapes;
}

template <typename T, typename S>
std::vector<std::vector<int>> ComputeUniqueConsecutiveByAxis(const T *input, int num_elements,
                                                             const std::vector<int64_t> &input_shape, int64_t axis,
                                                             S *input_index, S *sorted_index, S *range_data,
                                                             T *indices_data, size_t *dev_input_shape,
                                                             size_t *dev_input_axis, T *output, S *index, S *counts,
                                                             cudaStream_t cuda_stream) {
  // Compute UniqueConsecutive by axis.
  auto policy = thrust::cuda::par.on(cuda_stream);
  std::vector<std::vector<int>> out_shapes;
  thrust::copy(policy, thrust::device_pointer_cast(input), thrust::device_pointer_cast(input) + num_elements,
               thrust::device_pointer_cast(output));
  // Do transpose.
  size_t shape_size = input_shape.size();
  cudaMemcpyAsync(dev_input_shape, input_shape.data(), sizeof(size_t) * shape_size, cudaMemcpyHostToDevice,
                  cuda_stream);
  // Used for transpose: dev_input_axis={0, 1, ..., axis, ...} -> dev_input_axis[0]=axis, dev_input_axis[axis]=0
  thrust::sequence(policy, thrust::device_pointer_cast(dev_input_axis),
                   thrust::device_pointer_cast(dev_input_axis) + shape_size);
  thrust::fill(policy, thrust::device_pointer_cast(dev_input_axis), thrust::device_pointer_cast(dev_input_axis) + 1,
               axis);
  thrust::fill(policy, thrust::device_pointer_cast(dev_input_axis) + axis,
               thrust::device_pointer_cast(dev_input_axis) + axis + 1, 0);
  CalTranspose(num_elements, input, dev_input_shape, dev_input_axis, shape_size, indices_data, cuda_stream);

  // Inverse indices.
  int64_t num_inp = input_shape[axis];
  int64_t n = num_elements / num_inp;
  thrust::sequence(policy, thrust::device_pointer_cast(range_data), thrust::device_pointer_cast(range_data) + num_inp);
  if (index == nullptr || num_elements == 0) {
    std::vector<int> idx_shape = {0};
    out_shapes.emplace_back(idx_shape);
  } else {
    thrust::adjacent_difference(policy, thrust::device_pointer_cast(range_data),
                                thrust::device_pointer_cast(range_data) + num_inp,
                                thrust::device_pointer_cast(input_index), BinaryNotEqual<T>(n, indices_data));
    thrust::fill(policy, thrust::device_pointer_cast(input_index), thrust::device_pointer_cast(input_index) + 1, 0);
    thrust::inclusive_scan(policy, thrust::device_pointer_cast(input_index),
                           thrust::device_pointer_cast(input_index) + num_inp,
                           thrust::device_pointer_cast(input_index));
    thrust::sequence(policy, thrust::device_pointer_cast(sorted_index),
                     thrust::device_pointer_cast(sorted_index) + num_inp);
    thrust::scatter(policy, thrust::device_pointer_cast(input_index),
                    thrust::device_pointer_cast(input_index) + num_inp, thrust::device_pointer_cast(sorted_index),
                    thrust::device_pointer_cast(index));
    std::vector<int> idx_shape;
    idx_shape.push_back(num_inp);
    out_shapes.emplace_back(idx_shape);
  }

  // Unique and count.
  int indices_size = num_inp;
  if (counts == nullptr || num_elements == 0) {
    indices_size = thrust::unique(policy, thrust::device_pointer_cast(range_data),
                                  thrust::device_pointer_cast(range_data) + num_inp, BinaryEqual<T>(n, indices_data)) -
                   thrust::device_pointer_cast(range_data);
    std::vector<int> counts_shape = {0};
    out_shapes.emplace_back(counts_shape);
  } else {
    thrust::sequence(policy, thrust::device_pointer_cast(sorted_index),
                     thrust::device_pointer_cast(sorted_index) + num_inp + 1);
    indices_size = thrust::unique_by_key(policy, thrust::device_pointer_cast(range_data),
                                         thrust::device_pointer_cast(range_data) + num_inp,
                                         thrust::device_pointer_cast(sorted_index), BinaryEqual<T>(n, indices_data))
                     .first -
                   thrust::device_pointer_cast(range_data);
    thrust::fill(policy, thrust::device_pointer_cast(sorted_index) + indices_size,
                 thrust::device_pointer_cast(sorted_index) + indices_size + 1, num_inp);
    thrust::sequence(policy, thrust::device_pointer_cast(counts), thrust::device_pointer_cast(counts) + num_inp);
    thrust::adjacent_difference(policy, thrust::device_pointer_cast(sorted_index) + 1,
                                thrust::device_pointer_cast(sorted_index) + num_inp + 1, counts);
    std::vector<int> counts_shape = {indices_size};
    out_shapes.emplace_back(counts_shape);
  }

  // Remove invalid dimensions according to indices, reshape the output.
  std::vector<int> output_shape(input_shape.begin(), input_shape.end());
  if (indices_size != num_inp) {
    thrust::sequence(policy, thrust::device_pointer_cast(input_index),
                     thrust::device_pointer_cast(input_index) + num_elements);
    thrust::transform(policy, thrust::device_pointer_cast(input_index),
                      thrust::device_pointer_cast(input_index) + num_elements,
                      thrust::device_pointer_cast(input_index),
                      IndexToAxis<S>(num_elements, axis, dev_input_shape, range_data, indices_size));
    thrust::remove_if(policy, thrust::device_pointer_cast(output), thrust::device_pointer_cast(output) + num_elements,
                      input_index, thrust::identity<T>());
    output_shape[axis] = indices_size;
  }
  out_shapes.insert(out_shapes.begin(), output_shape);
  return out_shapes;
}

template <typename T, typename S>
std::vector<std::vector<int>> CalUniqueConsecutive(const T *input, int num_elements,
                                                   const std::vector<int64_t> &input_shape, bool is_axis_none,
                                                   int64_t axis, S *input_index, S *sorted_index, S *range_data,
                                                   T *indices_data, size_t *dev_input_shape, size_t *dev_input_axis,
                                                   T *output, S *index, S *counts, cudaStream_t cuda_stream) {
  if (is_axis_none) {
    return ComputeUniqueConsecutive(input, num_elements, input_shape, range_data, output, index, counts, cuda_stream);
  }
  return ComputeUniqueConsecutiveByAxis(input, num_elements, input_shape, axis, input_index, sorted_index, range_data,
                                        indices_data, dev_input_shape, dev_input_axis, output, index, counts,
                                        cuda_stream);
}

template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<float, int>(
  const float *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_axis_none, int64_t axis,
  int *input_index, int *sorted_index, int *range_data, float *indices_data, size_t *dev_input_shape,
  size_t *dev_input_axis, float *output, int *index, int *counts, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<half, int>(
  const half *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_axis_none, int64_t axis,
  int *input_index, int *sorted_index, int *range_data, half *indices_data, size_t *dev_input_shape,
  size_t *dev_input_axis, half *output, int *index, int *counts, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<int, int>(
  const int *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_axis_none, int64_t axis,
  int *input_index, int *sorted_index, int *range_data, int *indices_data, size_t *dev_input_shape,
  size_t *dev_input_axis, int *output, int *index, int *counts, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<int64_t, int64_t>(
  const int64_t *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_axis_none, int64_t axis,
  int64_t *input_index, int64_t *sorted_index, int64_t *range_data, int64_t *indices_data, size_t *dev_input_shape,
  size_t *dev_input_axis, int64_t *output, int64_t *index, int64_t *counts, cudaStream_t cuda_stream);
