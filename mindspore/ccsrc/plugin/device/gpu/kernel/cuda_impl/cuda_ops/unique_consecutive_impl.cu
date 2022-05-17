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

std::vector<int> RemoveElementsIndex(const std::vector<int> &indices, int size,
                                     const std::vector<std::vector<size_t>> &pos_map, int64_t axis) {
  int num_elements = pos_map.size();
  std::vector<int> output(num_elements, 1);
  for (int pos = 0; pos < num_elements; ++pos) {
    // Check the axis of the element.
    int element_index = pos_map[pos][axis];
    for (int i = 0; i < size; ++i) {
      if (element_index == indices[i]) {
        output[pos] = 0;
      }
    }
  }
  return output;
}

std::vector<std::vector<size_t>> GetPositionArray(int num_elements, const std::vector<int64_t> &input_shape) {
  std::vector<std::vector<size_t>> pos_map(num_elements);
  size_t shape_size = input_shape.size();
  size_t temp_pos;
  size_t pos_size;

  for (int pos = 0; pos < num_elements; ++pos) {
    std::vector<size_t> array(shape_size, 0);
    temp_pos = pos;
    pos_size = num_elements / input_shape[0];
    array[0] = temp_pos / pos_size;
    for (size_t i = 1; i < shape_size; ++i) {
      temp_pos -= array[i - 1] * pos_size;
      pos_size = pos_size / input_shape[i];
      array[i] = temp_pos / pos_size;
    }
    pos_map[pos] = array;
  }
  return pos_map;
}

std::vector<int64_t> GetTransposeIndices(int num_elements, const std::vector<int64_t> &input_shape,
                                         const std::vector<std::vector<size_t>> &pos_map, int64_t axis) {
  // Get transpose axis.
  size_t shape_size = input_shape.size();
  int64_t cnt = 0;
  std::vector<int64_t> transpose_axis(shape_size, 0);
  std::generate(transpose_axis.begin(), transpose_axis.end(), [&] { return cnt++; });
  transpose_axis[0] = axis;
  transpose_axis[axis] = 0;
  // Do Transpose
  std::vector<int64_t> output_indices(num_elements, 0);
  for (int pos = 0; pos < num_elements; pos++) {
    auto pos_array = pos_map[pos];
    size_t new_pos = pos_array[transpose_axis[shape_size - 1]];
    size_t new_pos_size = 1;
    for (int64_t i = shape_size - 2; i >= 0; i--) {
      new_pos_size *= input_shape[transpose_axis[i + 1]];
      new_pos += pos_array[transpose_axis[i]] * new_pos_size;
    }
    output_indices[new_pos] = pos;
  }
  return output_indices;
}

template <typename T, typename S>
std::vector<std::vector<int>> ComputeUniqueConsecutiveFlattend(const T *input, int num_elements,
                                                               const std::vector<int64_t> &input_shape, S *input_index,
                                                               S *sorted_index, S *range_data, T *output, S *index,
                                                               S *counts, cudaStream_t cuda_stream) {
  auto policy = thrust::cuda::par.on(cuda_stream);
  std::vector<std::vector<int>> out_shapes;
  // Copy input to output.
  thrust::copy(thrust::device_pointer_cast(input), thrust::device_pointer_cast(input) + num_elements,
               thrust::device_pointer_cast(output));

  // Inverse indices.
  thrust::adjacent_difference(policy, thrust::device_pointer_cast(output),
                              thrust::device_pointer_cast(output) + num_elements,
                              thrust::device_pointer_cast(input_index), thrust::not_equal_to<T>());
  thrust::fill(policy, thrust::device_pointer_cast(input_index), thrust::device_pointer_cast(input_index) + 1, 0);
  thrust::inclusive_scan(policy, thrust::device_pointer_cast(input_index),
                         thrust::device_pointer_cast(input_index) + num_elements,
                         thrust::device_pointer_cast(input_index));
  if (index != nullptr) {
    thrust::sequence(policy, thrust::device_pointer_cast(sorted_index),
                     thrust::device_pointer_cast(sorted_index) + num_elements);
    thrust::scatter(policy, thrust::device_pointer_cast(input_index),
                    thrust::device_pointer_cast(input_index) + num_elements, thrust::device_pointer_cast(sorted_index),
                    thrust::device_pointer_cast(index));
    std::vector<int> idx_shape(input_shape.begin(), input_shape.end());
    out_shapes.emplace_back(idx_shape);
  } else {
    std::vector<int> idx_shape = {0};
    out_shapes.emplace_back(idx_shape);
  }
  // Unique.
  thrust::sequence(policy, thrust::device_pointer_cast(range_data),
                   thrust::device_pointer_cast(range_data) + num_elements + 1);
  int output_size = thrust::unique_by_key(policy, thrust::device_pointer_cast(output),
                                          thrust::device_pointer_cast(output) + num_elements,
                                          thrust::device_pointer_cast(range_data), thrust::equal_to<T>())
                      .first -
                    thrust::device_pointer_cast(output);
  std::vector<int> output_shape = {output_size};
  out_shapes.insert(out_shapes.begin(), output_shape);
  // Count.
  if (counts != nullptr) {
    thrust::fill(policy, thrust::device_pointer_cast(range_data) + output_size,
                 thrust::device_pointer_cast(range_data) + output_size + 1, num_elements);
    thrust::sequence(policy, thrust::device_pointer_cast(counts), thrust::device_pointer_cast(counts) + num_elements);
    thrust::adjacent_difference(policy, thrust::device_pointer_cast(range_data) + 1,
                                thrust::device_pointer_cast(range_data) + output_size + 1, counts);
    std::vector<int> counts_shape = {output_size};
    out_shapes.emplace_back(counts_shape);
  } else {
    std::vector<int> counts_shape = {0};
    out_shapes.emplace_back(counts_shape);
  }
  return out_shapes;
}

template <typename T, typename S>
std::vector<std::vector<int>> ComputeUniqueConsecutiveByAxis(const T *input, int num_elements,
                                                             const std::vector<int64_t> &input_shape, bool is_flattend,
                                                             int64_t axis, S *input_index, S *sorted_index,
                                                             S *range_data, T *indices_data, T *output, S *index,
                                                             S *counts, cudaStream_t cuda_stream) {
  // Compute UniqueConsecutive by axis.
  auto policy = thrust::cuda::par.on(cuda_stream);
  std::vector<std::vector<int>> out_shapes;
  // Do transpose.
  int64_t num_inp = input_shape[axis];
  int64_t n = num_elements / num_inp;
  thrust::copy(thrust::device_pointer_cast(input), thrust::device_pointer_cast(input) + num_elements,
               thrust::device_pointer_cast(output));
  std::vector<std::vector<size_t>> pos_map = GetPositionArray(num_elements, input_shape);
  std::vector<int64_t> transpose_indices = GetTransposeIndices(num_elements, input_shape, pos_map, axis);
  thrust::device_vector<int> indices_map(transpose_indices.begin(), transpose_indices.end());
  thrust::gather(policy, indices_map.begin(), indices_map.end(), thrust::device_pointer_cast(input),
                 thrust::device_pointer_cast(indices_data));

  // Inverse indices.
  thrust::sequence(policy, thrust::device_pointer_cast(range_data), thrust::device_pointer_cast(range_data) + num_inp);
  thrust::adjacent_difference(policy, thrust::device_pointer_cast(range_data),
                              thrust::device_pointer_cast(range_data) + num_inp,
                              thrust::device_pointer_cast(input_index), BinaryNotEqual<T>(n, indices_data));
  thrust::fill(policy, thrust::device_pointer_cast(input_index), thrust::device_pointer_cast(input_index) + 1, 0);
  thrust::inclusive_scan(policy, thrust::device_pointer_cast(input_index),
                         thrust::device_pointer_cast(input_index) + num_inp, thrust::device_pointer_cast(input_index));
  if (index != nullptr) {
    thrust::sequence(policy, thrust::device_pointer_cast(sorted_index),
                     thrust::device_pointer_cast(sorted_index) + num_inp);
    thrust::scatter(policy, thrust::device_pointer_cast(input_index),
                    thrust::device_pointer_cast(input_index) + num_inp, thrust::device_pointer_cast(sorted_index),
                    thrust::device_pointer_cast(index));
    std::vector<int> idx_shape;
    idx_shape.push_back(num_inp);
    out_shapes.emplace_back(idx_shape);
  } else {
    std::vector<int> idx_shape = {0};
    out_shapes.emplace_back(idx_shape);
  }

  // Unique.
  thrust::sequence(policy, thrust::device_pointer_cast(sorted_index),
                   thrust::device_pointer_cast(sorted_index) + num_inp + 1);
  int indices_size = thrust::unique_by_key(policy, thrust::device_pointer_cast(range_data),
                                           thrust::device_pointer_cast(range_data) + num_inp,
                                           thrust::device_pointer_cast(sorted_index), BinaryEqual<T>(n, indices_data))
                       .first -
                     thrust::device_pointer_cast(range_data);

  std::vector<int> indices(indices_size);
  cudaMemcpyAsync(indices.data(), range_data, indices_size * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
  cudaStreamSynchronize(cuda_stream);
  std::vector<int> elements_index = RemoveElementsIndex(indices, indices_size, pos_map, axis);
  thrust::device_vector<int> remove_map(elements_index.begin(), elements_index.end());
  thrust::remove_if(thrust::device_pointer_cast(output), thrust::device_pointer_cast(output) + num_elements,
                    remove_map.begin(), thrust::identity<T>());
  std::vector<int> output_shape(input_shape.begin(), input_shape.end());
  output_shape[axis] = indices_size;
  out_shapes.insert(out_shapes.begin(), output_shape);

  // Count.
  if (counts != nullptr) {
    thrust::fill(policy, thrust::device_pointer_cast(sorted_index) + indices_size,
                 thrust::device_pointer_cast(sorted_index) + indices_size + 1, num_inp);
    thrust::sequence(policy, thrust::device_pointer_cast(counts), thrust::device_pointer_cast(counts) + num_inp);
    thrust::adjacent_difference(policy, thrust::device_pointer_cast(sorted_index) + 1,
                                thrust::device_pointer_cast(sorted_index) + num_inp + 1, counts);
    std::vector<int> counts_shape = {indices_size};
    out_shapes.emplace_back(counts_shape);
  } else {
    std::vector<int> counts_shape = {0};
    out_shapes.emplace_back(counts_shape);
  }
  return out_shapes;
}

template <typename T, typename S>
std::vector<std::vector<int>> CalUniqueConsecutive(const T *input, int num_elements,
                                                   const std::vector<int64_t> &input_shape, bool is_flattend,
                                                   int64_t axis, S *input_index, S *sorted_index, S *range_data,
                                                   T *indices_data, T *output, S *index, S *counts,
                                                   cudaStream_t cuda_stream) {
  if (is_flattend) {
    return ComputeUniqueConsecutiveFlattend(input, num_elements, input_shape, input_index, sorted_index, range_data,
                                            output, index, counts, cuda_stream);
  }
  return ComputeUniqueConsecutiveByAxis(input, num_elements, input_shape, is_flattend, axis, input_index, sorted_index,
                                        range_data, indices_data, output, index, counts, cuda_stream);
}

template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<float, int>(
  const float *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_flattend, int64_t axis,
  int *input_index, int *sorted_index, int *range_data, float *indices_data, float *output, int *index, int *counts,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<half, int>(
  const half *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_flattend, int64_t axis,
  int *input_index, int *sorted_index, int *range_data, half *indices_data, half *output, int *index, int *counts,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<int, int>(
  const int *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_flattend, int64_t axis,
  int *input_index, int *sorted_index, int *range_data, int *indices_data, int *output, int *index, int *counts,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT std::vector<std::vector<int>> CalUniqueConsecutive<int64_t, int64_t>(
  const int64_t *input, int num_elements, const std::vector<int64_t> &input_shape, bool is_flattend, int64_t axis,
  int64_t *input_index, int64_t *sorted_index, int64_t *range_data, int64_t *indices_data, int64_t *output,
  int64_t *index, int64_t *counts, cudaStream_t cuda_stream);
