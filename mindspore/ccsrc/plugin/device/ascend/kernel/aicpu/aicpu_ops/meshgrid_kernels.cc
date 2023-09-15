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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/meshgrid_kernels.h"
#include <vector>
#include <string>
#include <map>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "common/kernel_log.h"
#include "common/kernel_errcode.h"
#include "common/tensor.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {

template <typename T>
uint32_t MeshgridTask(const std::vector<uintptr_t> &io_addrs_, const std::string &indexing, size_t ndim,
                      const std::vector<int> &bcast) {
  auto shards = [&](const int64_t begin, const int64_t end) {
    for (int i = begin; i < end; ++i) {  // 0~ndim
      auto new_i = i;
      auto s = bcast;
      if (indexing == "xy" && i < 2) {
        new_i = 1 - i;
        auto tmp = s[0];
        s[0] = s[1];
        s[1] = tmp;
      }
      size_t row_ = 1;
      size_t col_ = 1;
      for (int j = 0; j <= new_i; j++) {
        row_ *= s[j];
      }
      for (int j = new_i + 1; j < static_cast<int>(s.size()); j++) {
        col_ *= s[j];
      }

      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> input_map(reinterpret_cast<T *>(io_addrs_[i]), bcast[i],
                                                                       1);
      const auto &input = Eigen::Tensor<T, 2, Eigen::RowMajor>(input_map);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> output(reinterpret_cast<T *>(io_addrs_[ndim + i]), row_,
                                                                    col_);

      Eigen::Tensor<T, 2, Eigen::RowMajor> origin(bcast[i], row_ * col_ / bcast[i]);
      for (int c = 0; c < bcast[i]; ++c) {
        for (int r = 0; r < static_cast<int>(row_ * col_ / bcast[i]); ++r) {
          origin(c, r) = input(c, 0);
        }
      }

      for (size_t j = 0; j < row_ * col_ / bcast[i] / col_; ++j) {
        Eigen::array<int64_t, 2> offsets_in = {0, static_cast<int64_t>(col_ * j)};
        Eigen::array<int64_t, 2> offsets_out = {static_cast<int64_t>(bcast[i] * j), 0};
        Eigen::array<int64_t, 2> extents = {static_cast<int64_t>(bcast[i]), static_cast<int64_t>(col_)};
        output.slice(offsets_out, extents) = origin.slice(offsets_in, extents);
      }
    }
  };

  const int64_t perUnitSize = 1;  // shard unit size
  ParallelFor(ndim, perUnitSize, shards);

  return kAicpuKernelStateSucess;
}

uint32_t MeshgridKernel::DoCompute() {
  std::map<int, std::function<uint32_t(std::vector<uintptr_t> &, std::string &, size_t &, std::vector<int> &)>> calls;
  calls[aicpuops::DataType::MS_INT8] = MeshgridTask<int8_t>;
  calls[aicpuops::DataType::MS_INT16] = MeshgridTask<int16_t>;
  calls[aicpuops::DataType::MS_INT32] = MeshgridTask<int32_t>;
  calls[aicpuops::DataType::MS_INT64] = MeshgridTask<int64_t>;
  calls[aicpuops::DataType::MS_FLOAT16] = MeshgridTask<Eigen::half>;
  calls[aicpuops::DataType::MS_FLOAT32] = MeshgridTask<float>;
  calls[aicpuops::DataType::MS_FLOAT64] = MeshgridTask<double>;
  calls[aicpuops::DataType::MS_UINT8] = MeshgridTask<uint8_t>;
  calls[aicpuops::DataType::MS_UINT16] = MeshgridTask<uint16_t>;
  calls[aicpuops::DataType::MS_UINT32] = MeshgridTask<uint32_t>;
  calls[aicpuops::DataType::MS_UINT64] = MeshgridTask<uint64_t>;
  calls[aicpuops::DataType::MS_BOOL] = MeshgridTask<bool>;
  return calls[input_type_](io_addrs_, indexing_, ndim_, bcast_);
}

uint32_t MeshgridKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> nodedef_map = node_def_.attrs();
  indexing_ = nodedef_map["indexing"].s();

  input_type_ = static_cast<aicpuops::DataType>(node_def_.inputs(0).tensor_type());

  ndim_ = node_def_.inputs_size();
  bcast_.resize(ndim_);
  for (int n = 0; n < node_def_.inputs_size(); ++n) {
    aicpuops::Tensor input_tensor = node_def_.inputs(n);
    aicpuops::TensorShape input_shape = input_tensor.tensor_shape();
    if (input_shape.dim().size() != 1) {
      AICPU_LOGE("input tensor should be 1-D.");
    }
    bcast_[n] = input_shape.dim(0).size();
  }

  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t Meshgrid(void *param) {
  aicpu::MeshgridKernel meshgridKernel;
  return meshgridKernel.Compute(param);
}
}
