/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/base/depth_to_space_base.h"
#include "mindspore/lite/nnacl/common_func.h"

namespace mindspore {

class DepthToSpaceTestFp32 : public mindspore::CommonTest {
 public:
  DepthToSpaceTestFp32() = default;
};

TEST_F(DepthToSpaceTestFp32, DepthToSpaceTest2) {
  float input[16] = {1, 2, 10, 20, 5, 6, 3, 8, 18, 10, 11, 55, 3, 4, 15, 25};
  constexpr int kOutSize = 16;
  float expect_out[kOutSize] = {1, 2, 5, 6, 10, 20, 3, 8, 18, 10, 3, 4, 11, 55, 15, 25};

  float output[kOutSize];
  int in_shape[4] = {1, 2, 2, 4};
  int out_shape[4] = {1, 4, 4, 1};
  DepthToSpaceParameter param;
  param.block_size_ = 2;
  int in_strides[4];
  ComputeStrides(in_shape, in_strides, 4);
  int out_strides[4];
  ComputeStrides(out_shape, out_strides, 4);
  param.in_stride_dim0_ = in_strides[0];
  param.in_stride_dim1_ = in_strides[1];
  param.in_stride_dim2_ = in_strides[2];
  param.out_stride_dim0_ = out_strides[0];
  param.out_stride_dim1_ = out_strides[1];
  param.out_stride_dim2_ = out_strides[2];
  param.data_type_size_ = sizeof(float);
  DepthToSpaceForNHWC((const void *)input, output, in_shape, &param);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(DepthToSpaceTestFp32, DepthToSpaceTest3) {
  float input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int kOutSize = 8;
  float expect_out[kOutSize] = {1, 2, 3, 4, 5, 6, 7, 8};

  float output[kOutSize];
  int in_shape[4] = {1, 1, 1, 8};
  int out_shape[4] = {1, 2, 2, 2};
  DepthToSpaceParameter param;
  param.block_size_ = 2;
  int in_strides[4];
  ComputeStrides(in_shape, in_strides, 4);
  int out_strides[4];
  ComputeStrides(out_shape, out_strides, 4);
  param.in_stride_dim0_ = in_strides[0];
  param.in_stride_dim1_ = in_strides[1];
  param.in_stride_dim2_ = in_strides[2];
  param.out_stride_dim0_ = out_strides[0];
  param.out_stride_dim1_ = out_strides[1];
  param.out_stride_dim2_ = out_strides[2];
  param.data_type_size_ = sizeof(float);
  DepthToSpaceForNHWC((const void *)input, output, in_shape, &param);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}
}  // namespace mindspore
