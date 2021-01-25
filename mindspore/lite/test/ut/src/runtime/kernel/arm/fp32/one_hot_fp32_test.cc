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
#include <vector>
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"
#include "common/common_test.h"
#include "nnacl/fp32/one_hot_fp32.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "schema/ops_generated.h"

namespace mindspore {

class TestOneHotFp32 : public mindspore::CommonTest {
 public:
  TestOneHotFp32() = default;
  void Prepare(const std::vector<int> &indices_shape, int *indices_data, int *depth, float *off_on_value,
               const int axis, const std::vector<int> &output_shape, float *output_data, const int thread_num);

  void TearDown() override;

 public:
  float err_tol = 1e-5;
  lite::Tensor indices_tensor_;
  lite::Tensor depth_tensor_;
  lite::Tensor off_on_tensor_;
  lite::Tensor out_tensor_;
  OneHotParameter *param_;
  std::vector<lite::Tensor *> inputs_{&indices_tensor_, &depth_tensor_, &off_on_tensor_};
  std::vector<lite::Tensor *> outputs_{&out_tensor_};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt32, schema::PrimitiveType_OneHot};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestOneHotFp32::TearDown() {
  indices_tensor_.set_data(nullptr);
  depth_tensor_.set_data(nullptr);
  off_on_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
  delete (kernel_);
}

void TestOneHotFp32::Prepare(const std::vector<int> &indices_shape, int *indices_data, int *depth, float *off_on_value,
                             const int axis, const std::vector<int> &output_shape, float *output_data,
                             const int thread_num) {
  indices_tensor_.set_data_type(kNumberTypeInt32);
  indices_tensor_.set_shape(indices_shape);
  indices_tensor_.set_data(indices_data);

  depth_tensor_.set_data(depth);
  off_on_tensor_.set_data_type(kNumberTypeFloat32);
  off_on_tensor_.set_data(off_on_value);

  out_tensor_.set_shape(output_shape);
  out_tensor_.set_data(output_data);

  param_ = reinterpret_cast<OneHotParameter *>(malloc(sizeof(OneHotParameter)));
  param_->axis_ = axis;
  ctx_ = lite::InnerContext();
  ctx_.thread_num_ = thread_num;
  ctx_.Init();
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  kernel_ = creator_(inputs_, outputs_, reinterpret_cast<OpParameter *>(param_), &ctx_, desc);
}

// 3 3 axis -1 -> 3 3 4
TEST_F(TestOneHotFp32, Test1) {
  std::vector<int> indices_shape{3, 3};
  int indices[9] = {0, 0, 1, 0, 0, 2, 0, 1, 2};
  int depth[1] = {4};
  float off_on[2] = {0, 1};
  std::vector<int> output_shape{3, 3, 4};
  float out_data[36] = {0};

  Prepare(indices_shape, indices, depth, off_on, -1, output_shape, out_data, 2);

  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);
  std::vector<float> expect{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  ASSERT_EQ(0, CompareOutputData(out_data, expect.data(), 36, err_tol));
}

// 3 3 axis 1 -> 3 4 3
TEST_F(TestOneHotFp32, Test2) {
  std::vector<int> indices_shape{3, 3};
  int indices[9] = {0, 0, 1, 0, 0, 2, 0, 1, 2};
  int depth[1] = {4};
  float off_on[2] = {0, 1};
  std::vector<int> output_shape{3, 4, 3};
  float out_data[36] = {0};

  Prepare(indices_shape, indices, depth, off_on, 1, output_shape, out_data, 2);

  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);
  std::vector<float> expect{1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_EQ(0, CompareOutputData(out_data, expect.data(), 36, err_tol));
}

// 3 3 axis 0 -> 4 3 3
TEST_F(TestOneHotFp32, Test3) {
  std::vector<int> indices_shape{3, 3};
  int indices[9] = {0, 0, 1, 0, 0, 2, 0, 1, 2};
  int depth[1] = {4};
  float off_on[2] = {0, 1};
  std::vector<int> output_shape{4, 3, 3};
  float out_data[36] = {0};

  Prepare(indices_shape, indices, depth, off_on, 0, output_shape, out_data, 2);

  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);
  std::vector<float> expect{1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_EQ(0, CompareOutputData(out_data, expect.data(), 36, err_tol));
}

}  // namespace mindspore
