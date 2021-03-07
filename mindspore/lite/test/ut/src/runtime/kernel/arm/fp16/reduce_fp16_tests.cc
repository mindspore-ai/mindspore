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
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/kernel_registry.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp16/reduce_fp16.h"

namespace mindspore {

class TestReduceFp16 : public mindspore::CommonTest {
 public:
  TestReduceFp16() = default;
  void Prepare(const std::vector<int> &input_shape, const std::vector<int> &output_shape, float *input_data,
               float *output_data, const int num_axis, const int *axes, const int thread_num);

  void TearDown() override;

 public:
  float err_tol = 1e-5;
  lite::Tensor in_tensor_;
  lite::Tensor out_tensor_;
  std::vector<lite::Tensor *> inputs_{&in_tensor_};
  std::vector<lite::Tensor *> outputs_{&out_tensor_};
  ReduceParameter param_ = {{}};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat16, schema::PrimitiveType_ReduceFusion};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestReduceFp16::TearDown() {
  in_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

void TestReduceFp16::Prepare(const std::vector<int> &input_shape, const std::vector<int> &output_shape,
                             float *input_data, float *output_data, const int num_axis, const int *axes,
                             const int thread_num) {
  in_tensor_.set_data_type(kNumberTypeFloat32);
  in_tensor_.set_shape(input_shape);
  out_tensor_.set_data_type(kNumberTypeFloat32);
  out_tensor_.set_shape(output_shape);
  in_tensor_.set_data(input_data);
  out_tensor_.set_data(output_data);

  bool keep_axis = false;

  int mode = static_cast<int>(schema::ReduceMode_ReduceMean);
  ReduceParameter param_ = {{}};
  param_.keep_dims_ = keep_axis;
  for (auto i = 0; i < num_axis; i++) {
    param_.axes_[i] = axes[i];
  }
  param_.num_axes_ = num_axis;
  param_.mode_ = mode;

  desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat16, schema::PrimitiveType_ReduceFusion};
  ctx_ = lite::InnerContext();
  ctx_.thread_num_ = thread_num;
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator_, nullptr);
  kernel_ = creator_(inputs_, outputs_, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc);
  ASSERT_NE(kernel_, nullptr);
}
TEST_F(TestReduceFp16, Mean) {
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float out[24] = {0.0f};
  float correct[24] = {18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                       66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0};

  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {2, 1, 4, 3};

  int axes[] = {3};
  int num_axis = 1;
  int thread_num = 1;
  Prepare(input_shape, output_shape, in, out, num_axis, axes, thread_num);
  ASSERT_EQ(0, CompareOutputData(out, correct, 24, 1e-3));
}

}  // namespace mindspore
