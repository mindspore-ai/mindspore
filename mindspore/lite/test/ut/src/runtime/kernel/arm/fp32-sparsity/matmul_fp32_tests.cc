/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <random>
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "nnacl/matmul_parameter.h"
#include "src/tensor.h"
#include "securec/include/securec.h"
#include "src/litert/infer_manager.h"
#include "src/litert/inner_context.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32.h"
#include "src/litert/kernel/cpu/fp32_sparse/matmul_sparse_fp32.h"

namespace mindspore {
using mindspore::lite::Tensor;

class TestSPMMFp32 : public mindspore::CommonTest {
 public:
  TestSPMMFp32() = default;

 protected:
  virtual void GenerateData() = 0;

  std::vector<lite::Tensor *> GetInputs() {
    GenerateData();
    std::vector<lite::Tensor *> inputs;
    auto *in_tensor = new (std::nothrow) Tensor(kNumberTypeFloat, {row_, deep_}, mindspore::NHWC, lite::Category::VAR);
    if (in_tensor == nullptr) {
      std::cerr << "New tensor failed" << std::endl;
      FreeTensors(&inputs);
      return {};
    }
    inputs.emplace_back(in_tensor);

    auto *filter_tensor =
      new (std::nothrow) Tensor(kNumberTypeFloat, {deep_, col_}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    if (filter_tensor == nullptr) {
      std::cerr << "New tensor failed" << std::endl;
      FreeTensors(&inputs);
      return {};
    }
    inputs.emplace_back(filter_tensor);
    auto ret = filter_tensor->MallocData();
    if (ret != lite::RET_OK) {
      std::cerr << "Malloc filter data failed" << std::endl;
      FreeTensors(&inputs);
      return {};
    }
    ret = memcpy_s(filter_tensor->data(), filter_tensor->Size(), filter_.data(), sizeof(float) * filter_.size());
    if (ret != EOK) {
      std::cerr << "Memcpy data failed" << std::endl;
      FreeTensors(&inputs);
      return {};
    }

    auto *bias_tensor =
      new (std::nothrow) Tensor(kNumberTypeFloat, {col_}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    if (bias_tensor == nullptr) {
      std::cerr << "New tensor failed" << std::endl;
      FreeTensors(&inputs);
      return {};
    }
    inputs.emplace_back(bias_tensor);
    ret = bias_tensor->MallocData();
    if (ret != RET_OK) {
      std::cerr << "Malloc data failed" << std::endl;
      FreeTensors(&inputs);
      return {};
    }
    ret = memcpy_s(bias_tensor->data(), bias_tensor->Size(), bias_.data(), sizeof(float) * bias_.size());
    if (ret != EOK) {
      std::cerr << "Memcpy data failed" << std::endl;
      FreeTensors(&inputs);
      return {};
    }
    return inputs;
  }

  static MatMulParameter *GetMVMParameter() {
    auto parameter = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
    parameter->a_transpose_ = false;
    parameter->b_transpose_ = false;
    parameter->has_bias_ = true;
    parameter->act_type_ = ActType_No;
    parameter->op_parameter_.type_ = schema::PrimitiveType_MatMulFusion;
    parameter->op_parameter_.is_train_session_ = false;
    parameter->op_parameter_.thread_num_ = 1;
    return parameter;
  }

  static void FreeTensors(std::vector<lite::Tensor *> *tensors) {
    if (tensors == nullptr) {
      return;
    }
    for (auto *tensor : *tensors) {
      if (tensor == nullptr) {
        continue;
      }
      delete (tensor);
    }
    tensors->clear();
  }

  int row_ = 160;
  int col_ = 8;
  int deep_ = 5;
  std::vector<float> input_;
  std::vector<float> filter_;
  std::vector<float> bias_;
};

class TestSPMMFp32Accuracy : public TestSPMMFp32 {
 public:
  TestSPMMFp32Accuracy() {
    row_ = 32;
    col_ = 8;
    deep_ = 5;
  }

 protected:
  void GenerateData() override {
    input_.resize(row_ * deep_);
    filter_.resize(deep_ * col_);
    bias_.resize(col_);
    for (int i = 0; i < row_ * deep_; i++) {
      input_[i] = static_cast<float>(i);
    }
    for (int i = 0; i < col_; i++) {
      bias_[i] = static_cast<float>(i);
    }
    filter_ = {11, 0, 0, 0, 0, 0, 15, 0, 0,  13, 0,  0, 16, 0, 0, 17, 0, 0, 0, 14,
               0,  0, 0, 0, 0, 0, 0,  0, 18, 0,  20, 0, 0,  0, 0, 0,  5, 6, 0, 0};
  }

  float correct_[256] = {
    0.00,    14.00,   2.00,    31.00,   94.00,   29.00,   66.00,   24.00,   55.00,   79.00,   2.00,    101.00,  289.00,
    59.00,   241.00,  109.00,  110.00,  144.00,  2.00,    171.00,  484.00,  89.00,   416.00,  194.00,  165.00,  209.00,
    2.00,    241.00,  679.00,  119.00,  591.00,  279.00,  220.00,  274.00,  2.00,    311.00,  874.00,  149.00,  766.00,
    364.00,  275.00,  339.00,  2.00,    381.00,  1069.00, 179.00,  941.00,  449.00,  330.00,  404.00,  2.00,    451.00,
    1264.00, 209.00,  1116.00, 534.00,  385.00,  469.00,  2.00,    521.00,  1459.00, 239.00,  1291.00, 619.00,  440.00,
    534.00,  2.00,    591.00,  1654.00, 269.00,  1466.00, 704.00,  495.00,  599.00,  2.00,    661.00,  1849.00, 299.00,
    1641.00, 789.00,  550.00,  664.00,  2.00,    731.00,  2044.00, 329.00,  1816.00, 874.00,  605.00,  729.00,  2.00,
    801.00,  2239.00, 359.00,  1991.00, 959.00,  660.00,  794.00,  2.00,    871.00,  2434.00, 389.00,  2166.00, 1044.00,
    715.00,  859.00,  2.00,    941.00,  2629.00, 419.00,  2341.00, 1129.00, 770.00,  924.00,  2.00,    1011.00, 2824.00,
    449.00,  2516.00, 1214.00, 825.00,  989.00,  2.00,    1081.00, 3019.00, 479.00,  2691.00, 1299.00, 880.00,  1054.00,
    2.00,    1151.00, 3214.00, 509.00,  2866.00, 1384.00, 935.00,  1119.00, 2.00,    1221.00, 3409.00, 539.00,  3041.00,
    1469.00, 990.00,  1184.00, 2.00,    1291.00, 3604.00, 569.00,  3216.00, 1554.00, 1045.00, 1249.00, 2.00,    1361.00,
    3799.00, 599.00,  3391.00, 1639.00, 1100.00, 1314.00, 2.00,    1431.00, 3994.00, 629.00,  3566.00, 1724.00, 1155.00,
    1379.00, 2.00,    1501.00, 4189.00, 659.00,  3741.00, 1809.00, 1210.00, 1444.00, 2.00,    1571.00, 4384.00, 689.00,
    3916.00, 1894.00, 1265.00, 1509.00, 2.00,    1641.00, 4579.00, 719.00,  4091.00, 1979.00, 1320.00, 1574.00, 2.00,
    1711.00, 4774.00, 749.00,  4266.00, 2064.00, 1375.00, 1639.00, 2.00,    1781.00, 4969.00, 779.00,  4441.00, 2149.00,
    1430.00, 1704.00, 2.00,    1851.00, 5164.00, 809.00,  4616.00, 2234.00, 1485.00, 1769.00, 2.00,    1921.00, 5359.00,
    839.00,  4791.00, 2319.00, 1540.00, 1834.00, 2.00,    1991.00, 5554.00, 869.00,  4966.00, 2404.00, 1595.00, 1899.00,
    2.00,    2061.00, 5749.00, 899.00,  5141.00, 2489.00, 1650.00, 1964.00, 2.00,    2131.00, 5944.00, 929.00,  5316.00,
    2574.00, 1705.00, 2029.00, 2.00,    2201.00, 6139.00, 959.00,  5491.00, 2659.00};
};

TEST_F(TestSPMMFp32Accuracy, DenseMatmul) {
  std::vector<lite::Tensor *> inputs = GetInputs();
  ASSERT_FALSE(inputs.empty());
  auto out_tensor = new Tensor(kNumberTypeFloat, {}, mindspore::NHWC, lite::Category::VAR);
  ASSERT_NE(out_tensor, nullptr);
  std::vector<lite::Tensor *> outputs = {out_tensor};

  auto mvm_parameter = GetMVMParameter();
  auto ret = KernelInferShape(inputs, outputs, reinterpret_cast<OpParameter *>(mvm_parameter));
  ASSERT_EQ(ret, lite::RET_OK);
  ASSERT_EQ(2, out_tensor->shape().size());
  ASSERT_EQ(row_, out_tensor->shape()[0]);
  ASSERT_EQ(col_, out_tensor->shape()[1]);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  auto *matmul = new kernel::MatmulCPUKernel(reinterpret_cast<OpParameter *>(mvm_parameter), inputs, outputs, &ctx);
  ASSERT_EQ(lite::RET_OK, matmul->Init());

  ret = inputs.front()->MallocData();
  ASSERT_EQ(ret, lite::RET_OK);
  ret = memcpy_s(inputs.front()->data(), inputs.front()->Size(), input_.data(), sizeof(float) * input_.size());
  ASSERT_EQ(ret, EOK);
  ret = out_tensor->MallocData();
  ASSERT_EQ(ret, lite::RET_OK);

  ASSERT_EQ(lite::RET_OK, matmul->Run());
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensor->data()), correct_, 256, 0.0001));

  delete matmul;
  FreeTensors(&inputs);
  FreeTensors(&outputs);
}

TEST_F(TestSPMMFp32Accuracy, SparsityMatmul) {
  std::vector<lite::Tensor *> inputs = GetInputs();
  ASSERT_FALSE(inputs.empty());
  auto out_tensor = new Tensor(kNumberTypeFloat, {}, mindspore::NHWC, lite::Category::VAR);
  ASSERT_NE(out_tensor, nullptr);
  std::vector<lite::Tensor *> outputs = {out_tensor};

  auto mvm_parameter = GetMVMParameter();
  auto ret = KernelInferShape(inputs, outputs, reinterpret_cast<OpParameter *>(mvm_parameter));
  ASSERT_EQ(ret, lite::RET_OK);
  ASSERT_EQ(2, out_tensor->shape().size());
  ASSERT_EQ(row_, out_tensor->shape()[0]);
  ASSERT_EQ(col_, out_tensor->shape()[1]);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  auto *matmul =
    new kernel::MatmulSparseCPUKernel(reinterpret_cast<OpParameter *>(mvm_parameter), inputs, outputs, &ctx);
  ASSERT_EQ(lite::RET_OK, matmul->Init());

  ret = inputs.front()->MallocData();
  ASSERT_EQ(ret, lite::RET_OK);
  ret = memcpy_s(inputs.front()->data(), inputs.front()->Size(), input_.data(), sizeof(float) * input_.size());
  ASSERT_EQ(ret, EOK);
  ret = out_tensor->MallocData();
  ASSERT_EQ(ret, lite::RET_OK);

  ASSERT_EQ(lite::RET_OK, matmul->RunInstrinsics());
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensor->data()), correct_, 256, 0.0001));

  ASSERT_EQ(lite::RET_OK, matmul->Run());
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensor->data()), correct_, 256, 0.0001));
  delete matmul;
  FreeTensors(&inputs);
  FreeTensors(&outputs);
}

class TestSPMMFp32Performance : public TestSPMMFp32 {
 public:
  TestSPMMFp32Performance() {
    row_ = 160;
    col_ = 8;
    deep_ = 5;
  }

 protected:
  void GenerateData() override {
    size_t filter_size = deep_ * col_;
    input_.resize(row_ * deep_);
    for (int i = 0; i < row_ * deep_; i++) {
      input_[i] = static_cast<float>(i);
    }
    bias_.resize(col_);
    for (int i = 0; i < col_; i++) {
      bias_[i] = static_cast<float>(i);
    }

    auto zero_num = static_cast<size_t>(static_cast<float>(filter_size) * sparsity_);
    filter_.resize(filter_size, 0.0f);
    for (size_t i = zero_num; i < filter_size; i++) {
      filter_[i] = static_cast<float>(i);
    }
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::shuffle(filter_.begin(), filter_.end(), rng);
  }

  bool TestSparseMVM() {
    std::vector<lite::Tensor *> inputs = GetInputs();
    if (inputs.empty()) {
      return false;
    }
    auto out_tensor = new Tensor(kNumberTypeFloat, {}, mindspore::NHWC, lite::Category::VAR);
    if (out_tensor == nullptr) {
      return false;
    }
    std::vector<lite::Tensor *> outputs = {out_tensor};

    auto mvm_parameter = GetMVMParameter();
    auto ret = KernelInferShape(inputs, outputs, reinterpret_cast<OpParameter *>(mvm_parameter));
    if (ret != lite::RET_OK) {
      return false;
    }
    if (2 != out_tensor->shape().size()) {
      return false;
    }
    if (row_ != out_tensor->shape()[0]) {
      return false;
    }
    if (col_ != out_tensor->shape()[1]) {
      return false;
    }

    lite::InnerContext ctx;
    ctx.thread_num_ = 1;
    if (lite::RET_OK != ctx.Init()) {
      return false;
    }

    auto *matmul =
      new kernel::MatmulSparseCPUKernel(reinterpret_cast<OpParameter *>(mvm_parameter), inputs, outputs, &ctx);
    if (lite::RET_OK != matmul->Init()) {
      return false;
    }

    ret = inputs.front()->MallocData();
    if (ret != lite::RET_OK) {
      return false;
    }
    ret = memcpy_s(inputs.front()->data(), inputs.front()->Size(), input_.data(), sizeof(float) * input_.size());
    if (ret != EOK) {
      return false;
    }
    ret = out_tensor->MallocData();
    if (ret != lite::RET_OK) {
      return false;
    }

    if (lite::RET_OK != matmul->RunInstrinsics()) {
      return false;
    }
    auto start_time = lite::GetTimeUs();
    for (int i = 0; i < 1000; i++) {
      matmul->RunInstrinsics();
    }
    auto time_dur = lite::GetTimeUs() - start_time;
    std::cout << "Sparse nhwc matmul intrinsics, sparse: " << sparsity_
              << ", cost: " << (static_cast<float>(time_dur) / 1000) << "us" << std::endl;

    if (lite::RET_OK != matmul->Run()) {
      return false;
    }
    start_time = lite::GetTimeUs();
    for (int i = 0; i < 10000; i++) {
      matmul->Run();
    }
    time_dur = lite::GetTimeUs() - start_time;
    std::cout << "Sparse nhwc matmul instruction, sparse: " << sparsity_
              << ", cost: " << (static_cast<float>(time_dur) / 1000) << "us" << std::endl;
    delete matmul;
    FreeTensors(&inputs);
    FreeTensors(&outputs);
    return true;
  }

  float sparsity_ = 0.9;
};

TEST_F(TestSPMMFp32Performance, DenseMatmul) {
  std::vector<lite::Tensor *> inputs = GetInputs();
  ASSERT_FALSE(inputs.empty());
  auto out_tensor = new Tensor(kNumberTypeFloat, {}, mindspore::NHWC, lite::Category::VAR);
  ASSERT_NE(out_tensor, nullptr);
  std::vector<lite::Tensor *> outputs = {out_tensor};

  auto mvm_parameter = GetMVMParameter();
  auto ret = KernelInferShape(inputs, outputs, reinterpret_cast<OpParameter *>(mvm_parameter));
  ASSERT_EQ(ret, lite::RET_OK);
  ASSERT_EQ(2, out_tensor->shape().size());
  ASSERT_EQ(row_, out_tensor->shape()[0]);
  ASSERT_EQ(col_, out_tensor->shape()[1]);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  auto *matmul = new kernel::MatmulCPUKernel(reinterpret_cast<OpParameter *>(mvm_parameter), inputs, outputs, &ctx);
  ASSERT_EQ(lite::RET_OK, matmul->Init());

  ret = inputs.front()->MallocData();
  ASSERT_EQ(ret, lite::RET_OK);
  ret = memcpy_s(inputs.front()->data(), inputs.front()->Size(), input_.data(), sizeof(float) * input_.size());
  ASSERT_EQ(ret, EOK);
  ret = out_tensor->MallocData();
  ASSERT_EQ(ret, lite::RET_OK);

  ASSERT_EQ(lite::RET_OK, matmul->Run());

  auto start_time = lite::GetTimeUs();
  for (int i = 0; i < 1000; i++) {
    matmul->Run();
  }
  auto time_dur = lite::GetTimeUs() - start_time;
  std::cout << "Dense matmul : " << (static_cast<float>(time_dur) / 1000) << "us" << std::endl;
  delete matmul;
  FreeTensors(&inputs);
  FreeTensors(&outputs);
}

TEST_F(TestSPMMFp32Performance, SparsityMatmul) {
  this->sparsity_ = 0.3;
  auto ret = this->TestSparseMVM();
  EXPECT_EQ(ret, true);

  this->sparsity_ = 0.4;
  ret = this->TestSparseMVM();
  EXPECT_EQ(ret, true);

  this->sparsity_ = 0.5;
  ret = this->TestSparseMVM();
  EXPECT_EQ(ret, true);

  this->sparsity_ = 0.6;
  ret = this->TestSparseMVM();
  EXPECT_EQ(ret, true);

  this->sparsity_ = 0.7;
  ret = this->TestSparseMVM();
  EXPECT_EQ(ret, true);

  this->sparsity_ = 0.8;
  ret = this->TestSparseMVM();
  EXPECT_EQ(ret, true);

  this->sparsity_ = 0.9;
  ret = this->TestSparseMVM();
  EXPECT_EQ(ret, true);
}
}  // namespace mindspore
