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
#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "src/runtime/kernel/arm/fp32_grad/sparse_softmax_cross_entropy_with_logits.h"
#include "src/kernel_registry.h"

namespace mindspore {

class TestSoftmaxCrossEntropyFp32 : public mindspore::CommonTest {
 public:
  TestSoftmaxCrossEntropyFp32() {}
};

TEST_F(TestSoftmaxCrossEntropyFp32, SoftmaxCrossEntropyFp32) {
  // prepare stage
  auto sce_param = reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  ASSERT_NE(sce_param, nullptr);
  size_t input_size;

  std::string input_path = "./test_data/operators/sce_fp32_1_y_6_4.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_y({6, 4});
  lite::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
  y_tensor.set_data(input_data);

  std::string label_path = "./test_data/operators/sce_fp32_1_l_6.bin";
  auto ll_labels = reinterpret_cast<int64_t *>(mindspore::lite::ReadFile(label_path.c_str(), &input_size));
  ASSERT_NE(ll_labels, nullptr);
  auto labels = new float[6 * 4];
  ASSERT_NE(labels, nullptr);
  std::fill(labels, labels + 6 * 4, 0.f);
  for (int i = 0; i < 6; i++) labels[i * 4 + ll_labels[i]] = 1.0;

  std::vector<int> dim_l({6, 4});
  lite::Tensor l_tensor(TypeId::kNumberTypeInt32, dim_l);
  l_tensor.set_data(labels);

  std::vector<lite::Tensor *> inputs = {&y_tensor, &l_tensor};

  auto loss = new float[6];
  ASSERT_NE(loss, nullptr);
  std::vector<int> dim_dw({6, 1});
  lite::Tensor loss_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  loss_tensor.set_data(loss);
  auto grad = new float[24];
  ASSERT_NE(grad, nullptr);
  lite::Tensor grad_tensor(TypeId::kNumberTypeFloat32, dim_y);
  grad_tensor.set_data(grad);
  std::vector<lite::Tensor *> outputs = {&loss_tensor, &grad_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32,
                            schema::PrimitiveType_SoftmaxCrossEntropyWithLogits};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(sce_param), &context, desc);
  ASSERT_NE(kernel_obj, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel_obj->workspace_size());
  kernel_obj->Run();

  printf("==================total loss=================\n");
  std::cout << loss[0] << " ," << std::endl;

  printf("==================Testing Grad===============\n");

  std::string output_path = "./test_data/operators/sce_fp32_1_loss_1.bin";
  CompareOutput(loss, 1, output_path);

  ((mindspore::kernel::SparseSoftmaxCrossEntropyWithLogitsCPUKernel *)kernel_obj)->Train();
  kernel_obj->Run();
  // normalize by batch size the result
  for (int i = 0; i < 24; i++) {
    grad[i] /= 6;
  }
  printf("==================output data=================\n");
  for (int i = 0; i < 12; i++) {
    std::cout << grad[i] << " ,";
  }
  std::cout << std::endl;
  std::string grad_path = "./test_data/operators/sce_fp32_1_dy_6_4.bin";
  auto res = CompareRelativeOutput(grad, grad_path);
  EXPECT_EQ(res, 0);

  delete[] ll_labels;
  delete[] labels;
  delete[] input_data;
  delete[] loss;
  delete[] grad;
  l_tensor.set_data(nullptr);
  y_tensor.set_data(nullptr);
  loss_tensor.set_data(nullptr);
  grad_tensor.set_data(nullptr);
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel_obj;
  MS_LOG(INFO) << "SoftmaxCrossEntropyFp32 passed";
}

}  // namespace mindspore
