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
#include "utils/log_adapter.h"
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
  SoftmaxCrossEntropyParameter *sce_param = new SoftmaxCrossEntropyParameter();
  size_t input_size;

  std::string input_path = "./test_data/operators/sce_fp32_1_y_6_4.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_y({6, 4});
  lite::tensor::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
  y_tensor.SetData(input_data);

  std::string label_path = "./test_data/operators/sce_fp32_1_l_6.bin";
  auto ll_labels = reinterpret_cast<int64 *>(mindspore::lite::ReadFile(label_path.c_str(), &input_size));
  auto labels = new int[6];
  for (int i = 0; i < 6; i++) labels[i] = static_cast<int>(ll_labels[i]);

  std::vector<int> dim_l({6});
  lite::tensor::Tensor l_tensor(TypeId::kNumberTypeInt32, dim_l);
  l_tensor.SetData(labels);

  std::vector<lite::tensor::Tensor *> inputs = {&y_tensor, &l_tensor};

  auto loss = new float[1];
  std::vector<int> dim_dw({1});
  lite::tensor::Tensor loss_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  loss_tensor.SetData(loss);
  auto grad = new float[24];
  lite::tensor::Tensor grad_tensor(TypeId::kNumberTypeFloat32, dim_y);
  grad_tensor.SetData(grad);
  std::vector<lite::tensor::Tensor *> outputs = {&grad_tensor, &loss_tensor};

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SoftmaxCrossEntropy};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(sce_param), NULL, desc, nullptr);
  kernel_obj->Run();

  printf("==================total loss=================\n");
  std::cout << loss[0] << " ," << std::endl;

  printf("==================Testing Grad===============\n");

  std::string output_path = "./test_data/operators/sce_fp32_1_loss_1.bin";
  lite::CompareOutput(loss, output_path);

  ((mindspore::kernel::SparseSoftmaxCrossEntropyWithLogitsCPUKernel *)kernel_obj)->train();
  kernel_obj->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < 12; i++) {
    std::cout << grad[i] << " ,";
  }
  std::cout << std::endl;
  std::string grad_path = "./test_data/operators/sce_fp32_1_dy_6_4.bin";
  lite::CompareOutput(grad, grad_path);

  delete sce_param;
  l_tensor.SetData(NULL);
  y_tensor.SetData(NULL);
  MS_LOG(INFO) << "SoftmaxCrossEntropyFp32 passed";
}

}  // namespace mindspore
