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

#include "common/common_test.h"
#include "schema/ops_generated.h"
#include "src/extendrt/kernel/cuda/batchtospace.h"
#include "ut/src/extendrt/kernel/cuda/common.h"
#include "nnacl/batch_to_space_parameter.h"

namespace mindspore {
class CudaTest_BatchToSpace : public CommonTest {
 public:
  CudaTest_BatchToSpace() {}
};
namespace {
// input: [batch*block_size*block_size, height_pad/block_size, width_pad/block_size, depth]
// crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
// height = height_pad - crop_top - crop_bottom
// width = width_pad - crop_left - crop_right
OpParameter *CreateParameter() {
  auto *param = mindspore::lite::cuda::test::CreateParameter<BatchToSpaceParameter>(schema::PrimitiveType_BatchToSpace);
  param->block_shape_[0] = 2;
  param->block_shape_[1] = 2;
  param->crops_[0] = 0;
  param->crops_[1] = 0;
  param->crops_[THIRD_INPUT] = 0;
  param->crops_[FOURTH_INPUT] = 0;
  param->no_crop_ = true;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace
TEST_F(CudaTest_BatchToSpace, basic) {
  std::vector<int> input_shape{4, 1, 1, 1};
  lite::Tensor *in_tensor = new (std::nothrow) lite::Tensor(TypeId::kNumberTypeFloat32, input_shape);
  std::vector<lite::Tensor *> inputs{in_tensor};
  std::vector<int> output_shape{1, 2, 2, 1};
  lite::Tensor *out_tensor = new (std::nothrow) lite::Tensor(TypeId::kNumberTypeFloat32, output_shape);
  std::vector<lite::Tensor *> outputs{out_tensor};
  lite::InnerContext *ctx = new (std::nothrow) lite::InnerContext();
  kernel::BatchtoSpaceCudaKernel *kernel =
    new (std::nothrow) kernel::BatchtoSpaceCudaKernel(CreateParameter(), inputs, outputs, ctx);

  void *input_device_ptr = nullptr;
  cudaMalloc(&input_device_ptr, 4 * sizeof(float));
  float input_host_ptr[4]{1, 2, 3, 4};
  cudaMemcpy(input_device_ptr, input_host_ptr, 4 * sizeof(float), cudaMemcpyHostToDevice);
  in_tensor->set_data(input_device_ptr);

  void *output_device_ptr = nullptr;
  cudaMalloc(&output_device_ptr, 4 * sizeof(float));
  float output_host_ptr[4];
  out_tensor->set_data(output_device_ptr);

  kernel->Prepare();

  kernel->Run();

  cudaMemcpy(output_host_ptr, output_device_ptr, 4 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 4; i++) {
    MS_LOG(ERROR) << "BatchtoSpaceCudaKernel out: " << output_host_ptr[i];
  }

  cudaFree(output_device_ptr);
  cudaFree(input_device_ptr);
  in_tensor->set_data(nullptr);
  out_tensor->set_data(nullptr);
  delete in_tensor;
  delete out_tensor;
  delete kernel;
}
}  // namespace mindspore
