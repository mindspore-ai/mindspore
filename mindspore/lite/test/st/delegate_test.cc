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

#include "gtest/gtest.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "tools/converter/converter.h"
#include "include/api/types.h"
#include "include/api/delegate.h"
#include "include/api/model.h"

namespace mindspore {
class DelegateTest : public mindspore::CommonTest {
 public:
  DelegateTest() {}
};

class CustomSubgraph : public kernel::Kernel {
 public:
  CustomSubgraph(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr) {}

  ~CustomSubgraph() override {}

  int Init();

  int Prepare() override { return lite::RET_OK; }

  int Execute() override;

  int ReSize() override { return lite::RET_OK; }
};

int CustomSubgraph::Execute() {
  // Only a relu op in the model.
  auto in_data = reinterpret_cast<float *>(inputs_[0].MutableData());
  auto out_data = reinterpret_cast<float *>(outputs_[0].MutableData());
  for (auto i = 0; i < inputs_[0].ElementNum(); i++) {
    out_data[i] = in_data[i] >= 0 ? in_data[i] : 0.f;
  }
  return lite::RET_OK;
}

class CustomDelegate : public Delegate {
 public:
  CustomDelegate() : Delegate() {}

  ~CustomDelegate() override {}

  Status Init() override { return mindspore::kSuccess; }

  Status Build(DelegateModel<schema::Primitive> *model) override;
};

Status CustomDelegate::Build(DelegateModel<schema::Primitive> *model) {
  auto graph_kernel = new (std::nothrow) CustomSubgraph(model->inputs(), model->outputs());
  if (graph_kernel == nullptr) {
    return mindspore::kLiteNullptr;
  }
  model->Replace(model->BeginKernelIterator(), model->EndKernelIterator(), graph_kernel);
  return mindspore::kSuccess;
}

TEST_F(DelegateTest, CustomDelegate) {
  const char *converter_argv[] = {
    "./converter",
    "--fmk=TFLITE",
    "--modelFile=./relu.tflite",
    "--outputFile=./relu.tflite",
  };
  int converter_ret = mindspore::lite::RunConverter(4, converter_argv);
  ASSERT_EQ(converter_ret, lite::RET_OK);

  size_t size = 0;
  char *model_buf = lite::ReadFile("./relu.tflite.ms", &size);
  ASSERT_NE(model_buf, nullptr);

  // init context and set delegate
  auto context = std::make_shared<Context>();
  ASSERT_NE(context, nullptr);
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_list.push_back(device_info);
  auto custom_delegate = std::make_shared<CustomDelegate>();
  context->SetDelegate(custom_delegate);

  // build model
  Model *model = new Model();
  ASSERT_NE(model, nullptr);
  Status build_ret = model->Build(model_buf, size, kMindIR, context);
  ASSERT_EQ(build_ret == kSuccess, true);

  // set input data
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size() == 1, true);
  auto in_tensor = inputs[0];
  float *in_data = reinterpret_cast<float *>(malloc(in_tensor.DataSize()));
  float in[] = {12.755477, 7.647509,  14.670943, -8.03628,  -1.815172, 7.7517915, 17.758,
                -4.800611, -8.743361, 1.6797531, -0.234721, 7.7575417, 10.19116,  11.744166,
                -2.674233, 8.977257,  16.625568, -4.820712, -7.443196, -2.669484};
  memcpy(in_data, in, in_tensor.DataSize());
  in_tensor.SetData(in_data);

  // set output data
  auto outputs = model->GetOutputs();
  ASSERT_EQ(outputs.size() == 1, true);
  auto out_tensor = outputs[0];
  float *out_data = reinterpret_cast<float *>(malloc(out_tensor.DataSize()));
  out_tensor.SetData(out_data);
  out_tensor.SetAllocator(nullptr);

  // run graph
  Status predict_ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(predict_ret == kSuccess, true);

  // compare output data
  float out[] = {12.755477, 7.647509,  14.670943, 0,         0, 7.7517915, 17.758,    0, 0, 1.6797531,
                 0,         7.7575417, 10.19116,  11.744166, 0, 8.977257,  16.625568, 0, 0, 0};
  for (int i = 0; i < out_tensor.ElementNum(); i++) {
    ASSERT_LE((out[i] - out_data[i]), 0.001);
  }
  free(out_data);
  delete model;
}
}  // namespace mindspore
