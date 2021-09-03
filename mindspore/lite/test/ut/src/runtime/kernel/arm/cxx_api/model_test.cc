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
#include "common/common_test.h"
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/serialization.h"
#include "include/api/metrics/accuracy.h"

namespace mindspore {
class TestCxxApiLiteModel : public mindspore::CommonTest {
 public:
  TestCxxApiLiteModel() = default;
};

TEST_F(TestCxxApiLiteModel, test_build_context_uninitialized_FAILED) {
  Model model;
  Graph graph;

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  auto status = model.Build(GraphCell(graph), nullptr, nullptr);
  ASSERT_TRUE(status != kSuccess);
  auto err_mst = status.GetErrDescription();
  ASSERT_TRUE(err_mst.find("null") != std::string::npos);
}
TEST_F(TestCxxApiLiteModel, test_build_graph_uninitialized_FAILED) {
  Model model;
  GraphCell graph_cell;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);

  ASSERT_TRUE(model.Build(graph_cell, context, nullptr) != kSuccess);
}

TEST_F(TestCxxApiLiteModel, test_build_SUCCES) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, nullptr) == kSuccess);
}

TEST_F(TestCxxApiLiteModel, test_train_mode_FAILURE) {
  Model model;
  ASSERT_TRUE(model.SetTrainMode(true) != kSuccess);
}

TEST_F(TestCxxApiLiteModel, test_train_mode_SUCCES) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, nullptr) == kSuccess);
  ASSERT_TRUE(model.SetTrainMode(true) == kSuccess);
  ASSERT_TRUE(model.GetTrainMode() == true);
}

TEST_F(TestCxxApiLiteModel, test_outputs_FAILURE) {
  Model model;
  auto outputs = model.GetOutputs();
  ASSERT_EQ(outputs.size(), 0);
}

TEST_F(TestCxxApiLiteModel, test_outputs_SUCCESS) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, nullptr) == kSuccess);
  auto outputs = model.GetOutputs();
  ASSERT_GT(outputs.size(), 0);
}

TEST_F(TestCxxApiLiteModel, test_metrics_FAILURE) {
  Model model;
  AccuracyMetrics ac;
  ASSERT_TRUE(model.InitMetrics({&ac}) != kSuccess);
  auto metrics = model.GetMetrics();
  ASSERT_EQ(metrics.size(), 0);
}

TEST_F(TestCxxApiLiteModel, test_metrics_SUCCESS) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, nullptr) == kSuccess);
  AccuracyMetrics ac;
  ASSERT_TRUE(model.InitMetrics({&ac}) == kSuccess);
  auto metrics = model.GetMetrics();
  ASSERT_EQ(metrics.size(), 1);
}

TEST_F(TestCxxApiLiteModel, test_getparams_SUCCESS) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);
  auto train_cfg = std::make_shared<TrainCfg>();
  train_cfg->accumulate_gradients_ = true;

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);
  auto params = model.GetOptimizerParams();
  ASSERT_EQ(params.size(), 2);
  float pi = 3.141592647;
  for (size_t ix = 0; ix < params.size(); ix++) {
    static_cast<float *>(params[ix].MutableData())[0] = static_cast<float>(ix) + pi;
  }
  ASSERT_TRUE(model.SetOptimizerParams(params) == kSuccess);
  auto params1 = model.GetOptimizerParams();
  for (size_t ix = 0; ix < params1.size(); ix++) {
    ASSERT_EQ(static_cast<float *>(params1[ix].MutableData())[0], static_cast<float>(ix) + pi);
  }
  if (!params.empty()) {
    auto &param = params.at(0);
    param.SetShape({20, 20});
    param.SetDataType(DataType::kNumberTypeInt8);
  }
  ASSERT_TRUE(model.SetOptimizerParams(params) != kSuccess);

  if (!params.empty()) {
    auto &param = params.at(0);
    param.SetTensorName("failed_name");
  }
  ASSERT_TRUE(model.SetOptimizerParams(params) != kSuccess);
}

TEST_F(TestCxxApiLiteModel, test_getgrads_SUCCESS) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);
  auto train_cfg = std::make_shared<TrainCfg>();
  train_cfg->accumulate_gradients_ = true;

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);
  auto graients = model.GetGradients();
  ASSERT_EQ(graients.size(), 2);
  float pi = 3.141592647;
  for (size_t ix = 0; ix < graients.size(); ix++) {
    static_cast<float *>(graients[ix].MutableData())[0] = static_cast<float>(ix) + pi;
  }
  ASSERT_TRUE(model.ApplyGradients(graients) == kSuccess);
  if (!graients.empty()) {
    auto &param = graients.at(0);
    param.SetShape({20, 20});
  }

  ASSERT_TRUE(model.ApplyGradients(graients) != kSuccess);
  if (!graients.empty()) {
    auto &param = graients.at(0);
    param.SetTensorName("failed_name");
  }
  ASSERT_TRUE(model.ApplyGradients(graients) != kSuccess);
}

TEST_F(TestCxxApiLiteModel, test_fp32_SUCCESS) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu_context->SetEnableFP16(true);
  context->MutableDeviceInfo().push_back(cpu_context);
  auto train_cfg = std::make_shared<TrainCfg>();
  train_cfg->mix_precision_cfg_.is_raw_mix_precision_ = true;

  ASSERT_TRUE(Serialization::Load("./nets/conv_train_model.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);

  train_cfg->mix_precision_cfg_.is_raw_mix_precision_ = false;
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);

  cpu_context->SetEnableFP16(false);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);

  train_cfg->mix_precision_cfg_.is_raw_mix_precision_ = true;
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);
}

TEST_F(TestCxxApiLiteModel, test_fp16_SUCCESS) {
  Model model;
  Graph graph;
  auto context = std::make_shared<Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu_context->SetEnableFP16(true);
  context->MutableDeviceInfo().push_back(cpu_context);
  auto train_cfg = std::make_shared<TrainCfg>();
  train_cfg->mix_precision_cfg_.is_raw_mix_precision_ = true;

  ASSERT_TRUE(Serialization::Load("./nets/mix_lenet_tod.ms", ModelType::kMindIR, &graph) == kSuccess);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);

  train_cfg->mix_precision_cfg_.is_raw_mix_precision_ = false;
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);

  cpu_context->SetEnableFP16(false);
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);

  train_cfg->mix_precision_cfg_.is_raw_mix_precision_ = true;
  ASSERT_TRUE(model.Build(GraphCell(graph), context, train_cfg) == kSuccess);
}
}  // namespace mindspore
