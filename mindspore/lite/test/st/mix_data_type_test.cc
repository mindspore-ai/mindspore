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
#include "src/common/config_file.h"
#include "schema/inner/model_generated.h"
#include "schema/inner/ops_generated.h"
#include "schema/ops_generated.h"
#include "schema/model_generated.h"
#include "src/lite_kernel.h"
#include "src/lite_session.h"
#include "include/api/model.h"
#include "src/cxx_api/model/model_impl.h"

namespace mindspore {
class MixDataTypeTest : public mindspore::CommonTest {
 public:
  MixDataTypeTest() {}
};

TEST_F(MixDataTypeTest, Config1) {
  auto ret = system("echo [onther_plan1] > MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo op1=data_type:float16 >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo [execution_plan] >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo op1=data_type:float32 >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo \"op2=\\\"data_type:float16\\\"\" >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo [onther_plan2] >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo op1=data_type:float16 >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);

  std::string filename = "MixDataTypeTestConfig";
  std::string sectionname = "execution_plan";
  std::map<std::string, std::string> config_info;
  ret = lite::GetSectionInfoFromConfigFile(filename, sectionname, &config_info);
  ASSERT_EQ(ret, 0);

  ASSERT_EQ(config_info.size(), 2);

  auto info0 = config_info.at("op1");
  ASSERT_EQ(info0, "data_type:float32");

  auto inf01 = config_info.at("op2");
  ASSERT_EQ(inf01, "\"data_type:float16\"");

  std::map<std::string, TypeId> execution_plan;
  lite::ParserExecutionPlan(&config_info, &execution_plan);

  ASSERT_EQ(execution_plan.size(), 2);

  auto exe_info0 = execution_plan.at("op1");
  ASSERT_EQ(exe_info0, kNumberTypeFloat32);

  auto exe_inf01 = execution_plan.at("op2");
  ASSERT_EQ(exe_inf01, kNumberTypeFloat16);
}

void ConstructConfig() {
  auto ret = system("echo [execution_plan] > MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo op1=data_type:float16 >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo op2=data_type:float32 >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  /* op3 in fp16 */
  ret = system("echo op4=data_type:float32 >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
}

void ConstructModel(schema::MetaGraphT *meta_graph) {
  meta_graph->name = "mix_data_type_graph";
  meta_graph->version = mindspore::lite::Version();

  auto cos = std::make_unique<mindspore::schema::CNodeT>();
  cos->inputIndex = {0};
  cos->outputIndex = {1};
  cos->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cos->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cos_primitive = new mindspore::schema::CosT;
  cos->primitive->value.value = cos_primitive;
  cos->name = "op1";

  auto exp = std::make_unique<mindspore::schema::CNodeT>();
  exp->inputIndex = {1};
  exp->outputIndex = {2};
  exp->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  exp->primitive->value.type = mindspore::schema::PrimitiveType_ExpFusion;
  auto exp_primitive = new mindspore::schema::ExpFusionT;
  exp->primitive->value.value = exp_primitive;
  exp->name = "op2";

  auto sin = std::make_unique<mindspore::schema::CNodeT>();
  sin->inputIndex = {2};
  sin->outputIndex = {3};
  sin->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  sin->primitive->value.type = mindspore::schema::PrimitiveType_Sin;
  auto sin_primitive = new mindspore::schema::SinT;
  sin->primitive->value.value = sin_primitive;
  sin->name = "op3";

  auto cos2 = std::make_unique<mindspore::schema::CNodeT>();
  cos2->inputIndex = {3};
  cos2->outputIndex = {4};
  cos2->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cos2->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cos2_primitive = new mindspore::schema::CosT;
  cos2->primitive->value.value = cos2_primitive;
  cos2->name = "op4";

  /* tensors */
  auto tensor0 = std::make_unique<mindspore::schema::TensorT>();
  tensor0->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor0->format = mindspore::schema::Format_NHWC;
  tensor0->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor0->dims = {1, 2, 2, 1};
  tensor0->offset = -1;
  tensor0->name = "tensor0";

  auto tensor1 = std::make_unique<mindspore::schema::TensorT>();
  tensor1->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor1->format = mindspore::schema::Format_NHWC;
  tensor1->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor1->dims = {1, 2, 2, 1};
  tensor1->offset = -1;
  tensor1->name = "tensor1";

  auto tensor2 = std::make_unique<mindspore::schema::TensorT>();
  tensor2->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor2->format = mindspore::schema::Format_NHWC;
  tensor2->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor2->dims = {1, 2, 2, 1};
  tensor2->offset = -1;
  tensor2->name = "tensor2";

  auto tensor3 = std::make_unique<mindspore::schema::TensorT>();
  tensor3->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor3->format = mindspore::schema::Format_NHWC;
  tensor3->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor3->dims = {1, 2, 2, 1};
  tensor3->offset = -1;
  tensor3->name = "tensor3";

  auto tensor4 = std::make_unique<mindspore::schema::TensorT>();
  tensor4->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor4->format = mindspore::schema::Format_NHWC;
  tensor4->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor4->dims = {1, 2, 2, 1};
  tensor4->offset = -1;
  tensor4->name = "tensor4";

  meta_graph->nodes.emplace_back(std::move(cos));
  meta_graph->nodes.emplace_back(std::move(exp));
  meta_graph->nodes.emplace_back(std::move(sin));
  meta_graph->nodes.emplace_back(std::move(cos2));

  meta_graph->allTensors.emplace_back(std::move(tensor0));
  meta_graph->allTensors.emplace_back(std::move(tensor1));
  meta_graph->allTensors.emplace_back(std::move(tensor2));
  meta_graph->allTensors.emplace_back(std::move(tensor3));
  meta_graph->allTensors.emplace_back(std::move(tensor4));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {4};
}

TEST_F(MixDataTypeTest, mix1) {
  ConstructConfig();

  size_t size;
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  ConstructModel(meta_graph.get());

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size = builder.GetSize();
  auto flat_model = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto context = std::make_shared<mindspore::Context>();
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetEnableFP16(true);
  context->MutableDeviceInfo().push_back(device_info);

  auto impl = std::make_shared<mindspore::ModelImpl>();

  auto status = impl->LoadConfig("MixDataTypeTestConfig");
  ASSERT_EQ(status, kSuccess);

  status = impl->Build(flat_model, size, kFlatBuffer, context);
  ASSERT_EQ(status, kSuccess);

  /* check */
  auto kernels = reinterpret_cast<const lite::LiteSession *>(impl->GetSession())->get_kernels();

  ASSERT_EQ(4, kernels.size());
  ASSERT_EQ(kNumberTypeFloat16, kernels.at(0)->desc().data_type);
  ASSERT_EQ(kNumberTypeFloat32, kernels.at(1)->desc().data_type);
  ASSERT_EQ(kNumberTypeFloat16, kernels.at(2)->desc().data_type);
  ASSERT_EQ(kNumberTypeFloat32, kernels.at(3)->desc().data_type);

  /* set input data */
  std::vector<mindspore::MSTensor> inputs = impl->GetInputs();
  auto in = inputs[0];
  std::vector<float> in_float = {1.0, 2.0, 3.0, 4.0};
  memcpy(in.MutableData(), in_float.data(), in.DataSize());

  std::vector<mindspore::MSTensor> outputs = impl->GetOutputs();

  impl->Predict(inputs, &outputs, nullptr, nullptr);

  /* checkout output */
  auto out = outputs[0];
  void *out_data = out.MutableData();
  float *fp32_data = reinterpret_cast<float *>(out_data);
  ASSERT_LE(fabs(fp32_data[0] - (0.549187)), 0.01);
  ASSERT_LE(fabs(fp32_data[1] - (0.818051)), 0.01);
  ASSERT_LE(fabs(fp32_data[2] - (0.934805)), 0.01);
  ASSERT_LE(fabs(fp32_data[3] - (0.879054)), 0.01);
}
}  // namespace mindspore
