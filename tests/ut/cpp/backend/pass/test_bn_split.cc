/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "common/graph_optimizer_test_framework.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "common/common_test.h"
#include "backend/common/pass/ir_fission/bn_split.h"
#include "backend/common/pass/ir_fission/bn_grad_split.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"

namespace mindspore {
class BnSplit : public UT::Common {
 public:
  BnSplit() {}
};

/// Feature: A backend unify fission pass: BnSplit
/// Description: Convert BatchNorm to BNTrainingReduce+BNTrainingUpdate
/// Expectation: After optimize, match BNTrainingReduce+BNTrainingUpdate
TEST_F(BnSplit, test_bn_split) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat32, {4, 8, 16, 32});
  auto scale = c.NewTensorInput("scale", kFloat32, {8});
  auto bias = c.NewTensorInput("bias", kFloat32, {8});
  auto mean = c.NewTensorInput("mean", kFloat32, {8});
  auto variance = c.NewTensorInput("variance", kFloat32, {8});
  auto is_training = c.NewValueNode(MakeValue<bool>(true));
  auto epsilon = c.NewValueNode(MakeValue<float>(1e-5));
  auto momentum = c.NewValueNode(MakeValue<float>(0.1));
  auto data_format = c.NewValueNode(MakeValue<int64_t>(0));
  auto node =
    c.NewCNode("BatchNorm", {input, scale, bias, mean, variance, is_training, epsilon, momentum, data_format});

  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::BnSplit>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("scale")
    .AddVar("bias")
    .AddVar("mean")
    .AddVar("variance")
    .AddVar("is_training")
    .AddVar("epsilon")
    .AddVar("momentum")
    .AddVar("data_format")
    .AddVar("get1")
    .AddVar("get2")
    .AddCNode("bn_reduce", {std::make_shared<Primitive>("BNTrainingReduce"), "input", "data_format"})
    .AddCNode("get_item1", {std::make_shared<Primitive>("TupleGetItem"), "bn_reduce", "get1"})
    .AddCNode("get_item2", {std::make_shared<Primitive>("TupleGetItem"), "bn_reduce", "get2"})
    .AddCNode("bn_update", {std::make_shared<Primitive>("BNTrainingUpdate"), "input", "get_item1", "get_item2", "scale",
                            "bias", "mean", "variance", "data_format"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify fission pass: SyncBnSplit
/// Description: Convert BatchNorm to BNTrainingReduce+Cast+AllReduce+Mul+Cast+BNTrainingUpdate
/// Expectation: After optimize, match BNTrainingReduce+Cast+AllReduce+Mul+Cast+BNTrainingUpdate
TEST_F(BnSplit, test_sync_bn_split) {
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat16, {4, 8, 16, 32});
  auto scale = c.NewTensorInput("scale", kFloat16, {8});
  auto bias = c.NewTensorInput("bias", kFloat16, {8});
  auto mean = c.NewTensorInput("mean", kFloat16, {8});
  auto variance = c.NewTensorInput("variance", kFloat16, {8});
  auto node = c.NewCNode("SyncBatchNorm", {input, scale, bias, mean, variance},
                         {{"epsilon", MakeValue<float>(1e-5)},
                          {"momentum", MakeValue<float>(0.1)},
                          {"format", MakeValue<std::string>("NCHW")},
                          {"group", MakeValue<std::string>("hccl_world_group")},
                          {"device_num", MakeValue<int64_t>(8)}});

  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::SyncBnSplit>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input")
    .AddVar("scale")
    .AddVar("bias")
    .AddVar("mean")
    .AddVar("variance")
    .AddVar("data_format1")
    .AddVar("data_format2")
    .AddVar("get1")
    .AddVar("get2")
    .AddVar("num1")
    .AddVar("num2")
    .AddCNode("bn_reduce", {std::make_shared<Primitive>("BNTrainingReduce"), "input", "data_format1"})
    .AddCNode("get_item1", {std::make_shared<Primitive>("TupleGetItem"), "bn_reduce", "get1"})
    .AddCNode("get_item2", {std::make_shared<Primitive>("TupleGetItem"), "bn_reduce", "get2"})
    .AddCNode("cast1", {std::make_shared<Primitive>("Cast"), "get_item1"})
    .AddCNode("cast2", {std::make_shared<Primitive>("Cast"), "get_item2"})
    .AddCNode("allreduce1", {std::make_shared<Primitive>("AllReduce"), "cast1"})
    .AddCNode("allreduce2", {std::make_shared<Primitive>("AllReduce"), "cast2"})
    .AddCNode("mul1", {std::make_shared<Primitive>("Mul"), "allreduce1", "num1"})
    .AddCNode("mul2", {std::make_shared<Primitive>("Mul"), "allreduce2", "num2"})
    .AddCNode("cast3", {std::make_shared<Primitive>("Cast"), "mul1"})
    .AddCNode("cast4", {std::make_shared<Primitive>("Cast"), "mul2"})
    .AddCNode("bn_update", {std::make_shared<Primitive>("BNTrainingUpdate"), "input", "cast3", "cast4", "scale", "bias",
                            "mean", "variance", "data_format2"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify fission pass: BnGradSplit
/// Description: Convert BatchNorm to BNTrainingReduceGrad+BNTrainingUpdateGrad
/// Expectation: After optimize, match BNTrainingReduceGrad+BNTrainingUpdateGrad
TEST_F(BnSplit, test_bn_grad_split) {
  test::ConstructGraph c;
  auto dout = c.NewTensorInput("dout", kFloat32, {4, 8, 16, 32});
  auto input = c.NewTensorInput("input", kFloat32, {4, 8, 16, 32});
  auto gamma = c.NewTensorInput("gamma", kFloat32, {8});
  auto bmean = c.NewTensorInput("bmean", kFloat32, {8});
  auto bvar = c.NewTensorInput("bvar", kFloat32, {8});
  auto temp = c.NewTensorInput("temp", kFloat32, {8});
  auto is_training = c.NewValueNode(MakeValue<bool>(true));
  auto epsilon = c.NewValueNode(MakeValue<float>(1e-5));
  auto data_format = c.NewValueNode(MakeValue<int64_t>(0));
  auto node = c.NewCNode("BatchNormGrad", {dout, input, gamma, bmean, bvar, temp, is_training, epsilon, data_format},
                         {{"epsilon", MakeValue<float>(1e-5)}, {"format", MakeValue<int64_t>(0)}});

  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::BnGradSplit>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("dout")
    .AddVar("input")
    .AddVar("gamma")
    .AddVar("bmean")
    .AddVar("bvar")
    .AddVar("is_training")
    .AddVar("epsilon")
    .AddVar("momentum")
    .AddVar("data_format")
    .AddVar("get1")
    .AddVar("get2")
    .AddCNode("bn_update_grad", {std::make_shared<Primitive>("BNTrainingUpdateGrad"), "dout", "input", "bmean", "bvar"})
    .AddCNode("get_item1", {std::make_shared<Primitive>("TupleGetItem"), "bn_update_grad", "get1"})
    .AddCNode("get_item2", {std::make_shared<Primitive>("TupleGetItem"), "bn_update_grad", "get2"})
    .AddCNode("bn_reduce_grad", {std::make_shared<Primitive>("BNTrainingReduceGrad"), "dout", "input", "get_item1",
                                 "get_item2", "gamma", "bmean", "bvar"})
    .AddCNode("maketuple", {std::make_shared<Primitive>("MakeTuple"), "bn_reduce_grad", "get_item1", "get_item2"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend unify fission pass: SyncBnGradSplit
/// Description: Convert BatchNorm to BNTrainingReduceGrad+AllReduce+Mul+BNTrainingUpdateGrad
/// Expectation: After optimize, match BNTrainingReduceGrad+AllReduce+Mul+BNTrainingUpdateGrad
TEST_F(BnSplit, test_sync_bn_grad_split) {
  test::ConstructGraph c;
  auto dout = c.NewTensorInput("dout", kFloat32, {4, 8, 16, 32});
  auto input = c.NewTensorInput("input", kFloat32, {4, 8, 16, 32});
  auto gamma = c.NewTensorInput("gamma", kFloat32, {8});
  auto bmean = c.NewTensorInput("bmean", kFloat32, {8});
  auto bvar = c.NewTensorInput("bvar", kFloat32, {8});
  auto node = c.NewCNode("SyncBatchNormGrad", {dout, input, gamma, bmean, bvar},
                         {{"epsilon", MakeValue<float>(1e-5)},
                          {"momentum", MakeValue<float>(0.1)},
                          {"format", MakeValue<int64_t>(0)},
                          {"group", MakeValue<std::string>("hccl_world_group")},
                          {"device_num", MakeValue<int64_t>(8)}});

  c.SetOutput(node);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::SyncBnGradSplit>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("dout")
    .AddVar("input")
    .AddVar("gamma")
    .AddVar("bmean")
    .AddVar("bvar")
    .AddVar("get1")
    .AddVar("get2")
    .AddVar("num1")
    .AddVar("num2")
    .AddCNode("bn_update_grad", {std::make_shared<Primitive>("BNTrainingUpdateGrad"), "dout", "input", "bmean", "bvar"})
    .AddCNode("get_item1", {std::make_shared<Primitive>("TupleGetItem"), "bn_update_grad", "get1"})
    .AddCNode("get_item2", {std::make_shared<Primitive>("TupleGetItem"), "bn_update_grad", "get2"})
    .AddCNode("allreduce1", {std::make_shared<Primitive>("AllReduce"), "get_item1"})
    .AddCNode("allreduce2", {std::make_shared<Primitive>("AllReduce"), "get_item2"})
    .AddCNode("mul1", {std::make_shared<Primitive>("Mul"), "allreduce1", "num1"})
    .AddCNode("mul2", {std::make_shared<Primitive>("Mul"), "allreduce2", "num2"})
    .AddCNode("bn_reduce_grad", {std::make_shared<Primitive>("BNTrainingReduceGrad"), "dout", "input", "mul1", "mul2",
                                 "gamma", "bmean", "bvar"})
    .AddCNode("maketuple", {std::make_shared<Primitive>("MakeTuple"), "bn_reduce_grad", "mul1", "mul2"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace mindspore
