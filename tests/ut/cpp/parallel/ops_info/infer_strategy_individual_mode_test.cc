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

#include <string>
#include <list>
#include <vector>
#include "common/common_test.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/layer_norm_info.h"
#include "frontend/parallel/ops_info/batchnorm_info.h"
#include "frontend/parallel/ops_info/bias_add_info.h"
#include "frontend/parallel/ops_info/scatter_update_info.h"
#include "frontend/parallel/ops_info/conv2d_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class LayerNormInfo;
class BiasAddInfo;
class BatchNormInfo;
class ScatterUpdateInfo;
class Conv2DInfo;
using LayerNormInfoPtr = std::shared_ptr<LayerNormInfo>;
using BiasAddInfoPtr = std::shared_ptr<BiasAddInfo>;
using BatchNormInfoPtr = std::shared_ptr<BatchNormInfo>;
using ScatterUpdateInfoPtr = std::shared_ptr<ScatterUpdateInfo>;
using Conv2DInfoPtr = std::shared_ptr<Conv2DInfo>;
LayerNormInfoPtr layer_norm;
BiasAddInfoPtr bias_add;
BatchNormInfoPtr batch_norm;
ScatterUpdateInfoPtr scatter_update;
Conv2DInfoPtr conv2d;

class TestInferStrategyIndividualMode : public UT::Common {
 public:
  TestInferStrategyIndividualMode() {}
  void SetUp();
  void TearDown() {}
};

void TestInferStrategyIndividualMode::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 64; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(64);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  // layer_norm
  ValuePtr begin_norm_axis = MakeValue(std::int64_t(3));
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"begin_norm_axis", begin_norm_axis}};

  Shapes ln_inputs_shape = {{16, 32, 64, 96}, {32, 64, 96}, {32, 64, 96}};
  Shapes ln_outputs_shape = {{16, 32, 64, 96}, {16, 32, 1, 1}, {16, 32, 1, 1}};
  layer_norm = std::make_shared<LayerNormInfo>("layernorm_info", ln_inputs_shape, ln_outputs_shape, attr_1);

  // bias_add
  mindspore::HashMap<std::string, ValuePtr> attr_2;
  Shapes ba_inputs_shape = {{64, 96}, {96}};
  Shapes ba_outputs_shape = {{64, 96}};
  bias_add = std::make_shared<BiasAddInfo>("biasadd_info", ba_inputs_shape, ba_outputs_shape, attr_2);

  // batch_norm
  ValuePtr is_training = MakeValue(true);
  ValuePtr epsilon = MakeValue(std::float_t(1.0));
  ValuePtr momentum = MakeValue(std::float_t(1.0));
  ValuePtr format = MakeValue("NCHW");
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"is_training", is_training},
                                                      {"epsilon", epsilon},
                                                      {"momentum", momentum},
                                                      {"format", format}};

  Shapes bn_inputs_shape = {{64, 96, 32, 16}, {96}, {96}, {96}, {96}};
  Shapes bn_outputs_shape = {{64, 96, 32, 16}, {96}, {96}, {96}, {96}};
  batch_norm = std::make_shared<BatchNormInfo>("batchnorm_info", bn_inputs_shape, bn_outputs_shape, attr_3);

  // scatter_update
  mindspore::HashMap<std::string, ValuePtr> attr_4;
  Shapes su_inputs_shape = {{16, 32, 64, 96}, {128, 256}, {128, 256, 32, 64, 96}};
  Shapes su_outputs_shape = {{16, 32, 64, 96}};
  scatter_update = std::make_shared<ScatterUpdateInfo>("scatterupdate_info", su_inputs_shape, su_outputs_shape, attr_4);

  // conv2d
  ValuePtr out_channel = MakeValue(std::int64_t(10));
  ValuePtr kernel_size = MakeValue(std::vector<int64_t>{4, 4});
  ValuePtr mode = MakeValue(std::int64_t(1));
  ValuePtr pad_mode = MakeValue(std::int64_t(1));
  ValuePtr pad_list = MakeValue(std::vector<int64_t>{1, 1, 1, 1});
  ValuePtr stride = MakeValue(std::vector<int64_t>{1, 1, 2, 2});
  ValuePtr dilation = MakeValue(std::vector<int64_t>{1, 1, 1, 1});
  ValuePtr group = MakeValue(std::int64_t(1));
  mindspore::HashMap<std::string, ValuePtr> attr_5 = {{"out_channel", out_channel},
                                                      {"kernel_size", kernel_size},
                                                      {"pad_mode", pad_mode},
                                                      {"mode", mode},
                                                      {"pad_list", pad_list},
                                                      {"stride", stride},
                                                      {"dilation", dilation},
                                                      {"group", group},
                                                      {"format", format}};

  Shapes conv_inputs_shape = {{128, 2, 16, 16}, {10, 2, 4, 4}};
  Shapes conv_outputs_shape = {{128, 10, 8, 8}};
  conv2d = std::make_shared<Conv2DInfo>("conv2d_info", conv_inputs_shape, conv_outputs_shape, attr_5);
}

/// Feature: infer strategy for layer_norm
/// Description: the in strategy is {{2, 4, 8, 1}, {}, {}}
/// Expectation: the return strategy is {{2, 4, 8, 1}, {4, 8, 1}, {4, 8, 1}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy1) {
  Strategies in_strategy = {{2, 4, 8, 1}, {}, {}};
  Strategies ret = layer_norm->GenerateFullStrategy(in_strategy);

  Strategies expect = {{2, 4, 8, 1}, {4, 8, 1}, {4, 8, 1}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for layer_norm
/// Description: the in strategy is {{}, {4, 8, 1}, {4, 8, 1}}
/// Expectation: the return strategy is {{1, 4, 8, 1}, {4, 8, 1}, {4, 8, 1}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy2) {
  Strategies in_strategy = {{}, {4, 8, 1}, {4, 8, 1}};
  Strategies ret = layer_norm->GenerateFullStrategy(in_strategy);

  Strategies expect = {{1, 4, 8, 1}, {4, 8, 1}, {4, 8, 1}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for layer_norm
/// Description: the in strategy is {{}, {4, 8, 1}, {}}
/// Expectation: throw exception
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy3) {
  Strategies in_strategy = {{}, {4, 8, 1}, {}};
  ASSERT_ANY_THROW(layer_norm->GenerateFullStrategy(in_strategy));
}

/// Feature: infer strategy for bias_add
/// Description: the in strategy is {{4, 8}, {}}
/// Expectation: the return strategy is {{4, 8}, {8}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy4) {
  Strategies in_strategy = {{4, 8}, {}};
  Strategies ret = bias_add->GenerateFullStrategy(in_strategy);

  Strategies expect = {{4, 8}, {8}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for bias_add
/// Description: the in strategy is {{}, {8}}
/// Expectation: the return strategy is {{1, 8}, {8}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy5) {
  Strategies in_strategy = {{}, {8}};
  Strategies ret = bias_add->GenerateFullStrategy(in_strategy);

  Strategies expect = {{1, 8}, {8}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for batch_norm
/// Description: the in strategy is {{2, 4, 8, 16}, {}, {}, {}, {}}
/// Expectation: the return strategy is {{2, 4, 8, 16}, {4}, {4}, {4}, {4}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy6) {
  Strategies in_strategy = {{2, 4, 8, 16}, {}, {}, {}, {}};
  Strategies ret = batch_norm->GenerateFullStrategy(in_strategy);

  Strategies expect = {{2, 4, 8, 16}, {4}, {4}, {4}, {4}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for batch_norm
/// Description: the in strategy is {{}, {4}, {4}, {4}, {4}}
/// Expectation: the return strategy is {{1, 4, 1, 1}, {4}, {4}, {4}, {4}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy7) {
  Strategies in_strategy = {{}, {4}, {4}, {4}, {4}};
  Strategies ret = batch_norm->GenerateFullStrategy(in_strategy);

  Strategies expect = {{1, 4, 1, 1}, {4}, {4}, {4}, {4}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for batch_norm
/// Description: the in strategy is {{}, {4}, {}, {}, {4}}
/// Expectation: throw exception
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy8) {
  Strategies in_strategy = {{}, {4}, {}, {}, {4}};
  ASSERT_ANY_THROW(batch_norm->GenerateFullStrategy(in_strategy));
}

/// Feature: infer strategy for scatter_update
/// Description: the in strategy is {{1, 4, 8, 1}, {}, {}}
/// Expectation: the return strategy is {{1, 4, 8, 1}, {1, 1}, {1, 1, 4, 8, 1}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy9) {
  Strategies in_strategy = {{1, 4, 8, 1}, {}, {}};
  Strategies ret = scatter_update->GenerateFullStrategy(in_strategy);

  Strategies expect = {{1, 4, 8, 1}, {1, 1}, {1, 1, 4, 8, 1}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for scatter_update
/// Description: the in strategy is {{}, {1, 1}, {1, 1, 4, 8, 1}}
/// Expectation: the return strategy is {{1, 4, 8, 1}, {1, 1}, {1, 1, 4, 8, 1}}
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy10) {
  Strategies in_strategy = {{}, {1, 1}, {1, 1, 4, 8, 1}};
  Strategies ret = scatter_update->GenerateFullStrategy(in_strategy);

  Strategies expect = {{1, 4, 8, 1}, {1, 1}, {1, 1, 4, 8, 1}};
  ASSERT_EQ(ret, expect);
}

/// Feature: infer strategy for scatter_update
/// Description: the in strategy is {{}, {1, 1}, {}}
/// Expectation: throw exception
TEST_F(TestInferStrategyIndividualMode, GenerateFullStrategy11) {
  Strategies in_strategy = {{}, {1, 1}, {}};
  ASSERT_ANY_THROW(scatter_update->GenerateFullStrategy(in_strategy));
}
}  // namespace parallel
}  // namespace mindspore
