/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include <memory>
#include "common/common_test.h"
#include "ops/clip.h"
#include "ops/return.h"
#include "ops/make_tuple.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/anf_transform.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "mindapi/ir/tensor.h"

namespace mindspore {
constexpr float kNumClipFloatMinValue = -1.0;
constexpr float kNumClipFloatMaxValue = 1.0;
constexpr int kNumInputMinIndex = 1;
constexpr int kNumInputMaxIndex = 2;
class ClipMapperTest : public mindspore::CommonTest {
 public:
  ClipMapperTest() = default;
};

namespace {
CNodePtr AddReturn(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &return_inputs) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (return_inputs.empty()) {
    MS_LOG(ERROR) << "return node's input is empty!";
    return nullptr;
  }
  AnfNodePtr return_input;
  if (return_inputs.size() == 1) {
    return_input = return_inputs.front();
  } else {
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return nullptr;
    }
    auto prim_c = make_tuple_prim_ptr->GetPrim();
    if (prim_c == nullptr) {
      MS_LOG(ERROR) << "prim_c is nullptr!";
      return nullptr;
    }
    auto return_input_cnode = graph->NewCNode(prim_c, return_inputs);
    if (return_input_cnode == nullptr) {
      MS_LOG(ERROR) << "new make tuple cnode failed";
      return nullptr;
    }
    return_input_cnode->set_fullname_with_scope("return tuple");
    return_input = return_input_cnode;
  }

  auto return_prim = std::make_shared<ops::Return>();
  if (return_prim == nullptr) {
    MS_LOG(ERROR) << "create return primitive failed!";
    return nullptr;
  }
  auto return_prim_c = return_prim->GetPrim();
  if (return_prim_c == nullptr) {
    MS_LOG(ERROR) << "prim_c is nullptr!";
    return nullptr;
  }
  auto return_cnode = graph->NewCNode(return_prim_c, {return_input});
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "create Return failed";
    return nullptr;
  }
  return_cnode->set_fullname_with_scope("Return");
  graph->set_return(return_cnode);
  return return_cnode;
}

CNodePtr InitClipNode(const FuncGraphPtr &func_graph, const ParameterPtr &data, const ParameterPtr &min,
                      const ParameterPtr &max) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  auto clip_prim = std::make_shared<ops::Clip>();
  if (clip_prim == nullptr) {
    MS_LOG(ERROR) << "clip prim is nullptr!";
    return nullptr;
  }
  clip_prim->set_max(kNumClipFloatMaxValue);
  clip_prim->set_min(kNumClipFloatMinValue);
  auto clip_prim_c = clip_prim->GetPrim();
  if (clip_prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, clip node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {data};
  if (min != nullptr) {
    inputs.push_back(min);
  } else {
    clip_prim_c->AddAttr("empty_input_index", MakeValue<int>(kNumInputMinIndex));
  }
  if (max != nullptr) {
    inputs.push_back(max);
  } else {
    clip_prim_c->AddAttr("empty_input_index", MakeValue<int>(kNumInputMaxIndex));
  }
  auto clip_cnode = func_graph->NewCNode(clip_prim_c, inputs);
  if (clip_cnode == nullptr) {
    MS_LOG(ERROR) << "create clip node failed, clip node is nullptr!";
    return nullptr;
  }
  clip_cnode->set_fullname_with_scope("clip_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create clip node abstract failed, abstract is nullptr!";
    return nullptr;
  }
  clip_cnode->set_abstract(abstract);
  auto ret = AddReturn(func_graph, {clip_cnode});
  if (ret == nullptr) {
    MS_LOG(ERROR) << "add return node failed!";
    return nullptr;
  }
  return clip_cnode;
}

CNodePtr InitClipNodeWithAttr(const FuncGraphPtr &func_graph, const ParameterPtr &data) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  auto clip_prim = std::make_shared<ops::Clip>();
  if (clip_prim == nullptr) {
    MS_LOG(ERROR) << "clip prim is nullptr!";
    return nullptr;
  }
  clip_prim->set_max(kNumClipFloatMaxValue);
  clip_prim->set_min(kNumClipFloatMinValue);
  auto clip_prim_c = clip_prim->GetPrim();
  if (clip_prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, clip node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {data};
  auto clip_cnode = func_graph->NewCNode(clip_prim_c, inputs);
  if (clip_cnode == nullptr) {
    MS_LOG(ERROR) << "create clip node failed, clip node is nullptr!";
    return nullptr;
  }
  clip_cnode->set_fullname_with_scope("clip_node_with_attr");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create clip node abstract failed, abstract is nullptr!";
    return nullptr;
  }
  clip_cnode->set_abstract(abstract);
  auto ret = AddReturn(func_graph, {clip_cnode});
  if (ret == nullptr) {
    MS_LOG(ERROR) << "add return node failed!";
    return nullptr;
  }
  return clip_cnode;
}
}  //  namespace

TEST_F(ClipMapperTest, FloatClipNodeWithMinAndMaxInput) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto min_param = opt::BuildFloatValueParameterNode(func_graph, kNumClipFloatMinValue, "min");
  ASSERT_NE(min_param, nullptr);
  auto max_param = opt::BuildFloatValueParameterNode(func_graph, kNumClipFloatMaxValue, "max");
  ASSERT_NE(max_param, nullptr);
  auto cnode = InitClipNode(func_graph, data_param, min_param, max_param);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameClip);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto clip_input_2 = cnode->input(2);
  auto input2_is_param = utils::isa<ParameterPtr>(clip_input_2);
  ASSERT_EQ(input2_is_param, true);
  auto clip_input_3 = cnode->input(3);
  auto input3_is_param = utils::isa<ParameterPtr>(clip_input_3);
  ASSERT_EQ(input3_is_param, true);
  MS_LOG(INFO) << "PASS";
}

TEST_F(ClipMapperTest, FloatClipNodeWithMinInput) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto min_param = opt::BuildFloatValueParameterNode(func_graph, kNumClipFloatMaxValue, "min");
  ASSERT_NE(min_param, nullptr);
  ParameterPtr max_param = nullptr;
  auto cnode = InitClipNode(func_graph, data_param, min_param, max_param);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 3);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameClip);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  MS_LOG(INFO) << "PASS";
}

TEST_F(ClipMapperTest, FloatClipNodeWithMaxInput) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  ParameterPtr min_param = nullptr;
  auto max_param = opt::BuildFloatValueParameterNode(func_graph, kNumClipFloatMaxValue, "max");
  ASSERT_NE(max_param, nullptr);
  auto cnode = InitClipNode(func_graph, data_param, min_param, max_param);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 3);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameClip);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  MS_LOG(INFO) << "PASS";
}

// input data type is int
TEST_F(ClipMapperTest, IntClipNodeWithMinAndMaxInput) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildIntValueParameterNode(func_graph, 0, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto min_param = opt::BuildIntValueParameterNode(func_graph, kNumClipFloatMinValue, "min");
  ASSERT_NE(min_param, nullptr);
  auto max_param = opt::BuildIntValueParameterNode(func_graph, kNumClipFloatMaxValue, "max");
  ASSERT_NE(max_param, nullptr);
  auto cnode = InitClipNode(func_graph, data_param, min_param, max_param);
  ASSERT_NE(cnode, nullptr);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameClip);
  ASSERT_NE(mapper, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto clip_input_2 = cnode->input(2);
  auto input2_is_param = utils::isa<ParameterPtr>(clip_input_2);
  ASSERT_EQ(input2_is_param, true);
  auto clip_input_3 = cnode->input(3);
  auto input3_is_param = utils::isa<ParameterPtr>(clip_input_3);
  ASSERT_EQ(input3_is_param, true);
  MS_LOG(INFO) << "PASS";
}

TEST_F(ClipMapperTest, IntClipNodeWithMinInput) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildIntValueParameterNode(func_graph, 0, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto min_param = opt::BuildIntValueParameterNode(func_graph, kNumClipFloatMaxValue, "min");
  ASSERT_NE(min_param, nullptr);
  ParameterPtr max_param = nullptr;
  auto cnode = InitClipNode(func_graph, data_param, min_param, max_param);
  ASSERT_NE(cnode, nullptr);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameClip);
  ASSERT_NE(mapper, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 3);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto clip_input_2 = cnode->input(2);
  auto input_2_is_cast = opt::CheckPrimitiveType(clip_input_2, prim::kPrimCast);
  ASSERT_EQ(input_2_is_cast, true);
  auto clip_input_3 = cnode->input(3);
  auto input_3_is_cast = opt::CheckPrimitiveType(clip_input_2, prim::kPrimCast);
  ASSERT_EQ(input_3_is_cast, true);
  MS_LOG(INFO) << "PASS";
}

TEST_F(ClipMapperTest, IntClipNodeWithMaxInput) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildIntValueParameterNode(func_graph, 0, "input_data");
  ASSERT_NE(data_param, nullptr);
  ParameterPtr min_param = nullptr;
  auto max_param = opt::BuildIntValueParameterNode(func_graph, kNumClipFloatMaxValue, "max");
  ASSERT_NE(max_param, nullptr);
  auto cnode = InitClipNode(func_graph, data_param, min_param, max_param);
  ASSERT_NE(cnode, nullptr);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameClip);
  ASSERT_NE(mapper, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 3);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto clip_input_2 = cnode->input(2);
  auto input_2_is_cast = opt::CheckPrimitiveType(clip_input_2, prim::kPrimCast);
  ASSERT_EQ(input_2_is_cast, true);
  auto clip_input_3 = cnode->input(3);
  auto input_3_is_cast = opt::CheckPrimitiveType(clip_input_2, prim::kPrimCast);
  ASSERT_EQ(input_3_is_cast, true);
  MS_LOG(INFO) << "PASS";
}

// clip by attr
TEST_F(ClipMapperTest, FloatClipNodeWithAttr) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  ParameterPtr min_param = nullptr;
  ParameterPtr max_param = nullptr;
  auto cnode = InitClipNodeWithAttr(func_graph, data_param);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameClip);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto clip_input_2 = cnode->input(2);
  auto input2_is_param = utils::isa<ParameterPtr>(clip_input_2);
  ASSERT_EQ(input2_is_param, true);
  auto clip_input_3 = cnode->input(3);
  auto input3_is_param = utils::isa<ParameterPtr>(clip_input_3);
  ASSERT_EQ(input3_is_param, true);
  MS_LOG(INFO) << "PASS";
}
}  // namespace mindspore
