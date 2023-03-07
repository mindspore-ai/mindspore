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
#include "utils/ms_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_mod.h"
#include "include/common/utils/anfalgo.h"
#include "common/backend_common_test.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/common/utils/utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kTupleFirstItemIndex = 0;
constexpr auto kTupleSecondItemIndex = 1;
constexpr auto kDependRealInputSize = 2;

ParameterPtr TestCreateParameter(const KernelGraphPtr &g, const std::string &name,
                                 const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(g);
  auto parameter = g->AddFvParameter(name, abstract->BuildValue());
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "Cannot add weight parameter!";
  }
  parameter->set_abstract(abstract);
  parameter->set_kernel_info(std::make_shared<device::KernelInfo>());
  return parameter;
}

CNodePtr TestCreateCNode(const KernelGraphPtr &g, const std::string &prim_name, const AnfNodePtrList &real_inputs,
                         const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(g);
  auto inputs = AnfNodePtrList{NewValueNode(std::make_shared<Primitive>(prim_name))};
  inputs.insert(inputs.end(), real_inputs.begin(), real_inputs.end());
  auto cnode = g->NewCNode(inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Cannot create cnode!";
  }
  if (prim_name != "TupleGetItem" && prim_name != "MakeTuple") {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cnode);
  }

  cnode->set_abstract(abstract);
  auto cnode_kernel_info = std::make_shared<device::KernelInfo>();
  auto anf_node = cnode->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(anf_node);
  cnode_kernel_info->set_kernel_mod(
    std::make_shared<kernel::TbeKernelMod>(std::make_shared<kernel::KernelPack>(), anf_node));
  cnode->set_kernel_info(cnode_kernel_info);
  return cnode;
}

inline abstract::AbstractTensorPtr TestCreateTensor(const TypePtr &element_type, const ShapeVector &shape) {
  return std::make_shared<abstract::AbstractTensor>(element_type, shape);
}

inline abstract::AbstractTuplePtr TestCreateTupleTensor(const std::vector<TypePtr> &element_types,
                                                        const std::vector<ShapeVector> &shapes) {
  if (element_types.size() != shapes.size()) {
    MS_LOG(ERROR) << "Sizes for element type and shape are not match.";
  }

  AbstractBasePtrList abstract_list;
  for (size_t i = 0; i < element_types.size(); ++i) {
    abstract_list.emplace_back(TestCreateTensor(element_types[i], shapes[i]));
  }

  return std::make_shared<abstract::AbstractTuple>(abstract_list);
}

inline CNodePtr TestCreateDepend(const KernelGraphPtr &g, const AnfNodePtrList &inputs) {
  MS_EXCEPTION_IF_NULL(g);
  if (inputs.size() != kDependRealInputSize) {
    MS_LOG(ERROR) << "Input size for Depnd should be 2!";
  }

  auto depend_node = g->NewCNode(std::vector<AnfNodePtr>{
    NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), inputs[0], inputs[1]});
  MS_EXCEPTION_IF_NULL(depend_node);
  return depend_node;
}

inline CNodePtr TestCreateMakeTuple(const KernelGraphPtr &g, const AnfNodePtrList &inputs) {
  MS_EXCEPTION_IF_NULL(g);
  AbstractBasePtrList abstract_list;
  for (const auto &input : inputs) {
    abstract_list.emplace_back(input->abstract());
  }
  return TestCreateCNode(g, "MakeTuple", inputs, std::make_shared<abstract::AbstractTuple>(abstract_list));
}

inline void DumpGraph(const KernelGraphPtr &g) {
  MS_EXCEPTION_IF_NULL(g);
  std::string file_name = "debug_try_down_" + std::to_string(g->graph_id()) + ".ir";
  DumpIR(file_name, g);
  DumpIRProto(g, "try_down_" + std::to_string(g->graph_id()));
}
}  // namespace

class TestDynamicShapePass : public BackendCommon {
 public:
  TestDynamicShapePass() {}
  ~TestDynamicShapePass() override = default;
};

/// Feature: Dynamic shape
/// Description: Dynamic op + inherited dynmiac op.
///   before:
///   a = Unique(%p) (1, -1)
///         |
///   b = A(a) (1, -1)
/// Expectation: Graph as following.
///   after:
///    Unique(%p) Unique_Init Unique_Infer
///      |   \     /      \    /     |
///      |   depend       depend     |
///      |                           |
///      A         A_Init  A_Infer   |
///       \       /    \    /   \    |
///        depend      depend   depend
TEST_F(TestDynamicShapePass, test_dynamic_shape_pass_0) {
  // construct before graph
  auto before_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(before_fg != nullptr);

  auto before_p = TestCreateParameter(before_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_uniq_node = TestCreateCNode(before_fg, "Unique", AnfNodePtrList{before_p},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_a_node = TestCreateCNode(before_fg, "A", AnfNodePtrList{before_uniq_node},
                                       TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  before_fg->set_output(before_a_node);

  // run pass
  DynamicShapeConvertPass(before_fg);

  // construct after graph
  auto after_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(after_fg != nullptr);
  auto after_p = TestCreateParameter(after_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_uniq_node = TestCreateCNode(after_fg, "Unique", AnfNodePtrList{after_p},
                                         TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_a_node = TestCreateCNode(after_fg, "A", AnfNodePtrList{after_uniq_node},
                                      TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));

  auto infer_uniq = dynamic_shape::GenInferNode(after_uniq_node);
  auto init_uniq = dynamic_shape::GenInitNode(after_uniq_node);

  auto infer_a = dynamic_shape::GenInferNode(after_a_node);
  auto init_a = dynamic_shape::GenInitNode(after_a_node);

  auto depend0 = TestCreateDepend(after_fg, AnfNodePtrList{init_uniq, infer_uniq});
  auto depend1 = TestCreateDepend(after_fg, AnfNodePtrList{after_uniq_node, init_uniq});
  auto depend3 = TestCreateDepend(after_fg, AnfNodePtrList{init_a, infer_a});
  auto depend4 = TestCreateDepend(after_fg, AnfNodePtrList{after_a_node, init_a});
  auto depend5 = TestCreateDepend(after_fg, AnfNodePtrList{infer_a, infer_uniq});
  auto depend6 = TestCreateDepend(after_fg, AnfNodePtrList{infer_a, after_uniq_node});
  auto make_tuple =
    TestCreateMakeTuple(after_fg, AnfNodePtrList{after_a_node, depend0, depend1, depend3, depend4, depend5, depend6});
  auto get_item = TestCreateCNode(after_fg, "TupleGetItem",
                                  AnfNodePtrList{make_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                  after_a_node->abstract());
  after_fg->set_output(get_item);

  // assert
  EXPECT_TRUE(CheckEqualGraph(after_fg, before_fg));
}

/// Feature: Dynamic shape
/// Description: General op case.
///   before:
///   a = A(%p)
/// Expectation: Graph as following.
///   after:
///     A(%p) A_Init A_Infer
///       \    /  \     /
///       depend  depend
TEST_F(TestDynamicShapePass, test_dynamic_shape_pass_1) {
  // construct before graph
  auto before_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(before_fg != nullptr);

  auto before_p = TestCreateParameter(before_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_a_node =
    TestCreateCNode(before_fg, "A", AnfNodePtrList{before_p}, TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  before_fg->set_output(before_a_node);

  // run pass
  DynamicShapeConvertPass(before_fg);

  // construct after graph
  auto after_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(after_fg != nullptr);
  auto after_p = TestCreateParameter(after_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_a_node =
    TestCreateCNode(after_fg, "A", AnfNodePtrList{after_p}, TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));

  auto infer_a = dynamic_shape::GenInferNode(after_a_node);
  auto init_a = dynamic_shape::GenInitNode(after_a_node);

  auto depend0 = TestCreateDepend(after_fg, AnfNodePtrList{init_a, infer_a});
  auto depend1 = TestCreateDepend(after_fg, AnfNodePtrList{after_a_node, init_a});

  auto make_tuple = TestCreateMakeTuple(after_fg, AnfNodePtrList{after_a_node, depend0, depend1});
  auto get_item = TestCreateCNode(after_fg, "TupleGetItem",
                                  AnfNodePtrList{make_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                  after_a_node->abstract());
  after_fg->set_output(get_item);

  // assert
  EXPECT_TRUE(CheckEqualGraph(after_fg, before_fg));
}

/// Feature: Dynamic shape
/// Description: Get item and make tuple case.
///   before:
///       D0(%p)
///       |   |
///      [0] [1]
///       |   |
///       A   |
///       \  /
///    MakeTuple
/// Expectation: Graph as following.
///   after:
///     D0_Update   D0(%p)    D0_Init  D0_Infer
///         \     / |  \ \     /   \    /  |
///        depend [0] [1] \   /   Depend   |
///                |   |  Depend           |
///                A   |          A_Infer  |
///                |   |            |  \   |
///                |   |            | Depend
///                |   |            | D0_Update
///                |   |            |  /
///                \  /            Depend
///               MakeTuple
TEST_F(TestDynamicShapePass, test_dynamic_shape_pass_2) {
  // construct before graph
  auto before_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(before_fg != nullptr);

  auto before_p = TestCreateParameter(before_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  // This Unique is used to present a multiply outputs operatoration instead of its origin meanning.
  auto before_tuple = TestCreateCNode(
    before_fg, "Unique", AnfNodePtrList{before_p},
    TestCreateTupleTensor(std::vector<TypePtr>(2, kFloat32), std::vector<ShapeVector>(2, std::vector<int64_t>{1, -1})));
  auto before_first_item = TestCreateCNode(before_fg, "TupleGetItem",
                                           AnfNodePtrList{before_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                           TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_second_item = TestCreateCNode(
    before_fg, "TupleGetItem", AnfNodePtrList{before_tuple, NewValueNode(SizeToLong(kTupleSecondItemIndex))},
    TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_a = TestCreateCNode(before_fg, "A", AnfNodePtrList{before_first_item}, before_first_item->abstract());
  auto before_make_tuple = TestCreateMakeTuple(before_fg, AnfNodePtrList{before_a, before_second_item});
  before_fg->set_output(before_make_tuple);

  // run pass
  DynamicShapeConvertPass(before_fg);

  // construct after graph
  auto after_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(after_fg != nullptr);

  auto after_p = TestCreateParameter(after_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_tuple = TestCreateCNode(
    after_fg, "Unique", AnfNodePtrList{after_p},
    TestCreateTupleTensor(std::vector<TypePtr>(2, kFloat32), std::vector<ShapeVector>(2, std::vector<int64_t>{1, -1})));
  auto after_first_item = TestCreateCNode(after_fg, "TupleGetItem",
                                          AnfNodePtrList{after_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_second_item = TestCreateCNode(after_fg, "TupleGetItem",
                                           AnfNodePtrList{after_tuple, NewValueNode(SizeToLong(kTupleSecondItemIndex))},
                                           TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_a = TestCreateCNode(after_fg, "A", AnfNodePtrList{after_first_item}, after_first_item->abstract());
  auto after_make_tuple = TestCreateMakeTuple(after_fg, AnfNodePtrList{after_a, after_second_item});

  auto infer_tuple = dynamic_shape::GenInferNode(after_tuple);
  auto init_tuple = dynamic_shape::GenInitNode(after_tuple);
  auto depend0 = TestCreateDepend(after_fg, AnfNodePtrList{init_tuple, infer_tuple});
  auto depend1 = TestCreateDepend(after_fg, AnfNodePtrList{after_tuple, init_tuple});

  auto infer_a = dynamic_shape::GenInferNode(after_a);
  auto init_a = dynamic_shape::GenInitNode(after_a);
  auto depend3 = TestCreateDepend(after_fg, AnfNodePtrList{init_a, infer_a});
  auto depend4 = TestCreateDepend(after_fg, AnfNodePtrList{after_a, init_a});
  auto depend5 = TestCreateDepend(after_fg, AnfNodePtrList{infer_a, infer_tuple});
  auto depend6 = TestCreateDepend(after_fg, AnfNodePtrList{infer_a, after_tuple});

  auto make_tuple = TestCreateMakeTuple(
    after_fg, AnfNodePtrList{after_make_tuple, depend0, depend1, depend3, depend4, depend5, depend6});
  auto get_item = TestCreateCNode(after_fg, "TupleGetItem",
                                  AnfNodePtrList{make_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                  after_make_tuple->abstract());
  after_fg->set_output(get_item);

  // assert
  EXPECT_TRUE(CheckEqualGraph(after_fg, before_fg));
}

/// Feature: Dynamic shape
/// Description: Complecate case case.
/// Expectation: Graph as expected.
TEST_F(TestDynamicShapePass, test_dynamic_shape_pass_3) {
  // construct before graph
  auto before_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(before_fg != nullptr);

  auto before_p1 = TestCreateParameter(before_fg, "p1", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_p2 = TestCreateParameter(before_fg, "p2", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_a = TestCreateCNode(before_fg, "A", AnfNodePtrList{before_p2}, before_p2->abstract());
  // This Unique is used to present a multiply outputs operatoration instead of its origin meanning.
  auto before_tuple = TestCreateCNode(
    before_fg, "Unique", AnfNodePtrList{before_p1, before_a},
    TestCreateTupleTensor(std::vector<TypePtr>(2, kFloat32), std::vector<ShapeVector>(2, std::vector<int64_t>{1, -1})));
  auto before_first_item = TestCreateCNode(before_fg, "TupleGetItem",
                                           AnfNodePtrList{before_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                           TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_second_item = TestCreateCNode(
    before_fg, "TupleGetItem", AnfNodePtrList{before_tuple, NewValueNode(SizeToLong(kTupleSecondItemIndex))},
    TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));

  auto before_b = TestCreateCNode(before_fg, "B", AnfNodePtrList{before_first_item}, before_first_item->abstract());
  auto before_c = TestCreateCNode(before_fg, "C", AnfNodePtrList{before_b}, before_b->abstract());

  auto before_d = TestCreateCNode(before_fg, "D", AnfNodePtrList{before_second_item}, before_second_item->abstract());
  // This Unique is used to present a single outputs operatoration instead of its origin meanning,
  // and it take dynamic shape but general general shape.
  auto before_dync_end = TestCreateCNode(before_fg, "Unique", AnfNodePtrList{before_d},
                                         TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_e = TestCreateCNode(before_fg, "E", AnfNodePtrList{before_dync_end}, before_dync_end->abstract());

  auto before_make_tuple = TestCreateMakeTuple(before_fg, AnfNodePtrList{before_c, before_e});
  before_fg->set_output(before_make_tuple);

  // run pass
  DynamicShapeConvertPass(before_fg);

  // construct after graph
  auto after_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(after_fg != nullptr);

  auto after_p1 = TestCreateParameter(after_fg, "p1", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_p2 = TestCreateParameter(after_fg, "p2", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_a = TestCreateCNode(after_fg, "A", AnfNodePtrList{after_p2}, after_p2->abstract());
  // This Unique is used to present a multiply outputs operatoration instead of its origin meanning.
  auto after_tuple = TestCreateCNode(
    after_fg, "Unique", AnfNodePtrList{after_p1, after_a},
    TestCreateTupleTensor(std::vector<TypePtr>(2, kFloat32), std::vector<ShapeVector>(2, std::vector<int64_t>{1, -1})));
  auto after_first_item = TestCreateCNode(after_fg, "TupleGetItem",
                                          AnfNodePtrList{after_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_second_item = TestCreateCNode(after_fg, "TupleGetItem",
                                           AnfNodePtrList{after_tuple, NewValueNode(SizeToLong(kTupleSecondItemIndex))},
                                           TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));

  auto after_b = TestCreateCNode(after_fg, "B", AnfNodePtrList{after_first_item}, after_first_item->abstract());
  auto after_c = TestCreateCNode(after_fg, "C", AnfNodePtrList{after_b}, after_b->abstract());

  auto after_d = TestCreateCNode(after_fg, "D", AnfNodePtrList{after_second_item}, after_second_item->abstract());
  // This Unique is used to present a single outputs operatoration instead of its origin meanning,
  // and it take dynamic shape but general general shape.
  auto after_dync_end = TestCreateCNode(after_fg, "Unique", AnfNodePtrList{after_d},
                                        TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_e = TestCreateCNode(after_fg, "E", AnfNodePtrList{after_dync_end}, after_dync_end->abstract());

  auto after_make_tuple = TestCreateMakeTuple(after_fg, AnfNodePtrList{after_c, after_e});

  auto infer_a = dynamic_shape::GenInferNode(after_a);
  auto init_a = dynamic_shape::GenInitNode(after_a);

  auto infer_tuple = dynamic_shape::GenInferNode(after_tuple);
  auto init_tuple = dynamic_shape::GenInitNode(after_tuple);

  auto infer_b = dynamic_shape::GenInferNode(after_b);
  auto init_b = dynamic_shape::GenInitNode(after_b);

  auto infer_c = dynamic_shape::GenInferNode(after_c);
  auto init_c = dynamic_shape::GenInitNode(after_c);

  auto infer_d = dynamic_shape::GenInferNode(after_d);
  auto init_d = dynamic_shape::GenInitNode(after_d);

  auto infer_de = dynamic_shape::GenInferNode(after_dync_end);
  auto init_de = dynamic_shape::GenInitNode(after_dync_end);

  auto infer_e = dynamic_shape::GenInferNode(after_e);
  auto init_e = dynamic_shape::GenInitNode(after_e);

  auto depend0 = TestCreateDepend(after_fg, AnfNodePtrList{init_a, infer_a});
  auto depend1 = TestCreateDepend(after_fg, AnfNodePtrList{after_a, init_a});

  auto depend2 = TestCreateDepend(after_fg, AnfNodePtrList{init_tuple, infer_tuple});
  auto depend3 = TestCreateDepend(after_fg, AnfNodePtrList{after_tuple, init_tuple});
  auto depend5 = TestCreateDepend(after_fg, AnfNodePtrList{infer_tuple, infer_a});

  auto depend6 = TestCreateDepend(after_fg, AnfNodePtrList{init_b, infer_b});
  auto depend7 = TestCreateDepend(after_fg, AnfNodePtrList{after_b, init_b});
  auto depend8 = TestCreateDepend(after_fg, AnfNodePtrList{infer_b, infer_tuple});
  auto depend9 = TestCreateDepend(after_fg, AnfNodePtrList{infer_b, after_tuple});

  auto depend10 = TestCreateDepend(after_fg, AnfNodePtrList{init_c, infer_c});
  auto depend11 = TestCreateDepend(after_fg, AnfNodePtrList{after_c, init_c});
  auto depend12 = TestCreateDepend(after_fg, AnfNodePtrList{infer_c, infer_b});

  auto depend13 = TestCreateDepend(after_fg, AnfNodePtrList{init_d, infer_d});
  auto depend14 = TestCreateDepend(after_fg, AnfNodePtrList{after_d, init_d});
  auto depend15 = TestCreateDepend(after_fg, AnfNodePtrList{infer_d, infer_tuple});
  auto depend16 = TestCreateDepend(after_fg, AnfNodePtrList{infer_d, after_tuple});

  auto depend17 = TestCreateDepend(after_fg, AnfNodePtrList{init_de, infer_de});
  auto depend18 = TestCreateDepend(after_fg, AnfNodePtrList{after_dync_end, init_de});
  auto depend20 = TestCreateDepend(after_fg, AnfNodePtrList{infer_de, infer_d});

  auto depend21 = TestCreateDepend(after_fg, AnfNodePtrList{init_e, infer_e});
  auto depend22 = TestCreateDepend(after_fg, AnfNodePtrList{after_e, init_e});
  auto depend23 = TestCreateDepend(after_fg, AnfNodePtrList{infer_e, infer_de});
  auto depend24 = TestCreateDepend(after_fg, AnfNodePtrList{infer_e, after_dync_end});

  auto make_tuple = TestCreateMakeTuple(
    after_fg, AnfNodePtrList{after_make_tuple, depend0,  depend1,  depend2,  depend3,  depend5,  depend6,  depend7,
                             depend8,          depend9,  depend10, depend11, depend12, depend13, depend14, depend15,
                             depend16,         depend17, depend18, depend20, depend21, depend22, depend23, depend24});
  auto get_item = TestCreateCNode(after_fg, "TupleGetItem",
                                  AnfNodePtrList{make_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                  after_make_tuple->abstract());
  after_fg->set_output(get_item);

  // assert
  EXPECT_TRUE(CheckEqualGraph(after_fg, before_fg));
}

/// Feature: Dynamic shape
/// Description: Dynamic op + depend.
///   before:
///     a = Unique(%p) (1, -1)
///           |
///       b = A(a) (1, -1)  B(%p) (1, -1)
///                 \      /
///                  depend
/// Expectation: Graph as following.
///   after:
///     Unique_Update Unique(%p) Unique_Init Unique_Infer
///              \     /  |   \     /      \    /     |
///              depend   |   depend       depend     |
///                       |                           |
///              B        A         A_Init  A_Infer   |
///               \     /  \       /    \    /   \    |
///               depend    depend      depend   depend
TEST_F(TestDynamicShapePass, test_dynamic_shape_pass_with_depend) {
  // construct before graph
  auto before_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(before_fg != nullptr);

  auto before_p = TestCreateParameter(before_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_uniq_node = TestCreateCNode(before_fg, "Unique", AnfNodePtrList{before_p},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_a_node = TestCreateCNode(before_fg, "A", AnfNodePtrList{before_uniq_node},
                                       TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_b_node =
    TestCreateCNode(before_fg, "B", AnfNodePtrList{before_p}, TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_depend_node = TestCreateDepend(before_fg, AnfNodePtrList{before_a_node, before_b_node});
  before_fg->set_output(before_depend_node);

  // run pass
  DynamicShapeConvertPass(before_fg);

  // construct after graph
  auto after_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(after_fg != nullptr);
  auto after_p = TestCreateParameter(after_fg, "p", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_uniq_node = TestCreateCNode(after_fg, "Unique", AnfNodePtrList{after_p},
                                         TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_a_node = TestCreateCNode(after_fg, "A", AnfNodePtrList{after_uniq_node},
                                      TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_b_node =
    TestCreateCNode(after_fg, "B", AnfNodePtrList{after_p}, TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_depend_node = TestCreateDepend(after_fg, AnfNodePtrList{after_a_node, after_b_node});

  auto infer_uniq = dynamic_shape::GenInferNode(after_uniq_node);
  auto init_uniq = dynamic_shape::GenInitNode(after_uniq_node);

  auto infer_a = dynamic_shape::GenInferNode(after_a_node);
  auto init_a = dynamic_shape::GenInitNode(after_a_node);

  auto infer_b = dynamic_shape::GenInferNode(after_b_node);
  auto init_b = dynamic_shape::GenInitNode(after_b_node);

  auto depend0 = TestCreateDepend(after_fg, AnfNodePtrList{init_uniq, infer_uniq});
  auto depend1 = TestCreateDepend(after_fg, AnfNodePtrList{after_uniq_node, init_uniq});
  auto depend3 = TestCreateDepend(after_fg, AnfNodePtrList{init_a, infer_a});
  auto depend4 = TestCreateDepend(after_fg, AnfNodePtrList{after_a_node, init_a});
  auto depend5 = TestCreateDepend(after_fg, AnfNodePtrList{infer_a, infer_uniq});
  auto depend6 = TestCreateDepend(after_fg, AnfNodePtrList{infer_a, after_uniq_node});
  auto depend7 = TestCreateDepend(after_fg, AnfNodePtrList{init_b, infer_b});
  auto depend8 = TestCreateDepend(after_fg, AnfNodePtrList{after_b_node, init_b});

  auto make_tuple = TestCreateMakeTuple(after_fg, AnfNodePtrList{after_depend_node, depend0, depend1, depend3, depend4,
                                                                 depend5, depend6, depend7, depend8});
  auto get_item = TestCreateCNode(after_fg, "TupleGetItem",
                                  AnfNodePtrList{make_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                  after_a_node->abstract());
  after_fg->set_output(get_item);

  // assert
  EXPECT_TRUE(CheckEqualGraph(after_fg, before_fg));
}

/// Feature: Dynamic shape
/// Description: Dynamic op + monad.
///   before:
///                       u = kUMond()
///                       /         |
///     a = Assign(v, x1, u) -- c = UpdateState(u, a) -- d = Load(v, c)
///                                    \                   /
///                                     e = UpdateState(c,d)  -- Unique(%p)
///                                                       \        /
///                                                          depend
/// Expectation: Graph as following.
///   after:
///                               u = kUMond()
///                               /        |
///   Assign_Infer Assign_Init Assign -- UpdateState(u, a) --  Load(v, c)
///            \     /      \     /          \                   /
///             depend       depend          e = UpdateState(c,d)  -- Unique(%p) Unique_Init Unique_Infer
///                                                       \        /   |  \     /    \     /
///                                                         depend     |   depend     depend
///                                                                    |   Unique_Update
///                                                                    |      /
///                                                                     depend
TEST_F(TestDynamicShapePass, test_dynamic_shape_pass_with_monad) {
  // construct before graph
  auto before_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(before_fg != nullptr);

  auto before_v = TestCreateParameter(before_fg, "v", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_x1 = TestCreateParameter(before_fg, "x1", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto before_u = NewValueNode(kUMonad);
  auto before_assign = TestCreateCNode(before_fg, "Assign", AnfNodePtrList{before_v, before_x1, before_u},
                                       TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto before_update_state_1 =
    TestCreateCNode(before_fg, "UpdateState", AnfNodePtrList{before_u, before_assign}, kUMonad->ToAbstract());
  auto before_load =
    TestCreateCNode(before_fg, "Load", AnfNodePtrList{before_v, before_update_state_1}, before_v->abstract());
  auto before_update_state_2 = TestCreateCNode(
    before_fg, "UpdateState", AnfNodePtrList{before_update_state_1, before_load}, kUMonad->ToAbstract());
  auto before_uniq_node = TestCreateCNode(before_fg, "Unique", AnfNodePtrList{before_load},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));

  auto before_depend_node = TestCreateDepend(before_fg, AnfNodePtrList{before_uniq_node, before_update_state_2});
  before_fg->set_output(before_depend_node);

  // run pass
  DynamicShapeConvertPass(before_fg);

  // construct after graph
  auto after_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(after_fg != nullptr);
  auto after_v = TestCreateParameter(after_fg, "v", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_x1 = TestCreateParameter(after_fg, "x1", TestCreateTensor(kFloat32, std::vector<int64_t>{1, 10}));
  auto after_u = NewValueNode(kUMonad);
  auto after_assign = TestCreateCNode(after_fg, "Assign", AnfNodePtrList{after_v, after_x1, after_u},
                                      TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));
  auto after_update_state_1 =
    TestCreateCNode(after_fg, "UpdateState", AnfNodePtrList{after_u, after_assign}, kUMonad->ToAbstract());
  auto after_load =
    TestCreateCNode(after_fg, "Load", AnfNodePtrList{after_v, after_update_state_1}, after_v->abstract());
  auto after_update_state_2 =
    TestCreateCNode(after_fg, "UpdateState", AnfNodePtrList{after_update_state_1, after_load}, kUMonad->ToAbstract());
  auto after_uniq_node = TestCreateCNode(after_fg, "Unique", AnfNodePtrList{after_load},
                                         TestCreateTensor(kFloat32, std::vector<int64_t>{1, -1}));

  auto after_depend_node = TestCreateDepend(after_fg, AnfNodePtrList{after_uniq_node, after_update_state_2});

  auto infer_uniq = dynamic_shape::GenInferNode(after_uniq_node);
  auto init_uniq = dynamic_shape::GenInitNode(after_uniq_node);

  auto infer_assign = dynamic_shape::GenInferNode(after_assign);
  auto init_assign = dynamic_shape::GenInitNode(after_assign);

  auto depend0 = TestCreateDepend(after_fg, AnfNodePtrList{init_assign, infer_assign});
  auto depend1 = TestCreateDepend(after_fg, AnfNodePtrList{after_assign, init_assign});
  auto depend2 = TestCreateDepend(after_fg, AnfNodePtrList{init_uniq, infer_uniq});
  auto depend3 = TestCreateDepend(after_fg, AnfNodePtrList{after_uniq_node, init_uniq});

  auto make_tuple =
    TestCreateMakeTuple(after_fg, AnfNodePtrList{after_depend_node, depend0, depend1, depend2, depend3});
  auto get_item = TestCreateCNode(after_fg, "TupleGetItem",
                                  AnfNodePtrList{make_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                  after_uniq_node->abstract());
  after_fg->set_output(get_item);

  // assert
  EXPECT_TRUE(CheckEqualGraph(after_fg, before_fg));
}

/// Feature: Dynamic shape
/// Description: Need sync case(contain op such as Tile...).
/// Expectation: Graph as expected.
TEST_F(TestDynamicShapePass, test_dynamic_shape_pass_sync) {
  // construct before graph
  auto before_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(before_fg != nullptr);

  const auto &kTile = prim::kPrimTile->name();

  auto before_p1 = TestCreateParameter(before_fg, "p1", TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));
  auto before_p2 = TestCreateParameter(before_fg, "p2", TestCreateTensor(kFloat32, std::vector<int64_t>{2}));
  auto before_uniq_node = TestCreateCNode(before_fg, "Unique", AnfNodePtrList{before_p1},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{2, -1}));
  auto before_tile1_node = TestCreateCNode(before_fg, kTile, AnfNodePtrList{before_uniq_node},
                                           TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));
  auto before_a_node =
    TestCreateCNode(before_fg, "A", AnfNodePtrList{before_p2}, TestCreateTensor(kFloat32, std::vector<int64_t>{2}));
  auto before_b_node =
    TestCreateCNode(before_fg, "B", AnfNodePtrList{before_a_node}, TestCreateTensor(kFloat32, std::vector<int64_t>{2}));
  auto before_tile2_node = TestCreateCNode(before_fg, kTile, AnfNodePtrList{before_a_node, before_b_node},
                                           TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));
  auto before_add_node = TestCreateCNode(before_fg, "Add", AnfNodePtrList{before_tile1_node, before_tile2_node},
                                         TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));
  before_fg->set_output(before_add_node);

  // run pass
  DynamicShapeConvertPass(before_fg);

  // construct after graph
  auto after_fg = std::make_shared<session::KernelGraph>();
  ASSERT_TRUE(after_fg != nullptr);

  auto after_p1 = TestCreateParameter(after_fg, "p1", TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));
  auto after_p2 = TestCreateParameter(after_fg, "p2", TestCreateTensor(kFloat32, std::vector<int64_t>{2}));
  auto after_uniq_node = TestCreateCNode(after_fg, "Unique", AnfNodePtrList{after_p1},
                                         TestCreateTensor(kFloat32, std::vector<int64_t>{2, -1}));
  auto after_tile1_node = TestCreateCNode(after_fg, kTile, AnfNodePtrList{after_uniq_node},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));
  auto after_a_node =
    TestCreateCNode(after_fg, "A", AnfNodePtrList{after_p2}, TestCreateTensor(kFloat32, std::vector<int64_t>{2}));
  auto after_b_node =
    TestCreateCNode(after_fg, "B", AnfNodePtrList{after_a_node}, TestCreateTensor(kFloat32, std::vector<int64_t>{2}));
  auto after_tile2_node = TestCreateCNode(after_fg, kTile, AnfNodePtrList{after_a_node, after_b_node},
                                          TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));
  auto after_add_node = TestCreateCNode(after_fg, "Add", AnfNodePtrList{after_tile1_node, after_tile2_node},
                                        TestCreateTensor(kFloat32, std::vector<int64_t>{2, 10}));

  auto infer_uniq = dynamic_shape::GenInferNode(after_uniq_node);
  auto init_uniq = dynamic_shape::GenInitNode(after_uniq_node);

  auto infer_tile1 = dynamic_shape::GenInferNode(after_tile1_node);
  auto init_tile1 = dynamic_shape::GenInitNode(after_tile1_node);

  auto infer_a = dynamic_shape::GenInferNode(after_a_node);
  auto init_a = dynamic_shape::GenInitNode(after_a_node);

  auto infer_b = dynamic_shape::GenInferNode(after_b_node);
  auto init_b = dynamic_shape::GenInitNode(after_b_node);

  auto infer_tile2 = dynamic_shape::GenInferNode(after_tile2_node);
  auto init_tile2 = dynamic_shape::GenInitNode(after_tile2_node);

  auto infer_add = dynamic_shape::GenInferNode(after_add_node);
  auto init_add = dynamic_shape::GenInitNode(after_add_node);

  auto depend0 = TestCreateDepend(after_fg, AnfNodePtrList{init_uniq, infer_uniq});
  auto depend1 = TestCreateDepend(after_fg, AnfNodePtrList{after_uniq_node, init_uniq});

  auto depend3 = TestCreateDepend(after_fg, AnfNodePtrList{init_tile1, infer_tile1});
  auto depend4 = TestCreateDepend(after_fg, AnfNodePtrList{after_tile1_node, init_tile1});
  auto depend5 = TestCreateDepend(after_fg, AnfNodePtrList{infer_tile1, infer_uniq});
  auto depend6 = TestCreateDepend(after_fg, AnfNodePtrList{infer_tile1, after_uniq_node});

  auto depend7 = TestCreateDepend(after_fg, AnfNodePtrList{init_a, infer_a});
  auto depend8 = TestCreateDepend(after_fg, AnfNodePtrList{after_a_node, init_a});

  auto depend9 = TestCreateDepend(after_fg, AnfNodePtrList{init_b, infer_b});
  auto depend10 = TestCreateDepend(after_fg, AnfNodePtrList{after_b_node, init_b});
  auto depend11 = TestCreateDepend(after_fg, AnfNodePtrList{infer_b, infer_a});

  auto depend12 = TestCreateDepend(after_fg, AnfNodePtrList{init_tile2, infer_tile2});
  auto depend13 = TestCreateDepend(after_fg, AnfNodePtrList{after_tile2_node, init_tile2});
  auto depend14 = TestCreateDepend(after_fg, AnfNodePtrList{infer_tile2, infer_a});
  auto depend15 = TestCreateDepend(after_fg, AnfNodePtrList{infer_tile2, infer_b});
  auto depend17 = TestCreateDepend(after_fg, AnfNodePtrList{infer_tile2, after_b_node});

  auto depend18 = TestCreateDepend(after_fg, AnfNodePtrList{init_add, infer_add});
  auto depend19 = TestCreateDepend(after_fg, AnfNodePtrList{after_add_node, init_add});
  auto depend20 = TestCreateDepend(after_fg, AnfNodePtrList{infer_add, infer_tile1});
  auto depend21 = TestCreateDepend(after_fg, AnfNodePtrList{infer_add, infer_tile2});

  auto make_tuple = TestCreateMakeTuple(
    after_fg, AnfNodePtrList{after_add_node, depend0,  depend1,  depend3,  depend4,  depend5,  depend6,
                             depend7,        depend8,  depend9,  depend10, depend11, depend12, depend13,
                             depend14,       depend15, depend17, depend18, depend19, depend20, depend21});
  auto get_item = TestCreateCNode(after_fg, "TupleGetItem",
                                  AnfNodePtrList{make_tuple, NewValueNode(SizeToLong(kTupleFirstItemIndex))},
                                  after_add_node->abstract());
  after_fg->set_output(get_item);

  // assert
  EXPECT_TRUE(CheckEqualGraph(after_fg, before_fg));
}
}  // namespace opt
}  // namespace mindspore
