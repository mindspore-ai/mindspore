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

#include "common/common_test.h"

#include "pybind11/pybind11.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "abstract/utils.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace abstract {

class TestAbstract : public UT::Common {
 public:
  TestAbstract() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(TestAbstract, TestParseDataClass) {
  // Check initialization before callback to Python.
  if (Py_IsInitialized() == 0) {
    Py_Initialize();
  }
  PyEval_InitThreads();

  py::object fn = parse::python_adapter::GetPyFn("gtest_input.pipeline.parse.parser_test", "TestFoo");

  ClassPtr cls_ptr = parse::ParseDataClass(fn);
  ASSERT_TRUE(nullptr != cls_ptr);
  std::shared_ptr<Class> cls = dyn_cast<Class>(cls_ptr);
  ASSERT_TRUE(nullptr != cls);

  MS_LOG(INFO) << "" << cls->ToString();
  ASSERT_EQ(cls->tag(), Named(std::string("TestFoo")));

  ClassAttrVector attributes = cls->GetAttributes();
  ASSERT_EQ(attributes.size(), 2);
  for (auto &v : attributes) {
    if (v.first == std::string("x")) {
      ASSERT_TRUE(nullptr != dyn_cast<Float>(v.second));
    }
    if (v.first == std::string("y")) {
      ASSERT_TRUE(nullptr != dyn_cast<Int>(v.second));
    }
  }

  std::unordered_map<std::string, ValuePtr> methods = cls->methods();
  ASSERT_EQ(methods.size(), 4);
  int counts = 0;
  for (auto &v : methods) {
    if (v.first == std::string("inf")) {
      counts++;
    }
    MS_LOG(INFO) << "" << v.first;
  }
  ASSERT_EQ(counts, 1);

  ValuePtr obj = std::make_shared<parse::ClassObject>(fn, "TestFoo");

  ValueNodePtr fn_node = NewValueNode(obj);
  AnfNodeConfigPtr fn_conf = std::make_shared<AnfNodeConfig>(nullptr, fn_node, nullptr, nullptr);
  AbstractBasePtr foo = ToAbstract(obj, nullptr, fn_conf);
  ASSERT_TRUE(foo != nullptr);

  AbstractBasePtr abstract_x = FromValue(1.1, true);
  AbstractBasePtr abstract_y = FromValue(static_cast<int64_t>(5), true);

  auto partical_func = dyn_cast<PartialAbstractClosure>(foo);
  AbstractBasePtrList args_spec_list = partical_func->args();
  ASSERT_GT(args_spec_list.size(), 0);
  AbstractScalarPtr abs_scalar = dyn_cast<AbstractScalar>(args_spec_list[0]);

  AbstractBasePtrList args_list = {abs_scalar, abstract_x, abstract_y};

  auto eval_impl = GetPrimitiveInferImpl(prim::kPrimMakeRecord);
  ASSERT_TRUE(nullptr != eval_impl.infer_shape_impl_);

  AbstractBasePtr new_cls = eval_impl.infer_shape_impl_(nullptr, prim::kPrimMakeRecord, args_list);
  ASSERT_TRUE(nullptr != new_cls);
}

}  // namespace abstract
}  // namespace mindspore
