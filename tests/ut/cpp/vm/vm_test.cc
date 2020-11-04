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
#include "vm/vm.h"
#include "common/common_test.h"
#include "frontend/operator/ops.h"
#include "vm/backend.h"

namespace mindspore {
namespace compile {

class TestCompileVM : public UT::Common {
 public:
  TestCompileVM() {}
  virtual ~TestCompileVM() {}

 public:
  virtual void SetUp();
  virtual void TearDown();
};

void TestCompileVM::SetUp() { MS_LOG(INFO) << "TestCompileVM::SetUp()"; }

void TestCompileVM::TearDown() { MS_LOG(INFO) << "TestCompileVM::TearDown()"; }

TEST_F(TestCompileVM, StructPartial) {
  auto partial = new StructPartial(100, VectorRef({20, 60, 100.0}));
  std::stringstream ss;
  ss << *partial;
  delete partial;
  partial = nullptr;
}

TEST_F(TestCompileVM, FinalVM) {
  std::vector<std::pair<Instruction, VectorRef>> instr;
  instr.push_back({Instruction::kCall, VectorRef({static_cast<int64_t>(-1)})});
  instr.push_back(
    {Instruction::kTailCall, VectorRef({static_cast<int64_t>(-2), static_cast<int64_t>(1), static_cast<int64_t>(1)})});
  instr.push_back({Instruction::kReturn, VectorRef({static_cast<int64_t>(-1), static_cast<int64_t>(1)})});
  instr.push_back({Instruction::kPartial, VectorRef({static_cast<int64_t>(0), "cc"})});
  BackendPtr backend = std::make_shared<Backend>("vm");
  auto vm = new FinalVM(instr, backend);
  vm->Eval(VectorRef({static_cast<int64_t>(1), static_cast<int64_t>(2), static_cast<int64_t>(3),
                      static_cast<int64_t>(-1), "a", "b", "c"}));
  delete vm;
  vm = nullptr;
}

}  // namespace compile
}  // namespace mindspore
