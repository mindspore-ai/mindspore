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
#ifndef TESTS_UT_CPP_GRAPH_KERNEL_COMMON_GRAPH_KERNEL_TEST_SUITE_H_
#define TESTS_UT_CPP_GRAPH_KERNEL_COMMON_GRAPH_KERNEL_TEST_SUITE_H_

#include "common/common_test.h"
#include "common/graph_optimizer_test_framework.h"
#include "utils/ms_context.h"
#include "ir/tensor_new.h"

namespace mindspore::graphkernel::test {
using mindspore::test::ConstructGraph;
void RunPass(const FuncGraphPtr &graph, const std::vector<opt::PassPtr> &passes);

class GraphKernelCommonTestSuite : public UT::Common {
 public:
  GraphKernelCommonTestSuite(){};
  virtual ~GraphKernelCommonTestSuite() = default;

  void TearDown() override { SetGraphKernelFlags(""); }
  size_t pass_stage_ = 0;
  void RunPass(const FuncGraphPtr &graph, const std::vector<opt::PassPtr> &passes);

  void SetGraphKernelFlags(const std::string &flags);
  void SetDeviceTarget(const std::string &device);

  AnfNodePtrList GetAllNodes(const FuncGraphPtr &fg);
  CNodePtrList GetAllCNodes(const FuncGraphPtr &fg);
  CNodePtrList GetAllGKNodes(const FuncGraphPtr &fg);

  void CheckInputOutputType(const AnfNodePtr &node, const std::vector<TypeId> &target_inputs_type,
                            TypeId target_output_type);
};

template <typename T>
ValuePtr GenScalar(const TypePtr &type, T value) {
  MS_EXCEPTION_IF_NULL(type);
  switch (type->type_id()) {
    case kNumberTypeBool:
      return MakeValue(static_cast<bool>(value));
    case kNumberTypeInt8:
      return MakeValue(static_cast<int8_t>(value));
    case kNumberTypeInt16:
      return MakeValue(static_cast<int16_t>(value));
    case kNumberTypeInt32:
      return MakeValue(static_cast<int32_t>(value));
    case kNumberTypeInt64:
      return MakeValue(static_cast<int64_t>(value));
    case kNumberTypeUInt8:
      return MakeValue(static_cast<uint8_t>(value));
    case kNumberTypeUInt16:
      return MakeValue(static_cast<uint16_t>(value));
    case kNumberTypeUInt32:
      return MakeValue(static_cast<uint32_t>(value));
    case kNumberTypeUInt64:
      return MakeValue(static_cast<uint64_t>(value));
    case kNumberTypeFloat16:
      return std::make_shared<FP16Imm>(static_cast<float16>(value));
    case kNumberTypeFloat32:
      return MakeValue(static_cast<float>(value));
    case kNumberTypeFloat64:
      return MakeValue(static_cast<double>(value));
    case kNumberTypeBFloat16:
      return std::make_shared<BF16Imm>(static_cast<bfloat16>(value));
    default:
      MS_LOG(ERROR) << "Not support cast to dst type: " << type->ToString();
      return nullptr;
  }
}

template <typename T>
ValuePtr GenTensor(const TypePtr &type, T value) {
  MS_EXCEPTION_IF_NULL(type);
  switch (type->type_id()) {
    case kNumberTypeBool:
      return tensor::from_scalar(static_cast<bool>(value), type);
    case kNumberTypeInt8:
      return tensor::from_scalar(static_cast<int8_t>(value), type);
    case kNumberTypeInt16:
      return tensor::from_scalar(static_cast<int16_t>(value), type);
    case kNumberTypeInt32:
      return tensor::from_scalar(static_cast<int32_t>(value), type);
    case kNumberTypeInt64:
      return tensor::from_scalar(static_cast<int64_t>(value), type);
    case kNumberTypeUInt8:
      return tensor::from_scalar(static_cast<uint8_t>(value), type);
    case kNumberTypeUInt16:
      return tensor::from_scalar(static_cast<uint16_t>(value), type);
    case kNumberTypeUInt32:
      return tensor::from_scalar(static_cast<uint32_t>(value), type);
    case kNumberTypeUInt64:
      return tensor::from_scalar(static_cast<uint64_t>(value), type);
    case kNumberTypeFloat16:
      return tensor::from_scalar(static_cast<float16>(value), type);
    case kNumberTypeFloat32:
      return tensor::from_scalar(static_cast<float>(value), type);
    case kNumberTypeFloat64:
      return tensor::from_scalar(static_cast<double>(value), type);
    case kNumberTypeBFloat16:
      return tensor::from_scalar(static_cast<bfloat16>(value), type);
    default:
      MS_LOG(ERROR) << "Not support cast to dst type: " << type->ToString();
      return nullptr;
  }
}

template <typename T>
ValuePtr NewScalar(const TypePtr type, T value, bool is_scalar = true) {
  return is_scalar ? GenScalar(type, value) : GenTensor(type, value);
}
}  // namespace mindspore::graphkernel::test
#endif  // TESTS_UT_CPP_GRAPH_KERNEL_COMMON_GRAPH_KERNEL_TEST_SUITE_H_
