/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "mindspore/ccsrc/device/ascend/kernel_select_ascend.h"
#include "common/common_test.h"
#include "session/kernel_graph.h"
#include "kernel/kernel.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "operator/ops.h"
#include "mindspore/ccsrc/device/kernel_info.h"
#include "mindspore/ccsrc/kernel/kernel_build_info.h"
#include <vector>
namespace mindspore {
namespace device {
namespace ascend {
namespace {
using KernelInfo = device::KernelInfo;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
using KernelBuildInfo = kernel::KernelBuildInfo;
using KernelGraph = session::KernelGraph;
using KernelBuildInfoPtr = std::shared_ptr<KernelBuildInfo>;
using KernelBuilderPtr = std::shared_ptr<KernelBuildInfoBuilder>;
using Shape = std::vector<size_t>;
using ShapeList = std::vector<Shape>;
enum MatchCountPriority {
  MATCH_COUNT_PRIORITY_BEGIN = 0,
  MATCH_FORMAT_COUNT = MATCH_COUNT_PRIORITY_BEGIN,
  MATCH_DTYPE_COUNT,
  MATCH_NZ_FORMAT_COUNT,
  MATCH_5D_FORMAT_COUNT,
  MATCH_OUTPUT_DTYPE_COUNT,
  MATCH_COUNT_PRIORITY_END
};

const std::set<std::string> kOpFormatList = {
  kOpFormat_DEFAULT, kOpFormat_NC1KHKWHWC0, kOpFormat_ND,     kOpFormat_NCHW,      kOpFormat_NHWC,
  kOpFormat_HWCN,    kOpFormat_NC1HWC0,     kOpFormat_FRAC_Z, kOpFormat_C1HWNCoC0, kOpFormat_FRAC_NZ};

bool IsShapeMatchFormat(const std::vector<size_t> &shape, const std::string &format) {
  // if format is default,it remarkes support all format
  if (kOpFormatList.find(format) == kOpFormatList.end()) {
    MS_EXCEPTION(ArgumentError) << "got the unknow format " << format;
  }
  if (format == kOpFormat_DEFAULT) {
    return true;
  }
  // if shape size is 0,the shape will be a scalar
  if (shape.empty()) {
    return true;
  }
  if (shape.size() > kShapeSupportFormatMap.size()) {
    return false;
  }
  if (format == kOpFormat_FRAC_NZ && shape.size() >= 2) {
    return shape[shape.size() - 1] % 16 != 0 && shape[shape.size() - 2] % 16 != 0;
  }
  return !(kShapeSupportFormatMap[shape.size() - 1].find(format) == kShapeSupportFormatMap[shape.size() - 1].end());
}

bool IsValidKernelInfo(const std::shared_ptr<CNode> &kernel_node, const kernel::KernelBuildInfo &kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto check_function = [](const std::vector<size_t> &shape, const std::string &format) -> bool {
    if (!IsShapeMatchFormat(shape, format)) {
      return false;
    }
    for (auto shape_value : shape) {
      if (shape_value == 0) {
        MS_EXCEPTION(ArgumentError) << "dimension size of the tensor shape should be a positive integer, but got ["
                                    << shape_value << "]";
      }
    }
    return true;
  };
  for (size_t index = 0; index < kernel_build_info.GetOutputNum(); ++index) {
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, index);
    if (!check_function(output_shape, kernel_build_info.GetOutputFormat(index))) {
      return false;
    }
  }
  for (size_t index = 0; index < kernel_build_info.GetInputNum(); ++index) {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, index);
    if (!check_function(input_shape, kernel_build_info.GetInputFormat(index))) {
      return false;
    }
  }
  return true;
}

bool MatchInferOutputDataType(const CNodePtr &cnode, const kernel::KernelBuildInfo &kernel_build_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Check input data type
  for (size_t input_index = 0; input_index < kernel_build_info.GetInputNum(); ++input_index) {
    AnfNodePtr cur_input = cnode->input(input_index + 1);
    MS_EXCEPTION_IF_NULL(cur_input);
    TypeId input_origin_type;
    if (cur_input->isa<Parameter>() && AnfAlgo::IsParameterWeight(cur_input->cast<ParameterPtr>())) {
      // weight
      input_origin_type = AnfAlgo::GetOutputDeviceDataType(cur_input, 0);
    } else {
      // feature map
      input_origin_type = AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index);
    }
    if (input_origin_type == kTypeUnknown) {
      continue;
    }
    if (kernel_build_info.GetInputDeviceType(input_index) != input_origin_type) {
      return false;
    }
  }
  // Check output data type
  for (size_t output_index = 0; output_index < kernel_build_info.GetOutputNum(); ++output_index) {
    if (kernel_build_info.GetOutputDeviceType(output_index) != AnfAlgo::GetOutputInferDataType(cnode, output_index)) {
      return false;
    }
  }
  return true;
}

/**
 * compare too vector by priority,select a better vector,like compare too num,first compare highest num location,if
 * equal then next num location
 * example:[3,1,1,1] > [2,2,2,2] > [2,2,1,2] > [2,1,1,3]
 */
bool PriorityChooseItem(const std::vector<int> &cur_item, std::vector<int> *best_item) {
  MS_EXCEPTION_IF_NULL(best_item);
  if (cur_item.size() != best_item->size()) {
    MS_LOG(ERROR) << "item size should be same!";
    return false;
  }
  // Update the best_item by comparing the cur_item and best_item
  for (size_t i = 0; i < cur_item.size(); i++) {
    if (cur_item[i] > best_item->at(i)) {
      *best_item = cur_item;
      return true;
    } else if (cur_item[i] == best_item->at(i)) {
      continue;
    } else {
      return false;
    }
  }
  return false;
}

void UpdateCurMatchCounts(const kernel::KernelBuildInfo &kernel_build_info, const std::shared_ptr<CNode> &kernel_node,
                          std::vector<int> *const cur_kernelinfo_match_counts) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(cur_kernelinfo_match_counts);
  if (cur_kernelinfo_match_counts->size() < MATCH_COUNT_PRIORITY_END) {
    MS_EXCEPTION(ArgumentError) << "Out of range cur_kernelinfo_match_counts " << MATCH_COUNT_PRIORITY_END;
  }
  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(kernel_node); ++input_index) {
    AnfNodePtr input_anf_node = kernel_node->input(input_index + 1);
    MS_EXCEPTION_IF_NULL(input_anf_node);
    // if a input parameter is a weight with default format, the input shouldn't participate the judge
    if (input_anf_node->isa<Parameter>()) {
      auto para = input_anf_node->cast<ParameterPtr>();
      if (AnfAlgo::IsParameterWeight(para) && AnfAlgo::GetOutputDeviceDataType(para, 0) == kTypeUnknown) {
        continue;
      }
    }
    if (kernel_build_info.GetInputFormat(input_index) == AnfAlgo::GetPrevNodeOutputFormat(kernel_node, input_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_FORMAT_COUNT]++;
    }
    if (kernel_build_info.GetInputDeviceType(input_index) ==
        AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_DTYPE_COUNT]++;
    }
    if (kernel_build_info.GetInputFormat(input_index) == kOpFormat_FRAC_NZ) {
      (*cur_kernelinfo_match_counts)[MATCH_NZ_FORMAT_COUNT]++;
    }
    if (kernel_build_info.GetInputFormat(input_index) == kOpFormat_NC1HWC0) {
      (*cur_kernelinfo_match_counts)[MATCH_5D_FORMAT_COUNT]++;
    }
  }

  for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(kernel_node); ++output_index) {
    // cal count of same output dtype between abstract and kernel info
    if (kernel_build_info.GetOutputDeviceType(output_index) ==
        AnfAlgo::GetOutputInferDataType(kernel_node, output_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_OUTPUT_DTYPE_COUNT]++;
    }
  }
}

void SetKernelBuildInfo(KernelBuilderPtr builder) {
  builder->SetFusionType(kernel::OPAQUE);
  builder->SetKernelType(AUTO_DIFF_KERNEL);
  builder->SetProcessor(kernel::AICORE);
}

void test_select(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list) {
  std::vector<int> most_match_counts = {-1, -1, -1, -1, -1};
  int selected_index = -1;
  for (size_t info_index = 0; info_index < kernel_info_list.size(); ++info_index) {
    std::vector<int> cur_kernel_info_match_counts = {0, 0, 0, 0, 0};
    if (!IsValidKernelInfo(kernel_node, *(kernel_info_list[info_index]))) {
      continue;
    }
    if (!MatchInferOutputDataType(kernel_node, *(kernel_info_list[info_index]))) {
      continue;
    }
    std::shared_ptr<kernel::KernelBuildInfo> kernel_info_ptr = kernel_info_list[info_index];
    UpdateCurMatchCounts(*kernel_info_ptr, kernel_node, &cur_kernel_info_match_counts);
    // Currently the selection policy is the match format count first, and then is datatype counts.
    if (PriorityChooseItem(cur_kernel_info_match_counts, &most_match_counts)) {
      selected_index = SizeToInt(info_index);
    }
  }
  if (selected_index == -1) {
    MS_EXCEPTION(NotExistsError) << "" << kernel_node->DebugString() << " Cannot find valid kernel Info !";
  }
  auto index = IntToSize(selected_index);
  if (index >= kernel_info_list.size()) {
    MS_EXCEPTION(ArgumentError) << "index outof range";
  }
  std::shared_ptr<kernel::KernelBuildInfo> selected_kernel_info_ptr = kernel_info_list[index];
  MS_EXCEPTION_IF_NULL(selected_kernel_info_ptr);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info_ptr, kernel_node.get());
}

void SetParentAbstract(std::vector<AnfNodePtr> parent_list, std::vector<std::vector<size_t>> shapes,
                       std::vector<TypeId> types) {
  for (const auto &node : parent_list) {
    AnfAlgo::SetOutputInferTypeAndShape(types, shapes, node.get());
  }
}
}  // namespace
class AscendKernelSelctTest : public UT::Common {
 public:
  AscendKernelSelctTest() = default;
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(AscendKernelSelctTest, TestSelect) {
  std::vector<KernelBuilderPtr> build_list;
  std::vector<TypeId> type_list = {kNumberTypeFloat32};
  for (size_t i = 0; i <= 4; ++i) {
    build_list.push_back(std::make_shared<KernelBuildInfoBuilder>());
    SetKernelBuildInfo(build_list[i]);
    build_list[i]->SetInputsDeviceType(type_list);
    build_list[i]->SetOutputsDeviceType(type_list);
  }

  std::vector<std::string> nd_fmt = {kOpFormat_DEFAULT};
  std::vector<std::string> nz_fmt = {kOpFormat_FRAC_NZ};
  auto anf_graph = std::make_shared<KernelGraph>();

  // 16's multiple should not chose format NZ
  Shape nd_shapes = {2, 32, 224, 224};

  Shape nz_shapes = {3, 3, 5, 5};
  auto add_value = NewValueNode(prim::kPrimTensorAdd);
  auto a_node = anf_graph->NewCNode(std::vector<AnfNodePtr>{add_value});
  auto b_node = anf_graph->NewCNode(std::vector<AnfNodePtr>{add_value});
  std::vector<AnfNodePtr> parent_list = {add_value, a_node, b_node};

  auto c_node = anf_graph->NewCNode(parent_list);

  // a   b
  //  \ /
  //   c
  // a & b:  kernel_info:{output_format:{nz},dtype:{kNumberTypeFloat32}}
  //     infer_dtype:{kNumberTypeFloat32},infer_shape:{{3, 3, 5, 5}}
  // c:  infer_dtype:{kNumberTypeFloat32},infer_shape:{{3, 3,224, 224}}

  // set a & b's info
  SetParentAbstract(parent_list, ShapeList{nz_shapes}, type_list);
  // set abstract c
  AnfAlgo::SetOutputInferTypeAndShape(type_list, ShapeList{nd_shapes}, c_node.get());
  // set format of kernel info
  build_list[0]->SetOutputsFormat(nz_fmt);
  build_list[1]->SetOutputsFormat(nz_fmt);

  build_list[2]->SetInputsFormat(std::vector<std::string>{nd_fmt[0], nd_fmt[0]});
  build_list[3]->SetInputsFormat(std::vector<std::string>{nz_fmt[0], nz_fmt[0]});
  build_list[2]->SetInputsDeviceType(std::vector<TypeId>{kNumberTypeFloat32, kNumberTypeFloat32});
  build_list[3]->SetInputsDeviceType(std::vector<TypeId>{kNumberTypeFloat32, kNumberTypeFloat32});
  build_list[2]->SetOutputsFormat(nd_fmt);
  build_list[3]->SetOutputsFormat(nz_fmt);
  std::vector<KernelBuildInfoPtr> select_info_list;
  // set select info list
  select_info_list.emplace_back(build_list[2]->Build());
  select_info_list.emplace_back(build_list[3]->Build());

  // set device info for a & b
  AnfAlgo::SetSelectKernelBuildInfo(build_list[0]->Build(), a_node.get());
  AnfAlgo::SetSelectKernelBuildInfo(build_list[1]->Build(), b_node.get());

  test_select(c_node, select_info_list);
  EXPECT_EQ(AnfAlgo::GetInputFormat(c_node, 0), kOpFormat_DEFAULT);
  EXPECT_EQ(AnfAlgo::GetInputFormat(c_node, 1), kOpFormat_DEFAULT);

  // set a & b's info
  // a   b
  //  \ /
  //   c
  // a: kernel_info:{output_format:{5d},dtype:{kNumberTypeFloat32}}
  //    infer_dtype:{kNumberTypeFloat32},infer_shape:{{3, 3, 5, 5}}
  // b:  kernel_info:{output_format:{nz},dtype:{kNumberTypeFloat32}}
  //     infer_dtype:{kNumberTypeFloat32},infer_shape:{{3, 3, 5, 5}}
  // c:  infer_dtype:{kNumberTypeFloat32},infer_shape:{{3, 3, 5, 5}}

  // set a & b's info
  SetParentAbstract(parent_list, ShapeList{nz_shapes}, type_list);
  // set abstract c
  AnfAlgo::SetOutputInferTypeAndShape(type_list, ShapeList{nz_shapes}, c_node.get());
  // set format of kernel info
  build_list[0]->SetOutputsFormat(std::vector<std::string>{kOpFormat_NC1HWC0});
  build_list[1]->SetOutputsFormat(nz_fmt);

  build_list[2]->SetInputsFormat(std::vector<std::string>{kOpFormat_NC1HWC0, nd_fmt[0]});
  build_list[3]->SetInputsFormat(std::vector<std::string>{nd_fmt[0], nz_fmt[0]});
  build_list[2]->SetInputsDeviceType(std::vector<TypeId>{kNumberTypeFloat32, kNumberTypeFloat32});
  build_list[3]->SetInputsDeviceType(std::vector<TypeId>{kNumberTypeFloat32, kNumberTypeFloat32});
  build_list[2]->SetOutputsFormat(nd_fmt);
  build_list[3]->SetOutputsFormat(nz_fmt);
  // set select info list
  select_info_list.emplace_back(build_list[2]->Build());
  select_info_list.emplace_back(build_list[3]->Build());

  // set device info for a & b
  AnfAlgo::SetSelectKernelBuildInfo(build_list[0]->Build(), a_node.get());
  AnfAlgo::SetSelectKernelBuildInfo(build_list[1]->Build(), b_node.get());

  test_select(c_node, select_info_list);
  EXPECT_EQ(AnfAlgo::GetInputFormat(c_node, 0), kOpFormat_DEFAULT);
  EXPECT_EQ(AnfAlgo::GetInputFormat(c_node, 1), kOpFormat_FRAC_NZ);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore