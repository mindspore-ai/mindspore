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

#ifndef MINDSPORE_TBE_JSON_UTILS_H
#define MINDSPORE_TBE_JSON_UTILS_H
#include <memory>
#include <map>
#include <list>
#include <algorithm>
#include <vector>
#include <string>
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "runtime/dev.h"
#include "utils/ms_utils.h"
namespace mindspore::kernel {
constexpr auto kJFusionOpList = "op_list";
constexpr auto kJFusionKernelNamePrefix = "te_fusion_";
constexpr auto kJOptional = "optional_";
constexpr auto kJOpFormat_FRACTAL_Z = "FRACTAL_Z";
constexpr auto kJFullName = "full_name";
constexpr auto kJDtype = "dtype";
constexpr auto kJName = "name";
constexpr auto kJOriShape = "ori_shape";
constexpr auto kJOriFormat = "ori_format";
constexpr auto kJShape = "shape";
constexpr auto kJFormat = "format";
constexpr auto kJValid = "valid";
constexpr auto kJParamType = "param_type";
constexpr auto kJParamDynamic = "dynamic";
constexpr auto kJParamRequred = "required";
constexpr auto kJParamOptional = "optional";
constexpr auto kJDataType = "data_type";
constexpr auto kJOutputIndex = "output_index";
constexpr auto kJOutputDataDesc = "output_data_desc";
constexpr auto kJOutputDesc = "output_desc";
constexpr auto kJInputDesc = "input_desc";
constexpr auto kJRange = "range";
constexpr auto kVTypeInt = "int";
constexpr auto kVTypeStr = "str";
constexpr auto kVTypeBool = "bool";
constexpr auto kVTypeFloat = "float";
constexpr auto kVTypeFloat32 = "float32";
constexpr auto kVTypeListInt = "listInt";
constexpr auto kVTypeInt32 = "Int32";
constexpr auto kVTypeInt64 = "Int64";
constexpr auto kVTypeListInt64 = "listInt64";
constexpr auto kVTypeListUInt64 = "listUInt64";
constexpr auto kVTypeListFloat = "listFloat";
constexpr auto kVTypeListListInt = "listListInt";
constexpr auto kJValue = "value";
constexpr auto kJDynIndex = "dyn_index";
constexpr auto kJFuncName = "func_name";
constexpr auto kJL1AddrOffset = "L1_addr_offset";
constexpr auto kJL1FusionType = "L1_fusion_type";
constexpr auto kJL1WorkspaceSize = "L1_workspace_size";
constexpr auto kJAddrType = "addr_type";
constexpr auto kJSliceOffset = "slice_offset";
constexpr auto kJSplitIndex = "split_index";
constexpr auto kJTotalShape = "total_shape";
constexpr auto kJDynamicCompileStatic = "dynamic_compile_static";
constexpr auto kJIsDynamicImpl = "is_dynamic_impl";
constexpr auto kJInt64Mode = "int64mode";
constexpr auto kJValidShape = "valid_shape";
constexpr auto kJModuleName = "module_name";
constexpr auto kJModuleNamePrefix = "impl.";
constexpr auto kJPattern = "pattern";
constexpr auto kJPyModulePath = "py_module_path";
constexpr auto kJAttrs = "attrs";
constexpr auto kJAttrDesc = "attr_desc";
constexpr auto kJSocInfo = "SocInfo";
constexpr auto kJCoreType = "coreType";
constexpr auto kJFusionOpName = "fusion_op_name";
constexpr auto kJGraphID = "graph_id";
constexpr auto kJType = "type";
constexpr auto kJIsRef = "isRef";
constexpr auto kJL1Size = "l1_size";
constexpr auto kJScopeID = "scope_id";
constexpr auto kJGraphName = "graph_name";
constexpr auto kJOpList = "op_list";
constexpr auto kJNull = "NULL";
constexpr auto kJData = "Data";
constexpr auto kJOriName = "ori_name";
constexpr auto kJBuildType = "build_type";
constexpr auto kJMissSupportInfo = "miss_support_info";
constexpr auto kJMaxKernelID = "max_kernel_id";
constexpr auto kJOpName = "op_name";
constexpr auto kJUnknowShape = "unknown_shape";
constexpr auto kJListArgs = "list_args";
constexpr auto kAccuratelyBuild = "accurately_build";
constexpr auto kPyPath = "/usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe";
constexpr auto kJMaxKernelIDValue = 10;
constexpr auto kJConstValue = "const_value";
constexpr auto kJConstValueDtype = "const_value_dtype";
constexpr auto kJOpDebugConfig = "op_debug_config";
constexpr auto kJCValue = "input_c_values";

class TbeJsonUtils {
 public:
  static bool GetInputsRealNum(const AnfNodePtr &anf_node, const std::vector<OpIOInfoPtr> &inputs_ptr,
                               std::vector<size_t> *inputs_num);
  static bool GetOutputsRealNum(const AnfNodePtr &anf_node, const std::vector<OpIOInfoPtr> &outputs_ptr,
                                std::vector<size_t> *outputs_num);
  static bool IsNeedChangeDefaultFormat(const AnfNodePtr &anf_node);
  // just for generate json for ascend op build, it will be deleted after unify size_t and int64_t.
  static std::vector<int64_t> GetInputOriShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx);
  static std::vector<int64_t> GetInputDeviceShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx);
  static std::vector<int64_t> GetOutputOriShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx);
  static std::vector<int64_t> GetOutputDeviceShapeForTbeBuild(const AnfNodePtr &anf_node, size_t real_idx);
};

}  // namespace mindspore::kernel
#endif  // MINDSPORE_TBE_JSON_UTILS_H
