/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/acl_ir/acl_utils.h"
#include <algorithm>
#include <set>
#include "utils/ms_context.h"
#include "transform/acl_ir/acl_convert.h"
#include "transform/acl_ir/acl_allocator.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "acl/acl.h"
#include "acl/acl_mdl.h"

namespace {
/*
1. Write a acl dump config file `acl_dump_cfg.json`, contents is as below, please refer to
`https://gitee.com/mindspore/mindspore/blob/master/config/acl_dump_cfg.json`
```json
{
  "dump": {
    "dump_list": [],
    "dump_path": "/tmp/acl_data_dump",
    "dump_mode": "all",
    "dump_op_switch": "on"
  }
}
```

2. Set acl dump config file path by environment variable `MS_ACL_DUMP_CFG_PATH`
```bash
export MS_ACL_DUMP_CFG_PATH=/xxx/acl_dump_cfg.json
```

3. Run mindspore to execute acl operators

4. Convert acl dump data to numpy npy format
```bash
${HOME}/Ascend/CANN-6.4/tools/operator_cmp/compare/msaccucmp.py convert -d data/20230520102032/0/xxx/0/ \
  -out /tmp/npy_acl_data
```

5. Write a python script file `print_data.py` to display npy data files
```python
import sys
import numpy as np

if len(sys.argv) < 2:
    print(f"Usage: sys.argv[0] npy_file1 npy_file2 ...")
    sys.exit()

for npy_file in sys.argv[1:]:
    data = np.load(npy_file)
    print(f'content of file {npy_file}:')
    print(f'dtype: {data.dtype}, shape: {data.shape}')
    print(data)
    print("")
```

6. Display contents of numpy data files
```bash
python3 print_data.py /tmp/npy_acl_data/xxx*.npy
```
*/

constexpr auto kAclDumpConfigPath = "MS_ACL_DUMP_CFG_PATH";

bool g_acl_initialized = false;
std::mutex g_acl_init_mutex;

void InitializeAcl() {
  std::lock_guard<std::mutex> lock(g_acl_init_mutex);
  if (g_acl_initialized) {
    return;
  }

  if (aclInit(nullptr) != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "Call aclInit failed, acl data dump function will be unusable.";
  } else {
    MS_LOG(INFO) << "Call aclInit successfully";
  }
  g_acl_initialized = true;
}

class AclDumper {
 public:
  AclDumper(AclDumper const &) = delete;             // disable copy constructor
  AclDumper &operator=(AclDumper const &) = delete;  // disable assignment operator

  // constructor
  AclDumper() : acl_dump_config_(mindspore::common::GetEnv(kAclDumpConfigPath)) {
    // acl dump config path is not set
    if (acl_dump_config_.empty()) {
      return;
    }

    // NOTE: function `aclmdlInitDump` must be called after `aclInit` to take effect, MindSpore never call `aclInit`
    // before, so here call it once
    InitializeAcl();

    if (aclmdlInitDump() != ACL_ERROR_NONE) {
      acl_dump_config_ = "";
      MS_LOG(WARNING) << "Call aclmdlInitDump failed, , acl data dump function will be unusable.";
    }
  }

  // destructor
  ~AclDumper() {
    if (acl_dump_config_.empty()) {
      return;
    }

    if (aclmdlFinalizeDump() != ACL_ERROR_NONE) {
      MS_LOG(WARNING) << "Call aclmdlFinalizeDump failed.";
    }
  }

  // set dump
  void SetDump() {
    if (acl_dump_config_.empty()) {
      return;
    }
    if (aclmdlSetDump(acl_dump_config_.c_str()) != ACL_ERROR_NONE) {
      MS_LOG(WARNING)
        << "Call aclmdlSetDump failed, acl data dump function will be unusable. Please check whether the config file `"
        << acl_dump_config_ << "` set by environment variable `" << kAclDumpConfigPath
        << "` is json file and correct, or may not have permission to read it.";
    }
  }

 private:
  std::string acl_dump_config_ = "";
};
}  // namespace

namespace mindspore {
namespace transform {
void AclAttrMaker::SetAttr(const string &attr_name, const bool value, aclopAttr *attr) {
  auto ret = aclopSetAttrBool(attr, attr_name.c_str(), value);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const int64_t value, aclopAttr *attr) {
  auto ret = aclopSetAttrInt(attr, attr_name.c_str(), value);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const float value, aclopAttr *attr) {
  auto ret = aclopSetAttrFloat(attr, attr_name.c_str(), value);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::string &value, aclopAttr *attr) {
  auto ret = aclopSetAttrString(attr, attr_name.c_str(), value.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<uint8_t> &value, aclopAttr *attr) {
  auto ret = aclopSetAttrListBool(attr, attr_name.c_str(), value.size(), value.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<int64_t> &value, aclopAttr *attr) {
  auto ret = aclopSetAttrListInt(attr, attr_name.c_str(), value.size(), value.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<float> &value, aclopAttr *attr) {
  auto ret = aclopSetAttrListFloat(attr, attr_name.c_str(), value.size(), value.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<std::string> &value, aclopAttr *attr) {
  std::vector<const char *> convert_list;
  (void)std::transform(value.begin(), value.end(), std::back_inserter(convert_list),
                       [](const std::string &s) { return s.c_str(); });
  auto ret = aclopSetAttrListString(attr, attr_name.c_str(), value.size(), convert_list.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<std::vector<int64_t>> &value, aclopAttr *attr) {
  auto list_size = value.size();
  int64_t *values[list_size];
  std::vector<int> num_values;
  for (size_t i = 0; i < list_size; i++) {
    values[i] = const_cast<int64_t *>(value[i].data());
    (void)num_values.emplace_back(SizeToInt(value[i].size()));
  }
  auto ret = aclopSetAttrListListInt(attr, attr_name.c_str(), list_size, num_values.data(), values);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const ::ge::DataType value, aclopAttr *attr) {
  auto ret =
    aclopSetAttrDataType(attr, attr_name.c_str(), AclConverter::ConvertType(TransformUtil::ConvertGeDataType(value)));
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

AclRunner::~AclRunner() { Reset(); }

void AclRunner::Reset() {
  (void)std::for_each(acl_param_.input_desc.begin(), acl_param_.input_desc.end(), aclDestroyTensorDesc);
  (void)std::for_each(acl_param_.output_desc.begin(), acl_param_.output_desc.end(), aclDestroyTensorDesc);
  (void)std::for_each(acl_param_.input_buffer.begin(), acl_param_.input_buffer.end(), aclDestroyDataBuffer);
  (void)std::for_each(acl_param_.output_buffer.begin(), acl_param_.output_buffer.end(), aclDestroyDataBuffer);
  if (acl_param_.attr != nullptr) {
    aclopDestroyAttr(acl_param_.attr);
    acl_param_.attr = nullptr;
  }

  acl_param_.input_desc.clear();
  acl_param_.input_buffer.clear();

  acl_param_.output_desc.clear();
  acl_param_.output_buffer.clear();

  op_type_ = "";
  is_dynamic_ = true;
}

void AclRunner::SetStaticMode() {
  auto set_compile_flag = aclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "enable");
  if (set_compile_flag != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set static compile mode failed! op_name is " << op_type_ << " and error flag is "
                      << set_compile_flag;
  }
  is_dynamic_ = false;
}

void AclRunner::SetDynamicMode() {
  auto set_compile_flag = aclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "disable");
  if (set_compile_flag != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set static compile mode failed! op_name is " << op_type_ << " and error flag is "
                      << set_compile_flag;
  }
  is_dynamic_ = true;
}

void AclRunner::SetPrecisionMode(const AclPrecisionMode mode) {
  int ret = -1;
  static std::string precision_mode = "not_inited";
  if (precision_mode == "not_inited") {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    precision_mode = ms_context->get_param<std::string>(MS_CTX_PRECISION_MODE);
  }
  if (!precision_mode.empty()) {
    ret = aclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, precision_mode.c_str());
  } else if (mode == ALLOW_FP32_TO_FP16) {
    static const std::string allow_fp32_to_fp16 = "allow_fp32_to_fp16";
    ret = aclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, allow_fp32_to_fp16.c_str());
  } else if (mode == FORCE_FP32) {
    static const std::string force_fp32 = "force_fp32";
    ret = aclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, force_fp32.c_str());
  } else {
    MS_LOG(EXCEPTION) << "Acl set run mode failed! op_name is " << op_type_ << " and error mode is " << mode;
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set precision mode failed! op_name is " << op_type_ << " and error flag is " << ret;
  }
}

void AclRunner::AoeDump() {
  // Dump acl graph for aoe.
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->EnableAoeOffline()) {
    auto file_path = GetSaveGraphsPathName("acl_dump");
    auto real_path = FileUtils::CreateNotExistDirs(file_path, true);
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "Get real path failed. path=" << file_path;
    }
    MS_LOG(INFO) << "Start aclGenGraphAndDumpForOp of op_type: " << op_type_;
    auto set_compile_flag = aclopSetCompileFlag(ACL_OP_COMPILE_DEFAULT);
    if (set_compile_flag != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl set static compile mode failed! op_name is " << op_type_ << " and error flag is "
                        << set_compile_flag;
    }
    auto dump_ret = aclGenGraphAndDumpForOp(
      const_cast<char *>(op_type_.c_str()), GetNumRealInputs(),
      const_cast<aclTensorDesc **>(acl_param_.input_desc.data()),
      const_cast<aclDataBuffer **>(acl_param_.input_buffer.data()), GetNumRealOutputs(),
      const_cast<aclTensorDesc **>(acl_param_.output_desc.data()), acl_param_.output_buffer.data(), acl_param_.attr,
      ACL_ENGINE_SYS, const_cast<char *>(real_path.value().c_str()), nullptr);
    if (dump_ret != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Acl dump graph failed!";
    }
    if (is_dynamic_) {
      SetDynamicMode();
    } else {
      SetStaticMode();
    }
  }
}

void AclRunner::Run(void *stream_ptr, bool is_sync) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  AclAllocatorRegister::Instance().RegisterAllocator(stream_ptr);
  AoeDump();

  AclDumper acl_dumper;
  acl_dumper.SetDump();

  MS_LOG(DEBUG) << "Start aclopCompileAndExecute of op_type: " << op_type_;
  if (is_sync) {
    bool ret = aclrtSynchronizeStreamWithTimeout(stream_ptr, -1);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl syncsteam failed, op_type_:" << op_type_;
    }
    ret = aclopCompileAndExecuteV2(const_cast<char *>(op_type_.c_str()), GetNumRealInputs(),
                                   const_cast<aclTensorDesc **>(acl_param_.input_desc.data()),
                                   const_cast<aclDataBuffer **>(acl_param_.input_buffer.data()), GetNumRealOutputs(),
                                   const_cast<aclTensorDesc **>(acl_param_.output_desc.data()),
                                   acl_param_.output_buffer.data(), acl_param_.attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS,
                                   nullptr, stream_ptr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl compile and execute failed, op_type_:" << op_type_;
    }
  } else {
    bool ret = aclopCompileAndExecute(const_cast<char *>(op_type_.c_str()), GetNumRealInputs(),
                                      acl_param_.input_desc.data(), acl_param_.input_buffer.data(), GetNumRealOutputs(),
                                      acl_param_.output_desc.data(), acl_param_.output_buffer.data(), acl_param_.attr,
                                      ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream_ptr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl compile and execute failed, op_type_:" << op_type_;
    }
  }

  MS_LOG(DEBUG) << "Success launch of op_type_: " << op_type_;
}

std::vector<std::vector<int64_t>> AclRunner::SyncData() {
  // 2. get output shape
  std::vector<std::vector<int64_t>> outputs_shape;
  for (size_t out_idx = 0; out_idx < acl_param_.output_desc.size(); ++out_idx) {
    size_t output_dim = aclGetTensorDescNumDims(acl_param_.output_desc.data()[out_idx]);
    if (output_dim == ACL_UNKNOWN_RANK) {
      MS_LOG(EXCEPTION) << "Acl get output shape dims failed, op_type_:" << op_type_;
    }
    std::vector<int64_t> output_shape(output_dim);
    for (size_t index = 0; index < output_dim; ++index) {
      auto ret = aclGetTensorDescDimV2(acl_param_.output_desc.data()[out_idx], index, &output_shape[index]);
      if (ret != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "Acl get output shape failed, op_type_:" << op_type_;
      }
    }
    (void)outputs_shape.emplace_back(output_shape);
  }

  MS_LOG(DEBUG) << "Acl SyncData success, op_type_: " << op_type_ << ", output_shape: " << outputs_shape;
  return outputs_shape;
}
}  // namespace transform
}  // namespace mindspore
