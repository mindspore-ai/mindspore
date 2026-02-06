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

#include "kernel/ascend/acl_ir/acl_utils.h"
#include <algorithm>
#include <mutex>
#include <set>
#include "plugin/ascend/res_manager/device_context_conf/op_tuning_conf.h"
#include "utils/ms_context.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "kernel/ascend/acl_ir/acl_allocator.h"
#include "include/utils/common.h"
#include "utils/file_utils.h"
#include "tools/profiler/profiler.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_op_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

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

class AclDumper {
 public:
  AclDumper(AclDumper const &) = delete;             // disable copy constructor
  AclDumper &operator=(AclDumper const &) = delete;  // disable assignment operator

  // constructor
  AclDumper() : acl_dump_config_("") {
    // acl dump config path is not set
    if (acl_dump_config_.empty()) {
      return;
    }

    if (CALL_ASCEND_API(aclmdlInitDump) != ACL_SUCCESS) {
      acl_dump_config_ = "";
      MS_LOG(WARNING) << "Call aclmdlInitDump failed, , acl data dump function will be unusable.";
    }
  }

  // destructor
  ~AclDumper() {
    if (acl_dump_config_.empty()) {
      return;
    }

    if (CALL_ASCEND_API(aclmdlFinalizeDump) != ACL_SUCCESS) {
      MS_LOG(WARNING) << "Call aclmdlFinalizeDump failed.";
    }
  }

  // set dump
  void SetDump() {
    if (acl_dump_config_.empty()) {
      return;
    }
    if (CALL_ASCEND_API(aclmdlSetDump, acl_dump_config_.c_str()) != ACL_SUCCESS) {
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

namespace mindspore::device::ascend {
void AclAttrMaker::SetAttr(const string &attr_name, const bool value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrBool, attr, attr_name.c_str(), value);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const int64_t value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrInt, attr, attr_name.c_str(), value);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const float value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrFloat, attr, attr_name.c_str(), value);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::string &value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrString, attr, attr_name.c_str(), value.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<uint8_t> &value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrListBool, attr, attr_name.c_str(), value.size(), value.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<int64_t> &value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrListInt, attr, attr_name.c_str(), value.size(), value.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<float> &value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrListFloat, attr, attr_name.c_str(), value.size(), value.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<std::string> &value, aclopAttr *attr) {
  std::vector<const char *> convert_list;
  (void)std::transform(value.begin(), value.end(), std::back_inserter(convert_list),
                       [](const std::string &s) { return s.c_str(); });
  auto ret = CALL_ASCEND_API(aclopSetAttrListString, attr, attr_name.c_str(), value.size(), convert_list.data());
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
  auto ret = CALL_ASCEND_API(aclopSetAttrListListInt, attr, attr_name.c_str(), list_size, num_values.data(), values);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const ::ge::DataType value, aclopAttr *attr) {
  auto ret = CALL_ASCEND_API(aclopSetAttrDataType, attr, attr_name.c_str(),
                             AclConverter::ConvertType(device::ascend::TransformUtil::ConvertGeDataType(value)));
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

void AclAttrMaker::SetAttr(const string &attr_name, const std::vector<::ge::DataType> value, aclopAttr *attr) {
  auto list_size = value.size();
  std::vector<aclDataType> data;
  (void)std::transform(value.begin(), value.end(), std::back_inserter(data), [](const ::ge::DataType &val) {
    return AclConverter::ConvertType(device::ascend::TransformUtil::ConvertGeDataType(val));
  });
  auto ret = CALL_ASCEND_API(aclopSetAttrListDataType, attr, attr_name.c_str(), list_size, data.data());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value " << value << " failed!";
  }
}

AclRunner::~AclRunner() {
  (void)std::for_each(acl_param_.input_desc.begin(), acl_param_.input_desc.end(),
                      [&](const aclTensorDesc *desc) { CALL_ASCEND_API(aclDestroyTensorDesc, desc); });
  (void)std::for_each(acl_param_.input_buffer.begin(), acl_param_.input_buffer.end(),
                      [&](const aclDataBuffer *dataBuffer) { CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer); });
  (void)std::for_each(acl_param_.output_desc.begin(), acl_param_.output_desc.end(),
                      [&](const aclTensorDesc *desc) { CALL_ASCEND_API(aclDestroyTensorDesc, desc); });
  (void)std::for_each(acl_param_.output_buffer.begin(), acl_param_.output_buffer.end(),
                      [&](const aclDataBuffer *dataBuffer) { CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer); });

  acl_param_.input_desc.clear();
  acl_param_.input_buffer.clear();
  acl_param_.output_desc.clear();
  acl_param_.output_buffer.clear();
}

void AclRunner::Reset() {
  (void)std::for_each(acl_param_.output_desc.begin(), acl_param_.output_desc.end(),
                      [&](const aclTensorDesc *desc) { CALL_ASCEND_API(aclDestroyTensorDesc, desc); });
  (void)std::for_each(acl_param_.output_buffer.begin(), acl_param_.output_buffer.end(),
                      [&](const aclDataBuffer *dataBuffer) { CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer); });

  acl_param_.output_desc.clear();
  acl_param_.output_buffer.clear();

  is_dynamic_ = true;
}

void AclRunner::SetStaticMode() {
  auto set_compile_flag = AclUtil::SetCompileMode(0);
  if (set_compile_flag != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set static compile mode failed! op_name is " << op_type_ << " and error flag is "
                      << set_compile_flag;
  }
  is_dynamic_ = false;
}

void AclRunner::SetDynamicMode() {
  auto set_compile_flag = AclUtil::SetCompileMode(1);
  if (set_compile_flag != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set static compile mode failed! op_name is " << op_type_ << " and error flag is "
                      << set_compile_flag;
  }
  is_dynamic_ = true;
}

void AclRunner::SetPrecisionMode(const AclPrecisionMode mode) {
  auto real_mode = mode;
  if (mode == DEFAULT_MODE) {
    real_mode = (AclUtil::KeepOriginDType() == 1) ? MUST_KEEP_ORIGIN_DTYPE : ALLOW_FP32_TO_FP16;
  }
  auto iter = acl_precision_map.find(real_mode);
  if (iter == acl_precision_map.end()) {
    MS_LOG(EXCEPTION) << "Acl set run mode failed! op_name is " << op_type_ << " and error mode is " << real_mode;
  }

  auto ret = AclUtil::SetPrecisionMode(iter->second);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set precision mode failed! mode is " << iter->second << ", op_name is " << op_type_
                      << " and error flag is " << ret;
  }
}

void AclRunner::AoeDump() {
  // Dump acl graph for aoe.
  auto op_tuning_conf = device::ascend::OpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  if (op_tuning_conf->EnableAoeOffline()) {
    auto file_path = GetSaveGraphsPathName("acl_dump");
    auto real_path = FileUtils::CreateNotExistDirs(file_path, true);
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "Get real path failed. path=" << file_path;
    }
    MS_LOG(INFO) << "Start aclGenGraphAndDumpForOp of op_type: " << op_type_;
    auto set_compile_flag = CALL_ASCEND_API(aclopSetCompileFlag, ACL_OP_COMPILE_DEFAULT);
    if (set_compile_flag != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl set static compile mode failed! op_name is " << op_type_ << " and error flag is "
                        << set_compile_flag;
    }
    auto dump_ret =
      CALL_ASCEND_API(aclGenGraphAndDumpForOp, const_cast<char *>(op_type_.c_str()), GetNumRealInputs(),
                      const_cast<aclTensorDesc **>(acl_param_.input_desc.data()),
                      const_cast<aclDataBuffer **>(acl_param_.input_buffer.data()), GetNumRealOutputs(),
                      const_cast<aclTensorDesc **>(acl_param_.output_desc.data()), acl_param_.output_buffer.data(),
                      acl_param_.attr, ACL_ENGINE_SYS, const_cast<char *>(real_path.value().c_str()), nullptr);
    if (dump_ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl dump graph failed!";
    }
    if (is_dynamic_) {
      SetDynamicMode();
    } else {
      SetStaticMode();
    }
  }
}

void AclRunner::FillOptInputWithPlaceHolder() {
  if (acl_param_.input_desc.empty()) {
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(acl_param_.input_desc.size() == acl_param_.input_buffer.size(),
                             "Acl param input_desc size is not equal to acl param input_buffer size");
  bool effective_flag = false;
  for (int i = static_cast<int>(acl_param_.input_desc.size()) - 1; i >= 0; --i) {
    if (acl_param_.input_desc[i] != nullptr && acl_param_.input_buffer[i] != nullptr) {
      if (!effective_flag) {
        effective_flag = true;
      }
      continue;
    }
    if (!effective_flag) {
      continue;
    }

    // create placeholder for input_desc
    if (acl_param_.input_desc[i] == nullptr) {
      acl_param_.input_desc[i] =
        CALL_ASCEND_API(aclCreateTensorDesc, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    }

    // create placeholder for input_buffer
    if (acl_param_.input_buffer[i] == nullptr) {
      acl_param_.input_buffer[i] = CALL_ASCEND_API(aclCreateDataBuffer, nullptr, 0);
    }
  }
}

void AclRunner::Run(void *stream_ptr, bool is_sync) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  AclAllocatorRegister::Instance().RegisterAllocator(stream_ptr);
  AoeDump();

  AclDumper acl_dumper;
  acl_dumper.SetDump();
  if (op_type_ == "ApplyAdamWMS") {
    op_type_ = "ApplyAdamW";
  }
  MS_LOG(DEBUG) << "Start aclopCompileAndExecute of op_type: " << op_type_;
  if (is_sync) {
    bool ret = CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream_ptr, -1);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl syncsteam failed, op_type_:" << op_type_;
    }
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunchInner,
                                       "aclopCompileAndExecuteV2", true);
    ret = CALL_ASCEND_API(aclopCompileAndExecuteV2, const_cast<char *>(op_type_.c_str()), GetNumRealInputs(),
                          const_cast<aclTensorDesc **>(acl_param_.input_desc.data()),
                          const_cast<aclDataBuffer **>(acl_param_.input_buffer.data()), GetNumRealOutputs(),
                          const_cast<aclTensorDesc **>(acl_param_.output_desc.data()), acl_param_.output_buffer.data(),
                          acl_param_.attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream_ptr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl compile and execute failed, op_type_:" << op_type_;
    }
  } else {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunchInner,
                                       "aclopCompileAndExecute", true);
    bool ret = CALL_ASCEND_API(aclopCompileAndExecute, const_cast<char *>(op_type_.c_str()), GetNumRealInputs(),
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
    size_t output_dim = CALL_ASCEND_API(aclGetTensorDescNumDims, acl_param_.output_desc.data()[out_idx]);
    if (output_dim == ACL_UNKNOWN_RANK) {
      MS_LOG(EXCEPTION) << "Acl get output shape dims failed, op_type_:" << op_type_;
    }
    std::vector<int64_t> output_shape(output_dim);
    for (size_t index = 0; index < output_dim; ++index) {
      auto ret =
        CALL_ASCEND_API(aclGetTensorDescDimV2, acl_param_.output_desc.data()[out_idx], index, &output_shape[index]);
      if (ret != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "Acl get output shape failed, op_type_:" << op_type_;
      }
    }
    (void)outputs_shape.emplace_back(output_shape);
  }

  MS_LOG(DEBUG) << "Acl SyncData success, op_type_: " << op_type_ << ", output_shape: " << outputs_shape;
  return outputs_shape;
}
}  // namespace  mindspore::device::ascend
