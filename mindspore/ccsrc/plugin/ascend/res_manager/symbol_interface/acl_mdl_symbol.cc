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
#include "acl_mdl_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mindspore::device::ascend {
aclmdlAddDatasetBufferFunObj aclmdlAddDatasetBuffer_ = nullptr;
aclmdlCreateDatasetFunObj aclmdlCreateDataset_ = nullptr;
aclmdlCreateDescFunObj aclmdlCreateDesc_ = nullptr;
aclmdlDestroyDatasetFunObj aclmdlDestroyDataset_ = nullptr;
aclmdlDestroyDescFunObj aclmdlDestroyDesc_ = nullptr;
aclmdlExecuteFunObj aclmdlExecute_ = nullptr;
aclmdlExecuteV2FunObj aclmdlExecuteV2_ = nullptr;
aclmdlCreateExecConfigHandleFunObj aclmdlCreateExecConfigHandle_ = nullptr;
aclmdlDestroyExecConfigHandleFunObj aclmdlDestroyExecConfigHandle_ = nullptr;
aclmdlSetExecConfigOptFunObj aclmdlSetExecConfigOpt_ = nullptr;
aclmdlFinalizeDumpFunObj aclmdlFinalizeDump_ = nullptr;
aclmdlGetCurOutputDimsFunObj aclmdlGetCurOutputDims_ = nullptr;
aclmdlGetDatasetBufferFunObj aclmdlGetDatasetBuffer_ = nullptr;
aclmdlGetDatasetNumBuffersFunObj aclmdlGetDatasetNumBuffers_ = nullptr;
aclmdlGetDescFunObj aclmdlGetDesc_ = nullptr;
aclmdlGetInputDataTypeFunObj aclmdlGetInputDataType_ = nullptr;
aclmdlGetInputDimsFunObj aclmdlGetInputDims_ = nullptr;
aclmdlGetInputIndexByNameFunObj aclmdlGetInputIndexByName_ = nullptr;
aclmdlGetInputNameByIndexFunObj aclmdlGetInputNameByIndex_ = nullptr;
aclmdlGetInputSizeByIndexFunObj aclmdlGetInputSizeByIndex_ = nullptr;
aclmdlGetNumInputsFunObj aclmdlGetNumInputs_ = nullptr;
aclmdlGetNumOutputsFunObj aclmdlGetNumOutputs_ = nullptr;
aclmdlGetOutputDataTypeFunObj aclmdlGetOutputDataType_ = nullptr;
aclmdlGetOutputDimsFunObj aclmdlGetOutputDims_ = nullptr;
aclmdlGetOutputNameByIndexFunObj aclmdlGetOutputNameByIndex_ = nullptr;
aclmdlGetOutputSizeByIndexFunObj aclmdlGetOutputSizeByIndex_ = nullptr;
aclmdlInitDumpFunObj aclmdlInitDump_ = nullptr;
aclmdlLoadFromMemFunObj aclmdlLoadFromMem_ = nullptr;
aclmdlSetDumpFunObj aclmdlSetDump_ = nullptr;
aclmdlSetDynamicBatchSizeFunObj aclmdlSetDynamicBatchSize_ = nullptr;
aclmdlUnloadFunObj aclmdlUnload_ = nullptr;
aclmdlQuerySizeFromMemFunObj aclmdlQuerySizeFromMem_ = nullptr;
aclmdlBundleGetModelIdFunObj aclmdlBundleGetModelId_ = nullptr;
aclmdlBundleLoadFromMemFunObj aclmdlBundleLoadFromMem_ = nullptr;
aclmdlBundleUnloadFunObj aclmdlBundleUnload_ = nullptr;
aclmdlLoadFromMemWithMemFunObj aclmdlLoadFromMemWithMem_ = nullptr;
aclmdlSetDatasetTensorDescFunObj aclmdlSetDatasetTensorDesc_ = nullptr;
aclmdlGetInputFormatFunObj aclmdlGetInputFormat_ = nullptr;
aclmdlGetDatasetTensorDescFunObj aclmdlGetDatasetTensorDesc_ = nullptr;
aclmdlSetInputDynamicDimsFunObj aclmdlSetInputDynamicDims_ = nullptr;
aclmdlGetOutputFormatFunObj aclmdlGetOutputFormat_ = nullptr;
aclmdlGetInputDimsV2FunObj aclmdlGetInputDimsV2_ = nullptr;
aclmdlGetDynamicHWFunObj aclmdlGetDynamicHW_ = nullptr;
aclmdlGetInputDynamicDimsFunObj aclmdlGetInputDynamicDims_ = nullptr;
aclmdlGetInputDynamicGearCountFunObj aclmdlGetInputDynamicGearCount_ = nullptr;
aclmdlGetDynamicBatchFunObj aclmdlGetDynamicBatch_ = nullptr;
aclmdlSetDynamicHWSizeFunObj aclmdlSetDynamicHWSize_ = nullptr;
#if defined(__linux__) && defined(WITH_BACKEND)
aclmdlRICaptureBeginFunObj aclmdlRICaptureBegin_ = nullptr;
aclmdlRICaptureGetInfoFunObj aclmdlRICaptureGetInfo_ = nullptr;
aclmdlRICaptureEndFunObj aclmdlRICaptureEnd_ = nullptr;
aclmdlRIExecuteAsyncFunObj aclmdlRIExecuteAsync_ = nullptr;
aclmdlRIDestroyFunObj aclmdlRIDestroy_ = nullptr;
#endif

void LoadAclMdlApiSymbol(const std::string &ascend_path) {
  std::string aclmdl_plugin_path = ascend_path + "lib64/libascendcl.so";
  auto handler = GetLibHandler(aclmdl_plugin_path);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << aclmdl_plugin_path << " failed!" << dlerror();
    return;
  }
  aclmdlAddDatasetBuffer_ = DlsymAscendFuncObj(aclmdlAddDatasetBuffer, handler);
  aclmdlCreateDataset_ = DlsymAscendFuncObj(aclmdlCreateDataset, handler);
  aclmdlCreateDesc_ = DlsymAscendFuncObj(aclmdlCreateDesc, handler);
  aclmdlDestroyDataset_ = DlsymAscendFuncObj(aclmdlDestroyDataset, handler);
  aclmdlDestroyDesc_ = DlsymAscendFuncObj(aclmdlDestroyDesc, handler);
  aclmdlExecute_ = DlsymAscendFuncObj(aclmdlExecute, handler);
  aclmdlExecuteV2_ = DlsymAscendFuncObj(aclmdlExecuteV2, handler);
  aclmdlCreateExecConfigHandle_ = DlsymAscendFuncObj(aclmdlCreateExecConfigHandle, handler);
  aclmdlDestroyExecConfigHandle_ = DlsymAscendFuncObj(aclmdlDestroyExecConfigHandle, handler);
  aclmdlSetExecConfigOpt_ = DlsymAscendFuncObj(aclmdlSetExecConfigOpt, handler);
  aclmdlFinalizeDump_ = DlsymAscendFuncObj(aclmdlFinalizeDump, handler);
  aclmdlGetCurOutputDims_ = DlsymAscendFuncObj(aclmdlGetCurOutputDims, handler);
  aclmdlGetDatasetBuffer_ = DlsymAscendFuncObj(aclmdlGetDatasetBuffer, handler);
  aclmdlGetDatasetNumBuffers_ = DlsymAscendFuncObj(aclmdlGetDatasetNumBuffers, handler);
  aclmdlGetDesc_ = DlsymAscendFuncObj(aclmdlGetDesc, handler);
  aclmdlGetInputDataType_ = DlsymAscendFuncObj(aclmdlGetInputDataType, handler);
  aclmdlGetInputDims_ = DlsymAscendFuncObj(aclmdlGetInputDims, handler);
  aclmdlGetInputIndexByName_ = DlsymAscendFuncObj(aclmdlGetInputIndexByName, handler);
  aclmdlGetInputNameByIndex_ = DlsymAscendFuncObj(aclmdlGetInputNameByIndex, handler);
  aclmdlGetInputSizeByIndex_ = DlsymAscendFuncObj(aclmdlGetInputSizeByIndex, handler);
  aclmdlGetNumInputs_ = DlsymAscendFuncObj(aclmdlGetNumInputs, handler);
  aclmdlGetNumOutputs_ = DlsymAscendFuncObj(aclmdlGetNumOutputs, handler);
  aclmdlGetOutputDataType_ = DlsymAscendFuncObj(aclmdlGetOutputDataType, handler);
  aclmdlGetOutputDims_ = DlsymAscendFuncObj(aclmdlGetOutputDims, handler);
  aclmdlQuerySizeFromMem_ = DlsymAscendFuncObj(aclmdlQuerySizeFromMem, handler);
  aclmdlGetOutputNameByIndex_ = DlsymAscendFuncObj(aclmdlGetOutputNameByIndex, handler);
  aclmdlGetOutputSizeByIndex_ = DlsymAscendFuncObj(aclmdlGetOutputSizeByIndex, handler);
  aclmdlInitDump_ = DlsymAscendFuncObj(aclmdlInitDump, handler);
  aclmdlLoadFromMem_ = DlsymAscendFuncObj(aclmdlLoadFromMem, handler);
  aclmdlSetDump_ = DlsymAscendFuncObj(aclmdlSetDump, handler);
  aclmdlSetDynamicBatchSize_ = DlsymAscendFuncObj(aclmdlSetDynamicBatchSize, handler);
  aclmdlUnload_ = DlsymAscendFuncObj(aclmdlUnload, handler);
  aclmdlBundleGetModelId_ = DlsymAscendFuncObj(aclmdlBundleGetModelId, handler);
  aclmdlBundleLoadFromMem_ = DlsymAscendFuncObj(aclmdlBundleLoadFromMem, handler);
  aclmdlBundleUnload_ = DlsymAscendFuncObj(aclmdlBundleUnload, handler);
  aclmdlLoadFromMemWithMem_ = DlsymAscendFuncObj(aclmdlLoadFromMemWithMem, handler);
  aclmdlSetDatasetTensorDesc_ = DlsymAscendFuncObj(aclmdlSetDatasetTensorDesc, handler);
  aclmdlGetInputFormat_ = DlsymAscendFuncObj(aclmdlGetInputFormat, handler);
  aclmdlGetDatasetTensorDesc_ = DlsymAscendFuncObj(aclmdlGetDatasetTensorDesc, handler);
  aclmdlSetInputDynamicDims_ = DlsymAscendFuncObj(aclmdlSetInputDynamicDims, handler);
  aclmdlGetOutputFormat_ = DlsymAscendFuncObj(aclmdlGetOutputFormat, handler);
  aclmdlGetInputDimsV2_ = DlsymAscendFuncObj(aclmdlGetInputDimsV2, handler);
  aclmdlGetDynamicHW_ = DlsymAscendFuncObj(aclmdlGetDynamicHW, handler);
  aclmdlGetInputDynamicDims_ = DlsymAscendFuncObj(aclmdlGetInputDynamicDims, handler);
  aclmdlGetInputDynamicGearCount_ = DlsymAscendFuncObj(aclmdlGetInputDynamicGearCount, handler);
  aclmdlGetDynamicBatch_ = DlsymAscendFuncObj(aclmdlGetDynamicBatch, handler);
  aclmdlSetDynamicHWSize_ = DlsymAscendFuncObj(aclmdlSetDynamicHWSize, handler);
#if defined(__linux__) && defined(WITH_BACKEND)
  aclmdlRICaptureBegin_ = DlsymAscendFuncObj(aclmdlRICaptureBegin, handler);
  aclmdlRICaptureGetInfo_ = DlsymAscendFuncObj(aclmdlRICaptureGetInfo, handler);
  aclmdlRICaptureEnd_ = DlsymAscendFuncObj(aclmdlRICaptureEnd, handler);
  aclmdlRIExecuteAsync_ = DlsymAscendFuncObj(aclmdlRIExecuteAsync, handler);
  aclmdlRIDestroy_ = DlsymAscendFuncObj(aclmdlRIDestroy, handler);
#endif

  MS_LOG(INFO) << "Load acl mdl api success!";
}

void LoadSimulationAclMdlApi() {
  ASSIGN_SIMU(aclmdlAddDatasetBuffer);
  ASSIGN_SIMU(aclmdlCreateDataset);
  ASSIGN_SIMU(aclmdlCreateDesc);
  ASSIGN_SIMU(aclmdlDestroyDataset);
  ASSIGN_SIMU(aclmdlDestroyDesc);
  ASSIGN_SIMU(aclmdlExecute);
  ASSIGN_SIMU(aclmdlExecuteV2);
  ASSIGN_SIMU(aclmdlCreateExecConfigHandle);
  ASSIGN_SIMU(aclmdlDestroyExecConfigHandle);
  ASSIGN_SIMU(aclmdlSetExecConfigOpt);
  ASSIGN_SIMU(aclmdlFinalizeDump);
  ASSIGN_SIMU(aclmdlGetCurOutputDims);
  ASSIGN_SIMU(aclmdlGetDatasetBuffer);
  ASSIGN_SIMU(aclmdlGetDatasetNumBuffers);
  ASSIGN_SIMU(aclmdlGetDesc);
  ASSIGN_SIMU(aclmdlGetInputDataType);
  ASSIGN_SIMU(aclmdlGetInputDims);
  ASSIGN_SIMU(aclmdlGetInputIndexByName);
  ASSIGN_SIMU(aclmdlGetInputNameByIndex);
  ASSIGN_SIMU(aclmdlGetInputSizeByIndex);
  ASSIGN_SIMU(aclmdlGetNumInputs);
  ASSIGN_SIMU(aclmdlGetNumOutputs);
  ASSIGN_SIMU(aclmdlGetOutputDataType);
  ASSIGN_SIMU(aclmdlGetOutputDims);
  ASSIGN_SIMU(aclmdlGetOutputNameByIndex);
  ASSIGN_SIMU(aclmdlGetOutputSizeByIndex);
  ASSIGN_SIMU(aclmdlInitDump);
  ASSIGN_SIMU(aclmdlLoadFromMem);
  ASSIGN_SIMU(aclmdlSetDump);
  ASSIGN_SIMU(aclmdlSetDynamicBatchSize);
  ASSIGN_SIMU(aclmdlUnload);
  ASSIGN_SIMU(aclmdlQuerySizeFromMem);
  ASSIGN_SIMU(aclmdlBundleGetModelId);
  ASSIGN_SIMU(aclmdlBundleLoadFromMem);
  ASSIGN_SIMU(aclmdlBundleUnload);
  ASSIGN_SIMU(aclmdlLoadFromMemWithMem);
  ASSIGN_SIMU(aclmdlSetDatasetTensorDesc);
  ASSIGN_SIMU(aclmdlGetInputFormat);
  ASSIGN_SIMU(aclmdlGetDatasetTensorDesc);
  ASSIGN_SIMU(aclmdlSetInputDynamicDims);
  ASSIGN_SIMU(aclmdlGetOutputFormat);
  ASSIGN_SIMU(aclmdlGetInputDimsV2);
  ASSIGN_SIMU(aclmdlGetDynamicHW);
  ASSIGN_SIMU(aclmdlGetInputDynamicDims);
  ASSIGN_SIMU(aclmdlGetInputDynamicGearCount);
  ASSIGN_SIMU(aclmdlGetDynamicBatch);
  ASSIGN_SIMU(aclmdlSetDynamicHWSize);
#if defined(__linux__) && defined(WITH_BACKEND)
  ASSIGN_SIMU(aclmdlRICaptureBegin);
  ASSIGN_SIMU(aclmdlRICaptureGetInfo);
  ASSIGN_SIMU(aclmdlRICaptureEnd);
  ASSIGN_SIMU(aclmdlRIExecuteAsync);
  ASSIGN_SIMU(aclmdlRIDestroy);
#endif
}
}  // namespace mindspore::device::ascend
