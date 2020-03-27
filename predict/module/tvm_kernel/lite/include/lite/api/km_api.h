/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this ${file} except in compliance with the License.
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
#ifndef PREDICT_MODULE_TVM_KERNEL_LITE_INCLUDE_LITE_API_KM_API_H_
#define PREDICT_MODULE_TVM_KERNEL_LITE_INCLUDE_LITE_API_KM_API_H_

#include <dlpack/dlpack.h>
#include <functional>
#include <string>
#include <vector>
#include "schema/inner/ms_generated.h"
#include "schema/inner/op_generated.h"

#define PUBLIC __attribute__((visibility("default")))

/*!
 * \brief Call tvm kernel.
 * \param fid tvm kernel id.
 * \param tensors tvm kernel arguments.
 * \return 0 if SUCCESS.
 */
PUBLIC int CallKernel(const std::string &fid, const std::vector<DLTensor *> &tensors);

/*!
 * \brief Get tvm kernel by id.
 * \param fid tvm kernel id.
 * \return std::function if SUCCESS else nullptr.
 */
PUBLIC std::function<int(const std::vector<DLTensor *> &)> GetKernel(const std::string &fid);

/*!
 * \brief Get tvm kernel by OpDef.
 * \param opdef defined by predict schema.
 * \param tensors.
 * \param option.
 * \return std::function if SUCCESS else nullptr.
 */
struct PUBLIC KernelOption {
  int numThreads = 0;
  std::string device;
};

PUBLIC std::function<int(const std::vector<DLTensor *> &)> GetKernel(const mindspore::predict::OpDef &opdef,
                                                                     const std::vector<DLTensor *> &tensors,
                                                                     const KernelOption &option);

/*!
 * \brief load TVM Kernel lib
 * \param mode 0 indicate shared lib
 * \param fname shared lib path when mode equals 0
 * \return 0 if SUCCESS
 */
PUBLIC void InitKernelManager(int mode, const std::string &fname);

/*
 * \brief config ThreadPool using mode
 * \param mode: -1 using mid speed cpu first, 1 using higher speed cpu first
 * \param nthreads: threads num to be used, can't exceed cpu num
 *       if mode==-1  bind mid cpu first
 *       if mode==1   bind higher cpu first
 *       if mode==0   no bind
 * \param execute_self: cur thread do arithmetic or not
 *       execute_self: true  cur thread do arithmetic work
 *       execute_self: false  cur thread not do arithmetic work
 */
PUBLIC void ConfigThreadPool(int mode = -1, int nthreads = 2, bool execute_self = true);

/*
 * \brief provid simple api for mslite, mslite not care mode
 */
inline void CfgThreadPool(int nthread) { ConfigThreadPool(-1, nthread, true); }

/*
 *  the Callback function to do cpu bind for master thread.
 */
PUBLIC void DoMasterThreadBind(bool bindflg);

PUBLIC void DoAllThreadBind(bool ifBind);

#undef PUBLIC

#endif  // PREDICT_MODULE_TVM_KERNEL_LITE_INCLUDE_LITE_API_KM_API_H_
