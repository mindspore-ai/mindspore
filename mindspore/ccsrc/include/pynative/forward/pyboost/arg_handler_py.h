/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_ARG_HANDLER_H
#define MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_ARG_HANDLER_H

#include <string>
#include <memory>
#include <vector>
#include <Python.h>
#include "ir/scalar.h"
#include "include/utils/visible.h"

namespace mindspore {

namespace pynative {

PYNATIVE_EXPORT PyObject *DtypeToTypeId(const std::string &op_name, const std::string &arg_name, PyObject *obj);

PYNATIVE_EXPORT PyObject *StrToEnum(const std::string &op_name, const std::string &arg_name, PyObject *obj);

PYNATIVE_EXPORT PyObject *ToPair(const std::string &op_name, const std::string &arg_name, PyObject *arg_val);

PYNATIVE_EXPORT PyObject *To2dPaddings(const std::string &op_name, const std::string &arg_name, PyObject *pad);

PYNATIVE_EXPORT PyObject *ToKernelSize(const std::string &op_name, const std::string &arg_name, PyObject *kernel_size);

PYNATIVE_EXPORT PyObject *ToStrides(const std::string &op_name, const std::string &arg_name, PyObject *stride);

PYNATIVE_EXPORT PyObject *ToDilations(const std::string &op_name, const std::string &arg_name, PyObject *dilation);

PYNATIVE_EXPORT PyObject *ToOutputPadding(const std::string &op_name, const std::string &arg_name,
                                          PyObject *output_padding);

PYNATIVE_EXPORT PyObject *ToRates(const std::string &op_name, const std::string &arg_name, PyObject *rates);

PYNATIVE_EXPORT PyObject *NormalizeIntSequence(const std::string &op_name, const std::string &arg_name,
                                               PyObject *arg_val);
PYNATIVE_EXPORT PyObject *ScalarTensorToScalar(const std::string &op_name, const std::string &arg_name,
                                               PyObject *arg_val);
PYNATIVE_EXPORT PyObject *ScalarTensorToInt(const std::string &op_name, const std::string &arg_name, PyObject *arg_val);
PYNATIVE_EXPORT PyObject *ScalarTensorToFloat(const std::string &op_name, const std::string &arg_name,
                                              PyObject *arg_val);

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_ARG_HANDLER_H
