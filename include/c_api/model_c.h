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
#ifndef MINDSPORE_INCLUDE_C_API_MODEL_C_H
#define MINDSPORE_INCLUDE_C_API_MODEL_C_H

#include "include/c_api/tensor_c.h"
#include "include/c_api/context_c.h"
#include "include/c_api/status_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *MSModelHandle;

typedef struct MSTensorHandleArray {
  size_t handle_num;
  MSTensorHandle *handle_list;
} MSTensorHandleArray;

#define MS_MAX_SHAPE_NUM 32
typedef struct MSShapeInfo {
  size_t shape_num;
  int64_t shape[MS_MAX_SHAPE_NUM];
} MSShapeInfo;

typedef struct MSCallBackParamC {
  char *node_name;
  char *node_type;
} MSCallBackParamC;

typedef bool (*MSKernelCallBackC)(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                                  const MSCallBackParamC kernel_Info);

/// \brief Create a model object.
///
/// \return Model object handle.
MS_API MSModelHandle MSModelCreate();

/// \brief Destroy the model object.
///
/// \param[in] model Model object handle address.
MS_API void MSModelDestroy(MSModelHandle *model);

/// \brief Set workspace for the model object. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] workspace Define the workspace address.
/// \param[in] workspace_size Define the workspace size.
MS_API void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size);

/// \brief Calculate the workspace size required for model inference. Only valid for Iot.
///
/// \param[in] model Model object handle.
MS_API size_t MSModelCalcWorkspaceSize(MSModelHandle model);

/// \brief Build the model from model file buffer so that it can run on a device.
///
/// \param[in] model Model object handle.
/// \param[in] model_data Define the buffer read from a model file.
/// \param[in] data_size Define bytes number of model file buffer.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
///
/// \return MSStatus.
MS_API MSStatus MSModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,
                             const MSContextHandle model_context);

/// \brief Load and build the model from model path so that it can run on a device.
///
/// \param[in] model Model object handle.
/// \param[in] model_path Define the model file path.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
///
/// \return MSStatus.
MS_API MSStatus MSModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,
                                     const MSContextHandle model_context);

/// \brief Resize the shapes of inputs.
///
/// \param[in] model Model object handle.
/// \param[in] inputs The array that includes all input tensor handles.
/// \param[in] shape_infos Defines the new shapes of inputs, should be consistent with inputs.
/// \param[in] shape_info_num The num of shape_infos.
///
/// \return MSStatus.
MS_API MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos,
                              size_t shape_info_num);

/// \brief Inference model.
///
/// \param[in] model Model object handle.
/// \param[in] inputs The array that includes all input tensor handles.
/// \param[out] outputs The array that includes all output tensor handles.
/// \param[in] before CallBack before predict.
/// \param[in] after CallBack after predict.
///
/// \return MSStatus.
MS_API MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                               const MSKernelCallBackC before, const MSKernelCallBackC after);

/// \brief Run model by step. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] before CallBack before RunStep.
/// \param[in] after CallBack after RunStep.
///
/// \return MSStatus.
MS_API MSStatus MSModelRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after);

/// \brief Set the model running mode. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] train True means model runs in Train Mode, otherwise Eval Mode.
///
/// \return Status of operation.
MS_API MSStatus MSModelSetTrainMode(const MSModelHandle model, bool train);

/// \brief Export the weights of model to the binary file. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] export_path Define the export weight file path.
///
/// \return Status of operation.
MS_API MSStatus MSModelExportWeight(const MSModelHandle model, const char *export_path);

/// \brief Obtain all input tensor handles of the model.
///
/// \param[in] model Model object handle.
///
/// \return The array that includes all input tensor handles.
MS_API MSTensorHandleArray MSModelGetInputs(const MSModelHandle model);

/// \brief Obtain all output tensor handles of the model.
///
/// \param[in] model Model object handle.
///
/// \return The array that includes all output tensor handles.
MS_API MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model);

/// \brief Obtain the input tensor handle of the model by name.
///
/// \param[in] model Model object handle.
/// \param[in] tensor_name The name of tensor.
///
/// \return The input tensor handle with the given name, if the name is not found, an NULL is returned.
MS_API MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name);

/// \brief Obtain the output tensor handle of the model by name.
///
/// \param[in] model Model object handle.
/// \param[in] tensor_name The name of tensor.
///
/// \return The output tensor handle with the given name, if the name is not found, an NULL is returned.
MS_API MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char *tensor_name);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_MODEL_C_H
