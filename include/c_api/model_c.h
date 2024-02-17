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

typedef void *MSTrainCfgHandle;

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

/// \brief Create a TrainCfg object. Only valid for Lite Train.
///
/// \return TrainCfg object handle.
MS_API MSTrainCfgHandle MSTrainCfgCreate();

/// \brief Destroy the train_cfg object. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
MS_API void MSTrainCfgDestroy(MSTrainCfgHandle *train_cfg);

/// \brief Obtains part of the name that identify a loss kernel. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
/// \param[in] num The num of loss_name.
///
/// \return loss_name.
MS_API char **MSTrainCfgGetLossName(MSTrainCfgHandle train_cfg, size_t *num);

/// \brief Set part of the name that identify a loss kernel. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
/// \param[in] loss_name define part of the name that identify a loss kernel.
/// \param[in] num The num of loss_name.
MS_API void MSTrainCfgSetLossName(MSTrainCfgHandle train_cfg, const char **loss_name, size_t num);

/// \brief Obtains optimization level of the train_cfg. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
///
/// \return MSOptimizationLevel.
MS_API MSOptimizationLevel MSTrainCfgGetOptimizationLevel(MSTrainCfgHandle train_cfg);

/// \brief Set optimization level of the train_cfg. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
/// \param[in] level The optimization level of train_cfg.
MS_API void MSTrainCfgSetOptimizationLevel(MSTrainCfgHandle train_cfg, MSOptimizationLevel level);

/// \brief Build the train model from model buffer so that it can run on a device. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] model_data Define the buffer read from a model file.
/// \param[in] data_size Define bytes number of model file buffer.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
/// \param[in] train_cfg Define the config used by training.
///
/// \return MSStatus.
MS_API MSStatus MSTrainModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,
                                  const MSContextHandle model_context, const MSTrainCfgHandle train_cfg);

/// \brief Build the train model from model file buffer so that it can run on a device. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] model_path Define the model path.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
/// \param[in] train_cfg Define the config used by training.
///
/// \return MSStatus.
MS_API MSStatus MSTrainModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,
                                          const MSContextHandle model_context, const MSTrainCfgHandle train_cfg);

/// \brief Train model by step. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] before CallBack before predict.
/// \param[in] after CallBack after predict.
///
/// \return MSStatus.
MS_API MSStatus MSRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after);

/// \brief Sets the Learning Rate of the training. Only valid for Lite Train.
///
/// \param[in] learning_rate to set.
///
/// \return MSStatus of operation.
MS_API MSStatus MSModelSetLearningRate(MSModelHandle model, float learning_rate);

/// \brief Obtains the Learning Rate of the optimizer. Only valid for Lite Train.
///
/// \return Learning rate. 0.0 if no optimizer was found.
MS_API float MSModelGetLearningRate(MSModelHandle model);

/// \brief Obtains all weights tensors of the model. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
///
/// \return The vector that includes all gradient tensors.
MS_API MSTensorHandleArray MSModelGetWeights(MSModelHandle model);

/// \brief update weights tensors of the model. Only valid for Lite Train.
///
/// \param[in] new_weights A vector new weights.
///
/// \return MSStatus
MS_API MSStatus MSModelUpdateWeights(MSModelHandle model, const MSTensorHandleArray new_weights);

/// \brief Get the model running mode.
///
/// \param[in] model Model object handle.
///
/// \return Is Train Mode or not.
MS_API bool MSModelGetTrainMode(MSModelHandle model);

/// \brief Set the model running mode. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] train True means model runs in Train Mode, otherwise Eval Mode.
///
/// \return MSStatus.
MS_API MSStatus MSModelSetTrainMode(MSModelHandle model, bool train);

/// \brief Setup training with virtual batches. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] virtual_batch_multiplier - virtual batch multiplier, use any number < 1 to disable.
/// \param[in] lr - learning rate to use for virtual batch, -1 for internal configuration.
/// \param[in] momentum - batch norm momentum to use for virtual batch, -1 for internal configuration.
///
/// \return MSStatus.
MS_API MSStatus MSModelSetupVirtualBatch(MSModelHandle model, int virtual_batch_multiplier, float lr, float momentum);

/// \brief Export training model from file. Only valid for Lite Train.
///
/// \param[in] model The model data.
/// \param[in] model_type The model file type.
/// \param[in] model_file The exported model file.
/// \param[in] quantization_type The quantification type.
/// \param[in] export_inference_only Whether to export a reasoning only model.
/// \param[in] output_tensor_name The set the name of the output tensor of the exported reasoning model, default as
/// empty, and export the complete reasoning model.
/// \param[in] num The number of output_tensor_name.
///
/// \return MSStatus.
MS_API MSStatus MSExportModel(MSModelHandle model, MSModelType model_type, const char *model_file,
                              MSQuantizationType quantization_type, bool export_inference_only,
                              char **output_tensor_name, size_t num);

/// \brief Export training model from buffer. Only valid for Lite Train.
///
/// \param[in] model The model data.
/// \param[in] model_type The model file type.
/// \param[in] model_data The exported model buffer.
/// \param[in] data_size The exported model buffer size.
/// \param[in] quantization_type The quantification type.
/// \param[in] export_inference_only Whether to export a reasoning only model.
/// \param[in] output_tensor_name The set the name of the output tensor of the exported reasoning model, default as
/// empty, and export the complete reasoning model.
/// \param[in] num The number of output_tensor_name.
///
/// \return MSStatus.
MS_API MSStatus MSExportModelBuffer(MSModelHandle model, MSModelType model_type, char **model_data, size_t *data_size,
                                    MSQuantizationType quantization_type, bool export_inference_only,
                                    char **output_tensor_name, size_t num);

/// \brief Export model's weights, which can be used in micro only. Only valid for Lite Train.
///
/// \param[in] model The model data.
/// \param[in] model_type The model file type.
/// \param[in] weight_file The path of exported weight file.
/// \param[in] is_inference Whether to export weights from a reasoning model. Currently, only support this is `true`.
/// \param[in] enable_fp16 Float-weight is whether to be saved in float16 format.
/// \param[in] changeable_weights_name The set the name of these weight tensors, whose shape is changeable.
/// \param[in] num The number of changeable_weights_name.
///
/// \return MSStatus.
MS_API MSStatus MSExportWeightsCollaborateWithMicro(MSModelHandle model, MSModelType model_type,
                                                    const char *weight_file, bool is_inference, bool enable_fp16,
                                                    char **changeable_weights_name, size_t num);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_MODEL_C_H
