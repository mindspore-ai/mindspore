/*
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

package com.mindspore.config;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * The AscendDeviceInfo class is used to configure MindSpore Lite Ascend device options.
 *
 * @since v2.0
 */
public class AscendDeviceInfo {
    /**
     * The device type of current class, which is Ascend.
     */
    private int deviceType = DeviceType.DT_ASCEND;

    /**
     * The target device ID.
     */
    private int deviceId = 0;

    /**
     * The ID of the current device in the cluster, which starts from 0.
     */
    private int rankId = 0;

    /**
     * The device provider, default is empty "" which uses acl as backend. Or "ge", which uses ge as backend.
     */
    private String provider = "";

    /**
     * AIPP configuration file path.
     */
    private String insertOpConfigPath = "";

    /**
     * Format of model inputs.
     */
    private String inputFormat = "";

    /**
     * Shape of model inputs.
     */
    private String inputShape = "";

    /**
     * Shape of model inputs.
     */
    private HashMap<Integer, ArrayList<Integer>> inputShapeMap = new HashMap<>();

    /**
     * Dynamic batch sizes of model inputs.
     */
    private ArrayList<Integer> dynamicBatchSize = new ArrayList<>();

    /**
     * Dynamic image sizes of model inputs.
     */
    private String dynamicImageSize = "";

    /**
     * Output type of model output.
     */
    private int outputType;

    /**
     * Precision mode of model.
     */
    private String precisionMode = "enforce_fp16";

    /**
     * Op select implementation mode.
     */
    private String opSelectImplMode = "high_performance";

    /**
     * Fusion switch file config path.
     */
    private String fusionSwitchConfigPath = "";

    /**
     * Buffer optimize mode.
     */
    private String bufferOptimizeMode = "l2_optimize";

    /**
     * Construct function.
     */
    public AscendDeviceInfo() {}

    /**
     * Get the Ascend device ID.
     *
     * @return the deviceId
     */
    public int getDeviceID() {
        return deviceId;
    }

    /**
     * Get the device type.
     *
     * @return the deviceType
     */
    public int getDeviceType() {
        return deviceType;
    }

    /**
     * Get Ascend device provider.
     *
     * @return the provider
     */
    public String getProvider() {
        return provider;
    }

    /**
     * Set Ascend device provider.
     *
     * @param provider the provider to set
     */
    public void setProvider(String provider) {
        this.provider = provider;
    }

    /**
     * Set the Ascend device ID.
     *
     * @param deviceId the deviceId to set
     */
    public void setDeviceID(int deviceId) {
        this.deviceId = deviceId;
    }

    /**
     * Get the logical ID of a distributed model in the cluster.
     *
     * @return the rankId
     */
    public int getRankID() {
        return rankId;
    }

    /**
     * Set the logical ID of a distributed model in the cluster.
     *
     * @param rankId the rankId to set
     */
    public void setRankID(int rankId) {
        this.rankId = rankId;
    }

    /**
     * Get the AIPP configuration file path.
     *
     * @return the insertOpConfigPath
     */
    public String getInsertOpConfigPath() {
        return insertOpConfigPath;
    }

    /**
     * Set the AIPP configuration file path.
     *
     * @param insertOpConfigPath AIPP configuration file path.
     */
    public void setInsertOpConfigPath(String insertOpConfigPath) {
        this.insertOpConfigPath = insertOpConfigPath;
    }

    /**
     * Get the input format of the model.
     *
     * @return the inputFormat
     */
    public String getInputFormat() {
        return inputFormat;
    }

    /**
     * Set the input format of the model.
     *
     * @param inputFormat Optional "NCHW", "NHWC", and "ND".
     */
    public void setInputFormat(String inputFormat) {
        this.inputFormat = inputFormat;
    }

    /**
     * Get the input shape of the model.
     *
     * @return the inputShape
     */
    public String getInputShape() {
        return inputShape;
    }

    /**
     * Set the input shape of the model.
     *
     * @param inputShape Model input shape. e.g. "input_op_name1:1,2,3,4;input_op_name2:4,3,2,1;"
     */
    public void setInputShape(String inputShape) {
        this.inputShape = inputShape;
    }

    /**
     * Get the input shape mapping of the model.
     *
     * @return the inputShapeMap
     */
    public HashMap<Integer, ArrayList<Integer>> getInputShapeMap() {
        return inputShapeMap;
    }

    /**
     * Model input shape. e.g. {{0, {1,2,3,4}}, {1, {4,3,2,1}}} means the first input shape is 1,2,3,4, and the second
     * input shape is 4,3,2,1.
     *
     * @param inputShapeMap the inputShapeMap to set.
     */
    public void setInputShapeMap(HashMap<Integer, ArrayList<Integer>> inputShapeMap) {
        this.inputShapeMap = inputShapeMap;
    }

    /**
     * Get the dynamic batch sizes of the model.
     *
     * @return the dynamicBatchSize
     */
    public ArrayList<Integer> getDynamicBatchSize() {
        return dynamicBatchSize;
    }

    /**
     *  Dynamic batch sizes of model inputs. Ranges from 2 to 100. e.g. {1, 2} means batch size 1 and 2 are configured.
     *
     * @param dynamicBatchSize the dynamicBatchSize to set.
     */
    public void setDynamicBatchSize(ArrayList<Integer> dynamicBatchSize) {
        this.dynamicBatchSize = dynamicBatchSize;
    }

    /**
     * Get the dynamic image sizes of the model.
     *
     * @return the dynamicImageSize
     */
    public String getDynamicImageSize() {
        return dynamicImageSize;
    }

    /**
     * Set the dynamic image sizes of the model.
     *
     * @param dynamicImageSize the dynamicImageSize to set
     */
    public void setDynamicImageSize(String dynamicImageSize) {
        this.dynamicImageSize = dynamicImageSize;
    }

    /**
     * Get the output type of the model.
     *
     * @return the outputType
     */
    public int getOutputType() {
        return outputType;
    }

    /**
     * Set the type of model outputs, can be DataType.kNumberTypeFloat32, DataType.kNumberTypeUInt8,
     * or DataType.kNumberTypeFloat16.
     *
     * @param outputType the outputType to set
     */
    public void setOutputType(int outputType) {
        this.outputType = outputType;
    }

    /**
     * Get the precision mode of the model.
     *
     * @return the precisionMode
     */
    public String getPrecisionMode() {
        return precisionMode;
    }

    /**
     * Set the precision mode.
     *
     * @param precisionMode Optional "enforce_fp16", "preferred_fp32", "enforce_origin", "enforce_fp32", and
     * "preferred_optimal". "enforce_fp16" is set as default.
     */
    public void setPrecisionMode(String precisionMode) {
        this.precisionMode = precisionMode;
    }

    /**
     * Get the operator selection implementation mode of the model.
     *
     * @return the opSelectImplMode
     */
    public String getOpSelectImplMode() {
        return opSelectImplMode;
    }

    /**
     * Set the operator selection implementation mode of the model.
     *
     * @param opSelectImplMode Optional "high_performance" and "high_precision". "high_performace" is set as default.
     */
    public void setOpSelectImplMode(String opSelectImplMode) {
        this.opSelectImplMode = opSelectImplMode;
    }

    /**
     * Set fusion switch config file path. It controls which fusion passes to be turned off.
     *
     * @return the fusionSwitchConfigPath
     */
    public String getFusionSwitchConfigPath() {
        return fusionSwitchConfigPath;
    }

    /**
     * Set the fusion switch config file path. It controls which fusion passes to be turned off.
     *
     * @param fusionSwitchConfigPath the fusionSwitchConfigPath to set
     */
    public void setFusionSwitchConfigPath(String fusionSwitchConfigPath) {
        this.fusionSwitchConfigPath = fusionSwitchConfigPath;
    }

    /**
     * Set the buffer optimize mode. Optional "l1_optimize", "l2_optimize", or "off_optimize". "l2_optimize" is set as
     * default.
     *
     * @return the bufferOptimizeMode
     */
    public String getBufferOptimizeMode() {
        return bufferOptimizeMode;
    }

    /**
     * Set the buffer optimization mode of the model.
     *
     * @param bufferOptimizeMode the bufferOptimizeMode to set
     */
    public void setBufferOptimizeMode(String bufferOptimizeMode) {
        this.bufferOptimizeMode = bufferOptimizeMode;
    }
}
