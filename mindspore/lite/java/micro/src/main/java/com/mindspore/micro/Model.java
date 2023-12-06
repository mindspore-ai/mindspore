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

package com.mindspore.micro;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;


public class Model {
    private static final Logger LOGGER = Logger.getLogger(MSTensor.class.toString());
    private long modelPtr = 0;

    /**
     * Construct function.
     */
    public Model() {
        this.modelPtr = 0;
    }

    /**
     * Build model.
     *
     * @param weightFile micro net.bin file.
     * @param context    model build context.
     * @return build status.
     */
    public boolean build(String weightFile, MSContext context) {
        if (context == null || weightFile == null) {
            return false;
        }
        if (modelPtr != 0) {
            this.free();
        }
        modelPtr = this.buildByWeight(weightFile, context.getMSContextPtr());
        return modelPtr != 0;
    }

    /**
     * Execute predict.
     *
     * @return predict status.
     */
    public boolean predict() {
        return this.runStep(modelPtr);
    }

    /**
     * Get model inputs tensor.
     *
     * @return input tensors.
     */
    public List<MSTensor> getInputs() {
        List<Long> ret = this.getInputs(modelPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Get model outputs.
     *
     * @return model outputs tensor.
     */
    public List<MSTensor> getOutputs() {
        List<Long> ret = this.getOutputs(modelPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Free model
     */
    public void free() {
        if (this.modelPtr != 0) {
            this.free(modelPtr);
            this.modelPtr = 0;
        } else {
            LOGGER.log(Level.SEVERE, "[Micro Model free] Pointer from java is nullptr.\n");
        }
    }

    private native long buildByWeight(String weightFile, long contextPtr);

    private native boolean runStep(long modelPtr);

    private native List<Long> getInputs(long modelPtr);

    private native List<Long> getOutputs(long modelPtr);

    private native void free(long modelPtr);
}
