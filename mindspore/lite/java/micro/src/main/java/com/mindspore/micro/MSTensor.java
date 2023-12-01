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

import java.nio.ByteBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;


public class MSTensor {
    private static final Logger LOGGER = Logger.getLogger(MSTensor.class.toString());
    private long tensorPtr;

    /**
     * MSTensor construct function.
     */
    public MSTensor() {
        this.tensorPtr = 0;
    }

    /**
     * MSTensor construct function.
     *
     * @param tensorPtr tensor pointer.
     */
    public MSTensor(long tensorPtr) {
        this.tensorPtr = tensorPtr;
    }

    /**
     * DataType is defined in com.mindspore.DataType.
     *
     * @return The MindSpore data type of the MindSpore MSTensor class.
     */
    public int getDataType() {
        return this.getDataType(this.tensorPtr);
    }

    /**
     * Get the shape of the MSTensor.
     *
     * @return A array of int as the shape of the MSTensor.
     */
    public int[] getShape() {
        return this.getShape(this.tensorPtr);
    }

    /**
     * Get output data of MSTensor, the data type is float.
     *
     * @return The float array containing all MSTensor output data.
     */
    public float[] getFloatData() {
        return this.getFloatData(this.tensorPtr);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of float[] type.
     * @return whether set data success.
     */
    public boolean setData(float[] data) {
        if (data == null) {
            return false;
        }
        return this.setFloatData(this.tensorPtr, data, data.length);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data data Input data of ByteBuffer type
     * @return whether set data success.
     */
    public boolean setData(ByteBuffer data) {
        if (data == null) {
            return false;
        }
        return this.setByteBufferData(this.tensorPtr, data);
    }

    /**
     * Get the size of the data in MSTensor in bytes.
     *
     * @return The size of the data in MSTensor in bytes.
     */
    public long size() {
        return this.size(this.tensorPtr);
    }

    /**
     * Free all temporary memory in MindSpore MSTensor.
     */
    public void free() {
        if (this.tensorPtr != 0) {
            this.free(this.tensorPtr);
            this.tensorPtr = 0;
        } else {
            LOGGER.log(Level.SEVERE, "[Micro MSTensor free] Pointer from java is nullptr.\n");
        }
    }

    private native int[] getShape(long tensorPtr);

    private native int getDataType(long tensorPtr);

    private native float[] getFloatData(long tensorPtr);

    private native boolean setFloatData(long tensorPtr, float[] data, long dataLen);

    private native boolean setByteBufferData(long tensorPtr, ByteBuffer buffer);

    private native long size(long tensorPtr);

    private native void free(long tensorPtr);
}
