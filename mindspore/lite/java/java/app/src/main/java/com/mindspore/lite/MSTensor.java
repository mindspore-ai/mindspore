/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

package com.mindspore.lite;

public class MSTensor {
    private long tensorPtr;

    public MSTensor() {
        this.tensorPtr = 0;
    }

    public MSTensor(long tensorPtr) {
        this.tensorPtr = tensorPtr;
    }

    public boolean init (int dataType, int[] shape) {
        this.tensorPtr = createMSTensor(dataType, shape, shape.length);
        return this.tensorPtr != 0;
    }

    public int[] getShape() {
        return this.getShape(this.tensorPtr);
    }

    public void setShape(int[] shape) {
        this.setShape(this.tensorPtr, shape, shape.length);
    }

    public int getDataType() {
        return this.getDataType(this.tensorPtr);
    }

    public void setDataType(int dataType) {
        this.setDataType(this.tensorPtr, dataType);
    }

    public byte[] getData() {
        return this.getData(this.tensorPtr);
    }

    public void setData(byte[] data) {
        this.setData(this.tensorPtr, data, data.length);
    }

    public long size() {
        return this.size(this.tensorPtr);
    }

    public int elementsNum() {
        return this.elementsNum(this.tensorPtr);
    }

    public void free() {
        this.free(this.tensorPtr);
        this.tensorPtr = 0;
    }

    private native long createMSTensor(int dataType, int[] shape, int shapeLen);

    private native int[] getShape(long tensorPtr);

    private native boolean setShape(long tensorPtr, int[] shape, int shapeLen);

    private native int getDataType(long tensorPtr);

    private native boolean setDataType(long tensorPtr, int dataType);

    private native byte[] getData(long tensorPtr);

    private native boolean setData(long tensorPtr, byte[] data, long dataLen);

    private native long size(long tensorPtr);

    private native int elementsNum(long tensorPtr);

    private native void free(long tensorPtr);
}
