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

package com.mindspore.flclient.cipher.struct;

public class DecryptShareSecrets {
    private String flID;
    private NewArray<byte[]> sSkVu;
    private NewArray<byte[]> bVu;
    private int sIndex;
    private int indexB;

    public String getFlID() {
        return flID;
    }

    public void setFlID(String flID) {
        this.flID = flID;
    }

    public NewArray<byte[]> getSSkVu() {
        return sSkVu;
    }

    public void setSSkVu(NewArray<byte[]> sSkVu) {
        this.sSkVu = sSkVu;
    }

    public NewArray<byte[]> getBVu() {
        return bVu;
    }

    public void setBVu(NewArray<byte[]> bVu) {
        this.bVu = bVu;
    }

    public int getSIndex() {
        return sIndex;
    }

    public void setSIndex(int sIndex) {
        this.sIndex = sIndex;
    }

    public int getIndexB() {
        return indexB;
    }

    public void setIndexB(int indexB) {
        this.indexB = indexB;
    }
}
