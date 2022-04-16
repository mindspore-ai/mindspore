package com.mindspore.flclient;

/**
 * UT Case can be read from UT case file, This class defines the Http response info that get from case file.
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
public class FLHttpRes {
    private String resName;
    private int resCode;
    // 0: from file, 1: origin contentData
    private int contendMode;
    private String contentData;

    public String getResName() {
        return resName;
    }

    public void setResName(String resName) {
        this.resName = resName;
    }

    public int getResCode() {
        return resCode;
    }

    public void setResCode(int resCode) {
        this.resCode = resCode;
    }

    public int getContendMode() {
        return contendMode;
    }

    public void setContendMode(int contendMode) {
        this.contendMode = contendMode;
    }

    public String getContentData() {
        return contentData;
    }

    public void setContentData(String contentData) {
        this.contentData = contentData;
    }
}
