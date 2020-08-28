package com.huawei.himindsporedemo.gallery.classify;

public class RecognitionObjectBean {

    private String name;
    private float score;

    public RecognitionObjectBean(String name, float score) {
        this.name = name;
        this.score = score;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public float getScore() {
        return score;
    }

    public void setScore(float score) {
        this.score = score;
    }


}
