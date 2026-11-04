import gradio as gr
import os
from PIL import Image
import json
import re
import random
import argparse
def remove_extra_commas(json_str):
    fixed_json = re.sub(r',\s*([\]}])', r'\1', json_str)
    return fixed_json

def convert_label(cls_select):
    if cls_select == "vru":
        return "vulnerable_road_users"
    elif cls_select == "traffic_signs":
        return "traffic signs"
    elif cls_select == "traffic_lights":
        return "traffic lights"
    elif cls_select == "traffic_cones":
        return "traffic cones"
    elif cls_select == "other_objects":
        return "other objects"
    else:
        return cls_select
    
def show(data_root, version, text, cls_select):
    assert version in ['Train', 'Val', 'Test', 'Mini']
    if version in ['Train']:
        images_type = 'val_'
    else:
        images_type = 'test_'
    data_path = os.path.join(data_root, version)
    img_name = images_type + text.zfill(4) + '.jpg'
    json_name = images_type + text.zfill(4) + '.json'
    img_root, img_id = img_name.replace(".jpg", "").split("_")
    new_label = convert_label(cls_select)
    img_path = os.path.join(data_root, "..",img_root, "images", img_id + ".jpg")
    img = Image.open(img_path)
    json_path = os.path.join(data_path, json_name)
    custum_sum = 0
    info_list = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            parsed_json = json.load(f)
        suggestion = parsed_json['driving_suggestion']
        parsed_json = parsed_json['general_perception']
        parsed_json['suggestions'] = suggestion
        if new_label in parsed_json.keys():
            info = parsed_json[new_label]
        else:
            info = parsed_json[cls_select]
        if new_label == 'suggestions':
            info_list.append(info)
        else:
            filtered_info = [d for d in info if d]
            for idx, item in enumerate(filtered_info):
                info_list.append("description: " + item['description'] + "\n" + "explanation: " + item['explanation'])
            custum_sum += len(filtered_info)
    info_list = info_list
    remove_list = [""] * (14 - len(info_list))
    return img, *info_list, *remove_list

def save_text(save_path, version, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, final_dro, text, cls_select):
    d_list = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14]
    assert version in ['Train', 'Val', 'Test', 'Mini']
    if version in ['Train']:
        json_type = 'val_'
    else:
        json_type = 'test_'
    good_text = ""
    modify_text = ""
    for d in d_list:
        if d == "good":
            good_text = good_text + '1 ' 
        else:
            good_text = good_text + '0 '
        if d == 'modify':
            modify_text = modify_text +'1 '
        else:
            modify_text = modify_text +'0 '
    if final_dro == "complete":
        complete_info = 1
    else:
        complete_info = 0
    complete_text_name = 'complete_' + cls_select +'.txt'
    good_txt_name = 'good_' + cls_select + '.txt'
    modify_txt_name = 'modify_' + cls_select + '.txt'
    refine_save_path = os.path.join(save_path, version, json_type + text.zfill(4))
    os.makedirs(refine_save_path, exist_ok=True)
    with open(os.path.join(refine_save_path, good_txt_name), 'w') as fp:
        fp.write(good_text)
    with open(os.path.join(refine_save_path, modify_txt_name), 'w') as fp:
        fp.write(modify_text)
    with open(os.path.join(refine_save_path, complete_text_name), 'w') as fp:
        fp.write(str(complete_info))
    return "Submit successfully!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="CODA/CODA-LM")
    parser.add_argument("--version", type=str, default="Test")
    parser.add_argument("--save_path", type=str, default="./Refine")
    args = parser.parse_args()
    rules_markdown = """ ### Guidelines
- Input the image number (ranging from 1 to 4884) along with the specific category you wish to examine, then press "Display" to reveal both the image and its associated annotation data.
- According to the annotation rules, select "good", "modify", or "delete" for each text box.
- Based on the selected good and modify annotations, assess whether the image completely describes the salient objects of that category, marking it as either "complete" or "incomplete".
- After making your selections, click "Submit". A "Submit successfully!" message will display upon successful submission.
- Click "Clear" to refresh the page and start a new round.
"""
    with gr.Blocks() as demo:
        data_root = gr.State(value = args.data_root)
        version = gr.State(value = args.version)
        save_path = gr.State(value = args.save_path)
        gr.Markdown(rules_markdown)
        with gr.Row():\
            
            with gr.Column(scale=20):
                text = gr.Textbox(
                    label="Image name",
                    placeholder="Input image name: 1"
                )
            with gr.Column(scale=5):
                cls_select = gr.Dropdown(["vehicles", "vru", "traffic_signs", "traffic_lights", "traffic_cones", "barriers", "other_objects", "suggestions"], label="Class selection")
        with gr.Row():
            display_btn = gr.Button(value="Display")

        with gr.Row():
            with gr.Column(scale=1):
                img1 = gr.Image()
        with gr.Column():
            with gr.Row():
                text1 = gr.Textbox(scale=3, show_label = False)
                tie_dro1 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text2 = gr.Textbox(scale=3, show_label = False)
                tie_dro2 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text3 = gr.Textbox(scale=3, show_label = False)
                tie_dro3 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text4 = gr.Textbox(scale=3, show_label = False)
                tie_dro4 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text5 = gr.Textbox(scale=3, show_label = False)
                tie_dro5 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text6 = gr.Textbox(scale=3, show_label = False)
                tie_dro6 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text7 = gr.Textbox(scale=3, show_label = False)
                tie_dro7 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text8 = gr.Textbox(scale=3, show_label = False)
                tie_dro8 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text9 = gr.Textbox(scale=3, show_label = False)
                tie_dro9 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text10 = gr.Textbox(scale=3, show_label = False)
                tie_dro10 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text11 = gr.Textbox(scale=3, show_label = False)
                tie_dro11 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text12 = gr.Textbox(scale=3, show_label = False)
                tie_dro12 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text13 = gr.Textbox(scale=3, show_label = False)
                tie_dro13 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                text14 = gr.Textbox(scale=3, show_label = False)
                tie_dro14 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
            with gr.Row():
                final_dro = gr.Dropdown(["complete","incomplete"], label="Ann Complete choice", scale=2)
                btn = gr.Button("Submit", scale=2)
                text_output = gr.Textbox(label="Output", scale=2)
            clear_btn = gr.Button(value="🗑️ Clear")

        img_list = [img1]
        text_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14]
        btn_list = [display_btn, btn, clear_btn]
        drop_down_list = [tie_dro1, tie_dro2, tie_dro3, tie_dro4, tie_dro5, tie_dro6, tie_dro7, tie_dro8, tie_dro9, tie_dro10, tie_dro11, tie_dro12, tie_dro13, tie_dro14, final_dro]
        display_btn.click(show, inputs=[data_root, version, text, cls_select], outputs=img_list + text_list)   
        btn.click(save_text, inputs=[save_path, version] + drop_down_list + [text, cls_select], outputs=text_output)
        
        clear_btn.click(lambda: ["", None, "", None, "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "Display", "Submit", "Clear"], 
            None, [text, cls_select, text_output] + img_list + text_list + drop_down_list + btn_list)

    demo.launch(share=True)