from flask import Flask, request, render_template, redirect, url_for
from transformers import AutoTokenizer, GPTJForCausalLM
import torch
import time
import json
model = GPTJForCausalLM.from_pretrained("/home/fleip/models/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("/home/fleip/models/gpt-j-6B")
tokenizer.pad_token_id = tokenizer.eos_token_id

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Open-Fnet')

@app.route('/submit', methods=['POST'])
def submit():
    prompt = request.form['prompt']
    global p 
    p = prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    gen_tokens = model.generate(input_ids, do_sample=True, num_return_sequences=4, max_length=100, early_stopping=True)

    outputs = []
    global outs
    outs = []
    for output in gen_tokens:
        outputs.append(tokenizer.decode(output, skip_special_tokens=True))
        outs.append(outputs[-1])

    return render_template('index.html', output1=outputs[0], output2=outputs[1], output3=outputs[2], output4=outputs[3])

@app.route('/rankings', methods=['POST'])
def rankings():
    ratings1 = request.form['rating1']
    ratings2 = request.form['rating2']
    ratings3 = request.form['rating3']
    ratings4 = request.form['rating4']

    print(p, outs)

    json_fmt = {
        "prompt": p,
        "output1": outs[0],
        "output2": outs[1],
        "output3": outs[2],
        "output4": outs[3],
        "rating1": ratings1,
        "rating2": ratings2,
        "rating3": ratings3,
        "rating4": ratings4
    }
    f = open("data/"+str(time.time()), 'w')
    f.write(json.dumps(json_fmt))
    f.close()

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="192.168.0.228", port=8080)