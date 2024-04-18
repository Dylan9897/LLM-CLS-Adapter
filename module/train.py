
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from tensorboardX import SummaryWriter
from collections import defaultdict
from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,logging
from logger import logger

def train(model,train_data_loader,valid_data_loader,test_data_loader):
    epoch = 10
    model = model.cuda()
    targets_weight_list = ["context_layer.fc.weight",'markov.fc.weight','hc1.fc.weight','reduce_dim.fc.weight','gru.input_gate.weight','gru.update_gate.weight','gru.reset_gate.weight']
    targets_bias_list = ["context_layer.fc.bias",'markov.fc.bias','hc1.fc.bias','reduce_dim.fc.bias','gru.input_gate.bias','gru.update_gate.bias','gru.reset_gate.bias']
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if n in targets_weight_list or ("lora" in n and "bias" not in n)
            ],
            "decay_parameters_module": [
                n for n, p in model.named_parameters() if n in targets_weight_list or ("lora" in n and "bias" not in n)
            ],
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n in targets_bias_list or ("lora" in n and "weight" not in n)
            ],
            "decay_parameters_module": [
                n for n, p in model.named_parameters() if n in targets_bias_list or ("lora" in n and "weight" not in n)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters,lr=5e-5,betas=[0.9,0.95],weight_decay=0.1,correct_bias=False)

    writer = SummaryWriter(log_dir="log"+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))

    total_steps = len(train_data_loader)*epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()
    
    
    
    history = defaultdict(list)
    best_accuracy = 0
    total_steps = 0
    for epoch in range(epoch):
        model = model.train()
        losses = []
        correct_predictions = 0
        for i,unit in enumerate(train_data_loader):
            
            input_ids = unit["input_ids"].cuda()
            attention_mask = unit["attention_mask"].cuda()
            targets = unit["labels"].cuda()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask = attention_mask
            )
            _,preds = torch.max(outputs,dim=1)
            loss = loss_fn(outputs,targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            loss.backward()
        
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_steps += 1
            if total_steps % 10 == 0:
                
                # for name, param in model.named_parameters():
                #     if name == "model.base_model.model.transformer.h.23.attn.c_proj.lora_B.default.weight":
                #         print(f"name is {name}, \n param is {param}")
                #         s = input()
                #     if name == "context_layer.fc.weight" :
                #         print(f"name is {name}, \n param is {param}")
                #         s = input()

                train_acc = correct_predictions.double()/6300
                train_loss = np.mean(losses)
                val_acc, val_loss = eval_model(
                                model,
                                valid_data_loader,
                                loss_fn,
                                "cuda",
                                350
                )
                
                logger.info("epoch:{},total step:{},Train loss is {},Train acc is {},valid loss is {},valid acc is {}".format(epoch,total_steps,train_loss,train_acc,val_loss,val_acc))
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join("ckpt",'best_model_state.ckpt'))
            best_accuracy = val_acc
    test_acc, _ = eval_model(model,test_data_loader,loss_fn,"cuda",350)
    logger.info(f"test result is {test_acc.item()}")
    y_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_data_loader)
    class_names = ['教育', '家居', '时尚', '时政', '科技', '房产', '财经']
    logger.info("accuracy is {}".format(test_acc))
    logger.info(classification_report(y_test, y_pred, target_names=[str(label) for label in class_names]))


def eval_model(model, data_loader, loss_fn, device,n_examples):
    model = model.eval() # 验证预测模式
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    model.train()
    return correct_predictions.double()/n_examples, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["texts"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return texts, predictions, prediction_probs, real_values
