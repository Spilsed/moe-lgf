from src.model.model import DeconstructedJetMoE, StatefulGate
from transformers import TrainerCallback
from datasets import load_dataset, concatenate_datasets, Dataset

class ResetGateCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        for module in model.modules():
            if isinstance(module, StatefulGate):
                module.reset_state()

# def get_high_quality_data(tokenizer, max_length=512):    
#     refined = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True).take(10000)
#     math = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
#     # code = load_dataset("bigcode/starcoderdata", data_dir="python", split="train[:2%]")
#     # pes2o = load_dataset("allenai/peS2o", split="train", streaming=True).take(10000)

#     def tokenize_function(examples):
#         text = examples.get("text") or examples.get("content") or ""
#         return tokenizer(text, truncation=True, max_length=max_length)

#     print("Tokenizing...")
    
#     datasets = [refined, math]
#     tokenized_list = []
    
#     for ds in datasets:
#         cols = ds.column_names
#         tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=cols)
#         tokenized_list.append(tokenized_ds)

#     return concatenate_datasets(tokenized_list).shuffle(seed=42)

def get_high_quality_data(tokenizer, max_length=512):
    print("Fetching data slices...")
    
    # Define sources and their specific text columns
    total_rows = 100_000
    sources = [
        {"path": "open-web-math/open-web-math", "col": "text", "percent": 0.25 + 0.4},
        {"path": "tiiuae/falcon-refinedweb", "col": "content", "percent": 0.64 + 0.4},
        # {"path": "wikimedia/wikipedia", "col": "text", "percent": 0.04},
        # {"path": "armanc/scientific_papers", "col": "article", "percent": 0.04}
    ]
    
    all_rows = []

    for s in sources:
        print(f"Streaming {s['path']}...")
        stream = load_dataset(s['path'], split="train", streaming=True).take(int(total_rows * s['percent']))
        
        for entry in stream:
            text = entry.get(s['col'])
            if text:
                tokenized = tokenizer(text, truncation=True, max_length=max_length)
                all_rows.append(tokenized)

    print(f"Creating static dataset from {len(all_rows)} rows...")
    final_ds = Dataset.from_list(all_rows)
    return final_ds.shuffle()

if __name__ == "__main__":
    from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoConfig

    model_wrapper = DeconstructedJetMoE(StatefulGate, StatefulGate)
    model = model_wrapper.base
    tokenizer = model_wrapper.tokenizer

    for param in model.parameters():
        param.requires_grad = False

    trainable_params = 0
    for name, module in model.named_modules():
        if isinstance(module, StatefulGate):
            for param in module.parameters():
                param.requires_grad = True
                trainable_params += param.numel()

    print(f"Total trainable parameters: {trainable_params:,}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    full_dataset = get_high_quality_data(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./jetmoe-trainer",
        learning_rate=1e-4,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        fp16=True,
        use_cpu=False,
        logging_steps=10,
        save_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=data_collator,
        callbacks=[ResetGateCallback()]
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained("./fine-tuned-jetmoe")