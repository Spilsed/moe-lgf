from src.model.model import DeconstructedJetMoE, StatefulGate
from transformers import TrainerCallback

class ResetGateCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        for module in model.modules():
            if isinstance(module, StatefulGate):
                module.reset_state()

if __name__ == "__main__":
    from datasets import load_dataset
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

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

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
        save_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        callbacks=[ResetGateCallback()]
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained("./fine-tuned-jetmoe")