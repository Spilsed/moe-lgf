from src.model.model import DeconstructedJetMoE, StatefulGate

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoConfig

    config = AutoConfig.from_pretrained("jetmoe/jetmoe-8b")
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0

    model_wrapper = DeconstructedJetMoE(StatefulGate, StatefulGate)
    model = model_wrapper.base
    tokenizer = model_wrapper.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./jetmoe-trainer",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        fp16=True,
        logging_steps=10,
        save_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained("./fine-tuned-jetmoe")