import smiles_gpt as gpt
from smiles_gpt.gpt_model import GPT2Config, GPT2LMHeadModel
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def train(checkpoint, data_path, gpus, num_workers, hyperparams, if_finetune=True):
    model_config = GPT2Config.from_pretrained(checkpoint)
    tokenizer_file = os.path.join(checkpoint, "tokenizer.json")
    tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer(
            tokenizer_file, 
            model_max_length=model_config.n_positions
        )

    data_module = gpt.LMDataModule(
            data_path, 
            tokenizer,
            batch_size=hyperparams["batch_size"],
            num_workers=num_workers,
        )
    data_module.setup()

    if if_finetune:
        model = GPT2LMHeadModel.from_pretrained(checkpoint)
    else:
        model = GPT2LMHeadModel(model_config)

    lit_model = gpt.GPT2LitModel(
            model,
            batch_size=hyperparams["batch_size"],
            learning_rate=hyperparams["learning_rate"],
            final_learning_rate=hyperparams["final_learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            adam_eps=hyperparams["adam_eps"],
            adam_betas=hyperparams["adam_betas"],
            scheduler_T_max=hyperparams["scheduler_T_max"],
            save_model_every=10, 
            checkpoint=checkpoint
        )

    trainer = Trainer(
            gpus=gpus,
            max_epochs=hyperparams["max_epochs"],
            callbacks=[EarlyStopping("ppl", 0.2, hyperparams["earlystop"])],
            auto_lr_find=False,  # Set to True to search for optimal learning rate.
            auto_scale_batch_size=False  # Set to True to scale batch size
        )

    trainer.fit(lit_model, data_module)


if __name__ == '__main__':
    save_checkpoint = "checkpoints/frag_token_2"  # ckpt18-7.285
    data_path = "data/frag_rotate_iso_mini2.txt"
    gpus = 1
    num_workers = 12
    if_finetune = True  # True
    train_for_redisc = False

    hyperparams = {
        "batch_size": 24,  # 64 32 for pretrain 16 13
        "max_epochs": 128, 
        "max_length": 1024,  # 768
        "learning_rate": 5e-4,
        "weight_decay": 0.0,
        "adam_eps": 1e-8,
        "adam_betas": (0.9, 0.999),
        "scheduler_T_max": 1_000,
        "final_learning_rate": 5e-8,
        "vocab_size": 10107,   # 10531, 1085, 
        "min_frequency": 2,
        "top_p": 0.96,
        "n_layer": 8,
        "n_head": 12,
        "n_embd": 384,
        "earlystop": 5,  # 10
    }

    if train_for_redisc or if_finetune:
        hyperparams["earlystop"] = 3
        hyperparams["learning_rate"] = 4e-4  # 3e-4

    train(save_checkpoint, data_path, gpus, num_workers, hyperparams, if_finetune=if_finetune)

