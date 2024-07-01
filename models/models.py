import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import GPT2Config

from .pocketTransformer import PocketTransformer3D
from .utils import accuracy2


class FragSmilesPocketGPT(GPT2LMHeadModel):
    def __init__(self, cfg, task=None, Tokenizer=None):
        if task is not None:
            tokenizer = task.tokenizer
        elif Tokenizer is not None:
            tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=cfg.MODEL.GPT_MODEL.n_layer,
            n_head=cfg.MODEL.GPT_MODEL.n_head,
            n_embd=cfg.MODEL.GPT_MODEL.n_embd,
            n_positions=cfg.DATA.MAX_SMILES_LEN,
            n_ctx=cfg.DATA.MAX_SMILES_LEN,
            add_cross_attention=True,
        )
        super().__init__(config)
        self.cfg = cfg

        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.encoder = PocketTransformer3D(**cfg.MODEL.POCKET_ENCODER)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pocket_seq: Optional[torch.LongTensor] = None,
        pocket_edge_type: Optional[torch.LongTensor] = None,
        pocket_dis: Optional[torch.FloatTensor] = None,
        pocket_coords: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        if_gen=False,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        input_ids = input_ids.to(self.device)
        pocket_seq = pocket_seq.to(self.device)
        pocket_edge_type = pocket_edge_type.to(self.device)
        pocket_dis = pocket_dis.to(self.device)
        pocket_coords = pocket_coords.to(self.device)

        if not if_gen:
            labels = labels.to(self.device)

        encoder_rep, padding_mask = self.encoder(
            src_tokens=pocket_seq,
            src_distance=pocket_dis,
            src_edge_type=pocket_edge_type,
            src_coords=pocket_coords,
        )
        # encoder_rep, encoder_attention_mask = None, None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # inputs_embeds = self.transformer.wte(input_ids)

        encoder_attention_mask = 1 - padding_mask.type(torch.int64)

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_rep,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc = accuracy2(lm_logits[:, :-1], labels[:, 1:])

        if self.training:
            return {'loss': loss, 'hit@1': acc}
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )


        


