"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
from lavis.models.blip2_models.osrt.layers import SlotAttention
from peft import LoraConfig, get_peft_model, TaskType
import transformers


@registry.register_model("blip2_opt_multiview")
class Blip2OPTMultiview(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            opt_model="facebook/opt-2.7b",
            prompt="",
            max_txt_len=32,
            apply_lemmatizer=False,
            views=6
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        self.check = True
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens, self.extra_query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16, load_in_8bit=True
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.slot_attention = SlotAttention(num_slots=32, input_dim=1408, slot_dim=1408, iters=1,
                                            randomize_initial_slots=False)
        self.view_pos_embedding = nn.Parameter(torch.zeros(views, 1408))

    def forward(self, samples):
        if self.check:
            print(samples["questions"])
            print(samples["answers"])
            self.check = False
        device = samples["vfeats"].device
        vfeats = samples["vfeats"]

        B = vfeats.shape[0]
        T = vfeats.shape[1]  # time dimension
        V = vfeats.shape[2]  # view dimension
        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image))
            if vfeats.dim() == 6 and V == 1:  # Single view case
                tmp_list = []
                for t in range(T):
                    curr_images = vfeats[:, t, 0, ...].squeeze(1).squeeze(1)
                    view_embeds = self.ln_vision(self.visual_encoder(curr_images))
                    view_embeds = self.slot_attention(view_embeds)
                    positional_embedding = self.view_pos_embedding[0].unsqueeze(0).unsqueeze(1) #1 1 1408
                    view_embeds = view_embeds + positional_embedding
                    tmp_list.append(view_embeds.unsqueeze(1))
                image_embeds = torch.cat(tmp_list, dim=1)  # Concatenate time steps
                image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])  # Flatten time
            elif vfeats.dim() == 6:
                tmp_list = []
                for t in range(T):
                    view_list = []
                    for v in range(V):
                        curr_images = vfeats[:, t, v, ...].squeeze(1).squeeze(1)
                        view_embeds = self.ln_vision(
                            self.visual_encoder(curr_images))
                        view_embeds = self.slot_attention(view_embeds)
                        positional_embedding = self.view_pos_embedding[v].unsqueeze(0).unsqueeze(1)
                        view_embeds = view_embeds + positional_embedding
                        view_list.append(view_embeds.unsqueeze(1))
                    time_embeds = torch.cat(view_list, dim=1)
                    tmp_list.append(time_embeds.unsqueeze(1))
                image_embeds = torch.cat(tmp_list, dim=1)
                image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1]) #6, 192, 1408
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            device
        ) #6, 192
        query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)  # 1 128 768
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1) #6 128 768
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ) #last hidden state 6 128 768

        inputs_opt = self.opt_proj(query_output.last_hidden_state)#6 128 2560
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)#6 128

        self.opt_tokenizer.padding_side = "right"

        # Handle text input and text output
        text_input = [t + "\n" for t in samples["questions"]]
        text_output = [t + "\n" for t in samples["answers"]]

        opt_tokens_input = self.opt_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len, #320
        ).to(device) # input ids 6, 27, attention mask 6, 27

        opt_tokens_output = self.opt_tokenizer(
            text_output,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)# input ids 6, 12, attention mask 6, 12

        targets = opt_tokens_output.input_ids.masked_fill(
            opt_tokens_output.input_ids == self.opt_tokenizer.pad_token_id, -100
        ) # 6, 12
        if self.prompt: #no prompt
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(device).fill_(-100)
        ) # 6 128
        empty_targets_questions = (
            torch.ones(opt_tokens_input.input_ids.size(), dtype=torch.long).to(device).fill_(-100)
        )# also pay no attention to questions
        targets = torch.cat([empty_targets, empty_targets_questions, targets], dim=1) #6, 140

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens_input.input_ids)#6 28 2560
        anwser_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens_output.input_ids)#6 28 2560
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds, anwser_embeds], dim=1)#6 155 2560
        attention_mask = torch.cat([atts_opt, opt_tokens_input.attention_mask, (
            torch.ones(opt_tokens_output.input_ids.size(), dtype=torch.long).to(device).fill_(0)
        )], dim=1)#6 155

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds, #6 167 2560
                attention_mask=attention_mask,# 6 167
                # output_attentions=opt_tokens_output.attention_mask,
                return_dict=True,
                labels=targets, #6 167
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            output_text = [text.strip() for text in output_text]
            return output_text

    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        if self.check:
            print(samples["questions"][0])
            print(samples["answers"][0])
            self.check = False

        device = samples["vfeats"].device
        vfeats = samples["vfeats"]

        B = vfeats.shape[0]
        T = vfeats.shape[1]  # time dimension
        V = vfeats.shape[2]  # view dimension
        with self.maybe_autocast():
            if vfeats.dim() == 6 and V == 1:  # Single view case
                tmp_list = []
                for t in range(T):
                    curr_images = vfeats[:, t, 0, ...].squeeze(1).squeeze(1)
                    view_embeds = self.ln_vision(self.visual_encoder(curr_images))
                    view_embeds = self.slot_attention(view_embeds)
                    positional_embedding = self.view_pos_embedding[0].unsqueeze(0).unsqueeze(1)
                    view_embeds = view_embeds + positional_embedding
                    tmp_list.append(view_embeds.unsqueeze(1))
                image_embeds = torch.cat(tmp_list, dim=1)  # Concatenate time steps
                image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])  # Flatten time
            elif vfeats.dim() == 6:  # Multi-view case
                tmp_list = []
                for t in range(T):
                    view_list = []
                    for v in range(V):
                        curr_images = vfeats[:, t, v, ...].squeeze(1).squeeze(1)
                        view_embeds = self.ln_vision(self.visual_encoder(curr_images))
                        view_embeds = self.slot_attention(view_embeds)
                        positional_embedding = self.view_pos_embedding[v].unsqueeze(0).unsqueeze(1)
                        view_embeds = view_embeds + positional_embedding
                        view_list.append(view_embeds.unsqueeze(1))
                    time_embeds = torch.cat(view_list, dim=1)  # Concatenate views
                    tmp_list.append(time_embeds.unsqueeze(1))
                image_embeds = torch.cat(tmp_list, dim=1)  # Concatenate time steps
                image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])  # Flatten time and views
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )
            #self.query token: 1,32,768, self.extra_query_tokens 1 96 768 is what I add additionally
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)  # 1 128 768
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                device
            )

            if isinstance(samples["questions"], str):
                samples["questions"] = [samples["questions"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["questions"]]
            else:
                text_input = samples["questions"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(device)

            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=5,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)


        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        for param in model.slot_attention.parameters():
            param.requires_grad = True

        model.extra_query_tokens.requires_grad = True

        return model