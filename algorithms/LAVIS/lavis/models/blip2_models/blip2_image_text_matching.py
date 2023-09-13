"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import onnxruntime as ort
import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer


@registry.register_model("blip2_image_text_matching")
class Blip2ITM(Blip2Qformer):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

    # def forward(self, samples, match_head="itm"):
    #     image = samples["image"]
    #     caption = samples["text_input"]

    #     with self.maybe_autocast():
    #         image_embeds = self.ln_vision(self.visual_encoder(image))
    #     image_embeds = image_embeds.float()
    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
    #         image.device
    #     )

    #     text = self.tokenizer(
    #         caption,
    #         truncation=True,
    #         max_length=self.max_txt_len,
    #         return_tensors="pt",
    #     ).to(image.device)

    #     if match_head == "itm":
    #         query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #         print(f"itm_query_tokens_shape: {query_tokens.shape}")
    #         query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
    #             image.device
    #         )
    #         attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
    #         output_itm = self.Qformer.bert(
    #             text.input_ids,
    #             query_embeds=query_tokens,
    #             attention_mask=attention_mask,
    #             encoder_hidden_states=image_embeds,
    #             encoder_attention_mask=image_atts,
    #             return_dict=True,
    #         )
    #         print(f"output_itm.last_hidden_state_shape: {output_itm.last_hidden_state.shape}")
    #         itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
    #         itm_logit = self.itm_head(itm_embeddings)
    #         itm_logit = itm_logit.mean(dim=1)

    #         return itm_logit

    #     elif match_head == "itc":
    #         query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

    #         query_output = self.Qformer.bert(
    #             query_embeds=query_tokens,
    #             encoder_hidden_states=image_embeds,
    #             encoder_attention_mask=image_atts,
    #             return_dict=True,
    #         )

    #         print(f"img_query_output.last_hidden_state: {query_output.last_hidden_state.shape}")
    #         image_feats = F.normalize(
    #             self.vision_proj(query_output.last_hidden_state), dim=-1
    #         )

    #         print(f"image_feats_shape: {image_feats.shape}")

    #         text_output = self.Qformer.bert(
    #             text.input_ids,
    #             attention_mask=text.attention_mask,
    #             return_dict=True,
    #         )

    #         print(f"text_output.last_hidden_state: {text_output.last_hidden_state.shape}")

    #         text_feat = F.normalize(
    #             self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
    #         )

    #         print(f"text_feat_shape: {text_feat.unsqueeze(-1).shape}")

    #         sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
    #         print(f"sims_shape: {sims.shape}")
    #         # print(f"sims_values: {sims}")
    #         sim, _ = torch.max(sims, dim=1)
    #         print((f"sim_values: {sim}"))
    #         return sim
        
    def forward(self, image, text_input_ids, text_attention_mask):

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        print(f"image_embeds_shape: {image_embeds.shape}")
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # text = self.tokenizer(
        #     caption,
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        print(f"itm_query_tokens_shape: {query_tokens.shape}")
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        print(f"itm_query_atts_shape: {query_atts.shape}")
        attention_mask = torch.cat([query_atts, text_attention_mask], dim=1)

        itm_embeddings = self.Qformer.bert(
            text_input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )

        # onnx_model_path = '/data/xcao/code/uni_recognize_demo/algorithms/blip2_onnx_export/bert_constant.onnx'
        # self.Qformer.bert.eval()
        # torch.onnx.export(self.Qformer.bert,
        #                   (text_input_ids, query_tokens, attention_mask, image_embeds, image_atts),
        #                   onnx_model_path,
        #                   export_params=True,
        #                   opset_version=13,
        #                   do_constant_folding=True,
        #                   input_names=['input_ids','query_embeds','attention_mask', 'encoder_hidden_states', 'encoder_attention_mask'],
        #                   output_names = ['output']
        #                   )
        print(f"output_itm.last_hidden_state_shape: {itm_embeddings.shape}")
        print(f"query_tokens.size(1): {query_tokens.size(1)}")

        # print(f"output_itm.last_hidden_state_shape: {output_itm.last_hidden_state.shape}")
        # itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]

        # ort_session = ort.InferenceSession(onnx_model_path)
        # inputs = {
        #     'input_ids': text_input_ids.cpu().detach().numpy(),
        #     'query_embeds': query_tokens.cpu().detach().numpy(),
        #     'attention_mask': attention_mask.cpu().detach().numpy(),
        #     'encoder_hidden_states': image_embeds.cpu().detach().numpy(),
        #     'encoder_attention_mask': image_atts.cpu().detach().numpy()
        # }

        # ort_outs = ort_session.run(None, inputs)
        # print(f"img_ort_outs: {ort_outs[0].shape}")
        # print(f"img_ort_outs_value: {ort_outs}")

        # input_embedding = torch.tensor(ort_outs[0]).cuda()
        # itm_logit = self.itm_head(input_embedding)
        itm_logit = self.itm_head(itm_embeddings)
        itm_logit = itm_logit.mean(dim=1)

        itm_output = torch.nn.functional.softmax(itm_logit, dim=1)
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)

        # return itm_logit
        return itm_scores[:, 1]