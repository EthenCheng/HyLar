import os
import pathlib
import torch
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from transformers.modeling_utils import unwrap_model

from src.trainer import CanvasSFTTrainer
from src.dataset import make_supervised_data_module_canvas
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.model.canvas_extractor import CanvasExtractor, CanvasExtractor_Siglip
from src.model.canvas import Canvas
from transformers import CLIPProcessor, SiglipImageProcessor, Siglip2Config
from transformers.modeling_utils import load_sharded_checkpoint

from train_utils import safe_save_model_for_hf_trainer
from src.train.monkey_patch_forward_canvas import replace_qwen2_5_with_canvas_forward

local_rank = None

# For debugging only Plese comment this during training
# torch.autograd.set_detect_anomaly(True)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with lvr
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    

    model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth)
    
    # Patch the forward function
    replace_qwen2_5_with_canvas_forward()


    print(model_pth)
    model = Canvas.from_pretrained(
        model_pth,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
    )
    

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    if "clip" in training_args.canvas_encoder:
        print("Loading canvas encoder: ", training_args.canvas_encoder)
        canvas_processor = CLIPProcessor.from_pretrained(training_args.canvas_encoder)
        canvas_config = AutoConfig.from_pretrained(training_args.canvas_encoder)
        canvas_extractor = CanvasExtractor(
            training_args.canvas_encoder, 
            canvas_token_num=data_args.canvas_token_num,
            llm_hidden_dim=model.config.hidden_size,
            config=canvas_config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )
    
    elif "siglip" in training_args.canvas_encoder:
        print("Loading canvas encoder: ", training_args.canvas_encoder)
        canvas_processor = SiglipImageProcessor.from_pretrained(training_args.canvas_encoder)
        canvas_config = AutoConfig.from_pretrained(training_args.canvas_encoder)
        canvas_extractor = CanvasExtractor_Siglip(
            training_args.canvas_encoder, 
            canvas_token_num=data_args.canvas_token_num,
            llm_hidden_dim=model.config.hidden_size,
            config=canvas_config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            use_compressor=training_args.use_canvas_compressor,
            compressor_n_heads=training_args.compressor_n_heads,
            compressor_n_layers=training_args.compressor_n_layers,
            adaptive_budget=training_args.adaptive_budget,
            max_n_queries=training_args.max_n_queries,
        )
    
    model.canvas_extractor = canvas_extractor

    if training_args.checkpoint_name:
        print(f"Loading full model state from checkpoint: {training_args.checkpoint_name}")
        load_sharded_checkpoint(model, training_args.checkpoint_name, strict=False)
        print("Successfully loaded weights from checkpoint.")
    else:
        print("Starting new training from base model weights.")


    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        model.enable_input_require_grads()

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)
    
    canvas_tokens = ["<|canvas|>", "<|canvas_start|>", "<|canvas_end|>"]
    processor.tokenizer.add_tokens(canvas_tokens, special_tokens=False)


    canvas_id = processor.tokenizer.convert_tokens_to_ids("<|canvas|>")
    canvas_start_id = processor.tokenizer.convert_tokens_to_ids("<|canvas_start|>")
    canvas_end_id = processor.tokenizer.convert_tokens_to_ids("<|canvas_end|>")

    model.config.canvas_id = canvas_id
    model.config.canvas_start_id = canvas_start_id
    model.config.canvas_end_id = canvas_end_id
    model.config.canvas_token_num = data_args.canvas_token_num
    model.config.compress_strategy = "compressor" if training_args.use_canvas_compressor else "average"

    # there are some dummy tokens in newer hf version
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    # configure canvas loss type
    model.config.canvas_loss = training_args.canvas_loss
    
    data_module = make_supervised_data_module_canvas(processor=processor,
                                                canvas_processor=canvas_processor,
                                                args=data_args)
    
    
    trainer = CanvasSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    raw_model = unwrap_model(trainer.model)
    
    if hasattr(raw_model, "canvas_extractor"):
        del raw_model.canvas_extractor
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
