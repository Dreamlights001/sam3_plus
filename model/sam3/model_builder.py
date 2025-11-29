# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from .model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from .model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .model.geometry_encoders import SequenceGeometryEncoder
from .model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .model.necks import Sam3DualViTDetNeck
from .model.position_encoding import PositionEmbeddingSine
from .model.sam3_image import Sam3Image
from .model.text_encoder_ve import VETextEncoder
from .model.tokenizer_ve import SimpleTokenizer
from .model.vitdet import ViT
from .model.vl_combiner import SAM3VLBackbone



# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder with enhanced error handling."""
    import traceback
    import sys
    
    print(f"=== Starting _create_text_encoder with bpe_path: {bpe_path} ===")
    
    try:
        # Check if bpe_path is a directory containing Hugging Face tokenizer files
        print(f"Checking if bpe_path is directory: {os.path.isdir(bpe_path)}")
        if os.path.isdir(bpe_path):
            model_dir = bpe_path
        else:
            model_dir = os.path.dirname(bpe_path) if not bpe_path.endswith('.gz') else os.path.dirname(os.path.dirname(bpe_path))
        
        print(f"Using model directory: {model_dir}")
        
        # Check directory existence
        if not os.path.exists(model_dir):
            print(f"ERROR: Model directory does not exist: {model_dir}")
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
        
        # List directory contents for debugging
        print(f"Contents of model directory:")
        for item in os.listdir(model_dir):
            print(f"  - {item}")
        
        tokenizer_json_path = os.path.join(model_dir, 'tokenizer.json')
        vocab_json_path = os.path.join(model_dir, 'vocab.json')
        
        print(f"Checking tokenizer files:")
        print(f"  tokenizer.json: {os.path.exists(tokenizer_json_path)}")
        print(f"  vocab.json: {os.path.exists(vocab_json_path)}")
        
        if os.path.exists(tokenizer_json_path) and os.path.exists(vocab_json_path):
            # Use Hugging Face tokenizer if available
            print(f"📝 Found Hugging Face tokenizer files, attempting to use them")
            
            # Check if transformers library is installed
            print("Checking if transformers library is installed...")
            try:
                import transformers
                print(f"Transformers library version: {transformers.__version__}")
            except ImportError as e:
                print(f"❌ Transformers library not found: {e}")
                print("Attempting to install transformers library...")
                # Try to install transformers if not available
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
                    print("✅ Transformers library installed successfully")
                    import transformers
                    print(f"Transformers library version: {transformers.__version__}")
                except Exception as install_error:
                    print(f"❌ Failed to install transformers: {install_error}")
                    raise RuntimeError("Transformers library not found and failed to install. Please install it manually with 'pip install transformers'")
            
            # Now try to import AutoTokenizer
            try:
                from transformers import AutoTokenizer
                print("✅ Successfully imported AutoTokenizer")
                
                # Attempt to create the tokenizer
                print(f"Creating tokenizer from directory: {model_dir}")
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                print("✅ Successfully created tokenizer")
                print(f"Tokenizer type: {type(tokenizer)}")
                print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
                
                # Create VETextEncoder
                print("Creating VETextEncoder...")
                text_encoder = VETextEncoder(
                    tokenizer=tokenizer,
                    d_model=256,
                    width=1024,
                    heads=16,
                    layers=24,
                )
                print("✅ Successfully created VETextEncoder")
                print(f"Text encoder type: {type(text_encoder)}")
                
                print("=== _create_text_encoder completed successfully ===")
                return text_encoder
                
            except Exception as e:
                print(f"❌ Error creating tokenizer or text encoder: {e}")
                print("Detailed error:")
                traceback.print_exc()
                raise
        else:
            # No Hugging Face tokenizer files found
            missing_files = []
            if not os.path.exists(tokenizer_json_path):
                missing_files.append(f"tokenizer.json (expected at {tokenizer_json_path})")
            if not os.path.exists(vocab_json_path):
                missing_files.append(f"vocab.json (expected at {vocab_json_path})")
            
            error_msg = f"Missing required tokenizer files: {', '.join(missing_files)}"
            print(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
            
    except Exception as e:
        print(f"CRITICAL ERROR in _create_text_encoder: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise
    
    print("This line should never be reached")


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    try:
        print(f"Attempting to load checkpoint from: {checkpoint_path}")
        # Check if file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Get file size
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"Checkpoint file size: {file_size:.2f} MB")
        
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            print("Loading checkpoint file...")
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
            print(f"Checkpoint loaded into CPU memory")
        
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            print("Using 'model' key from checkpoint")
            ckpt = ckpt["model"]
        
        sam3_image_ckpt = {
            k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
        }
        
        if model.inst_interactive_predictor is not None:
            print("Updating checkpoint for instance interactive predictor")
            sam3_image_ckpt.update(
                {
                    k.replace("tracker.", "inst_interactive_predictor.model."): v
                    for k, v in ckpt.items()
                    if "tracker" in k
                }
            )
        
        missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
        if len(missing_keys) > 0:
            print(
                f"Loaded {checkpoint_path} and found "
                f"missing and/or unexpected keys:\n{missing_keys=}"
            )
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        raise


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    try:
        print(f"Moving model to device: {device}")
        if device == "cuda":
            model = model.cuda()
        else:
            model = model.to(device)
        print(f"Model successfully moved to {device}")
        # Set evaluation mode if specified
        if eval_mode:
            print("Setting model to evaluation mode")
            model.eval()
        else:
            print("Model in training mode")
        return model
    except Exception as e:
        print(f"Error setting up device and mode: {e}")
        import traceback
        traceback.print_exc()
        raise


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile_mode: To enable compilation, set to "default"

    Returns:
        A SAM3 image model
    """
    # Check if checkpoint_path is a directory, if so, use it as the model directory
    model_dir = None
    if checkpoint_path and os.path.isdir(checkpoint_path):
        model_dir = checkpoint_path
    elif checkpoint_path:
        model_dir = os.path.dirname(checkpoint_path)
    
    # If bpe_path is not provided, check if we have Hugging Face tokenizer files in model_dir
    if bpe_path is None:
        if model_dir:
            # Check if we have Hugging Face tokenizer files
            tokenizer_json_path = os.path.join(model_dir, 'tokenizer.json')
            vocab_json_path = os.path.join(model_dir, 'vocab.json')
            if os.path.exists(tokenizer_json_path) and os.path.exists(vocab_json_path):
                # We have Hugging Face tokenizer files, no need for bpe_path
                print(f"📝 Found Hugging Face tokenizer files in {model_dir}")
                # Set a dummy bpe_path since we'll use Hugging Face tokenizer
                bpe_path = model_dir
            else:
                # No Hugging Face tokenizer files found, raise error
                raise FileNotFoundError(
                    f"No tokenizer files found in {model_dir}. Please ensure the directory contains tokenizer.json and vocab.json files."
                )
        else:
            # No model_dir provided, raise error
            raise ValueError(
                "Either bpe_path or a valid model_dir with tokenizer files must be provided."
            )
    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def download_ckpt_from_hf():
    SAM3_MODEL_ID = "facebook/sam3"
    SAM3_CKPT_NAME = "sam3.pt"
    SAM3_CFG_NAME = "config.json"
    _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)
    checkpoint_path = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME)
    return checkpoint_path



