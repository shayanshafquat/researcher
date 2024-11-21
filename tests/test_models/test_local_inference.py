import pytest
from researcher.core.config.model_config import get_model_config, ModelProvider, update_model_config
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@pytest.fixture
def local_model_setup():
    # Ensure we're using local model
    update_model_config(ModelProvider.LOCAL)
    config = get_model_config().local
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        token=config.hf_api_key
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        token=config.hf_api_key
    )
    return model, tokenizer, config

def test_local_model_inference(local_model_setup):
    """Test if local model can generate text"""
    model, tokenizer, config = local_model_setup
    
    # Test prompt
    prompt = "What is the capital of France?"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Assertions
    assert isinstance(response, str)
    assert len(response) > len(prompt)
    assert response != prompt 