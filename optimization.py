import torch
import torch.nn.utils.prune as prune

# Reinitialize the model
vocab_size = 65
pruned_model = GPTLanguageModel(vocab_size)

# Load the pruned state dictionary
pruned_state_dict = torch.load("pruned_model.pth", map_location="cpu")

# Use strict=False to load the state dictionary without errors from missing keys
pruned_model.load_state_dict(pruned_state_dict, strict=False)

# Apply pruning masks to ensure the model is pruned
for name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Create a mask manually if not present in the state_dict
        weight_mask = getattr(module, "weight_mask", torch.ones_like(module.weight))
        prune.custom_from_mask(module, name='weight', mask=weight_mask)

# Remove pruning hooks
for name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, "weight")

# Set the model to evaluation mode
pruned_model.eval()

print("Pruned model loaded and pruning hooks removed.")

# Load base model for comparison
base_model = GPTLanguageModel(vocab_size=65)
base_model.load_state_dict(torch.load("final_model.pth", map_location="cpu"))
base_model.eval()

# Define and quantize the model before loading the state dict
quantized_model = torch.quantization.quantize_dynamic(
    GPTLanguageModel(vocab_size=65),  # Model definition
    {torch.nn.Linear},  # Specify layers to quantize
    dtype=torch.qint8
)
quantized_model.load_state_dict(torch.load("quantized_model.pth", map_location="cpu"))
quantized_model.eval()

# Continue with performance evaluation
