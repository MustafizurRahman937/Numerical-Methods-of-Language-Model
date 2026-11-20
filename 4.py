import torch
import os

# Import the SimpleSLM model from 2.py
from importlib.util import spec_from_file_location, module_from_spec


current_directory = os.path.dirname(os.path.abspath(__file__))

model_file = os.path.join(current_directory, "2.py")
trained_model_file = os.path.join(
    current_directory,
    "simple_slm.pth"
)


spec = spec_from_file_location("slm_model", model_file)
slm_model = module_from_spec(spec)

if spec is None or spec.loader is None:
    raise ImportError("Could not load the model from 2.py")

spec.loader.exec_module(slm_model)

SimpleSLM = slm_model.SimpleSLM


model = SimpleSLM()

model.load_state_dict(
    torch.load(
        trained_model_file,
        weights_only=True
    )
)

# Set the model to evaluation mode
model.eval()

print("Trained model loaded successfully.")

print("\nLearned Weights:")
print(model.linear.weight)

print("\nLearned Bias:")
print(model.linear.bias)



# Unseen test data
test_input = torch.tensor(
    [[4.0, 6.0]],
    dtype=torch.float32
)
actual_output = 10.0

#prediction
with torch.no_grad():
    prediction = model(test_input)

predicted_output = prediction.item()

# Calculate error
error = abs(predicted_output - actual_output)

# Display evaluation result
print("\nModel Evaluation")
print("-------------------------")

print("Test Input: x=4, y=6")
print("Predicted Output:", predicted_output)
print("Actual Output:", actual_output)
print("Absolute Error:", round(error, 4))

