import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os

from importlib.util import spec_from_file_location, module_from_spec

# Locating the current directory and model file

current_directory = os.path.dirname(os.path.abspath(__file__))

model_file = os.path.join(current_directory, "2.py")
dataset_file = os.path.join(current_directory, "dataset.csv")
trained_model_file = os.path.join(current_directory, "simple_slm.pth")

# Importing ALBERT inspired prototype ML Model from 2.py


spec = spec_from_file_location("slm_model", model_file)
slm_model = module_from_spec(spec)

if spec is None or spec.loader is None:
    raise ImportError("Could not load the model from 2.py")

spec.loader.exec_module(slm_model)

SimpleSLM = slm_model.SimpleSLM



# Loading the previously generated dataset


inputs = []
targets = []

with open(dataset_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        x = float(row["input_x"])
        y = float(row["input_y"])
        output = float(row["output"])

        inputs.append([x, y])
        targets.append([output])





# Converting data to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)


print("Dataset loaded successfully.")
print("Number of training samples:", len(inputs))



# Initializing the model

model = SimpleSLM()

print("\nSimpleSLM model initiated successfully.")

print("\nInitial Weights:")
print(model.linear.weight)

print("\nInitial Bias:")
print(model.linear.bias)

criterion = nn.MSELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)



# Train the model

epochs = 1000

print("\nTraining started...\n")

for epoch in range(epochs):

    # Forward pass
    predictions = model(inputs)

    # Calculate loss
    loss = criterion(predictions, targets)

    # Remove previously calculated gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights and bias
    optimizer.step()

    # Display progress
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.6f}"
        )



# Display and save

print("\nTraining completed.")
print("\nLearned Weights:")
print(model.linear.weight)
print("\nLearned Bias:")
print(model.linear.bias)

torch.save(
    model.state_dict(),
    trained_model_file
)
print("\nTrained model saved successfully.")
print("Saved model:", trained_model_file)

