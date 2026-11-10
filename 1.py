import random
import csv
import os
def generate_sample():
    a = random.randint(1, 10)
    b = random.randint(1, 10)

    code = f"x = {a}\ny = {b}\nprint(x + y)"
    output = a + b

    return a, b, code, output

current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_file = os.path.join(current_directory, "dataset.csv")
dataset = [generate_sample() for _ in range(100)]
with open(dataset_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["input_x", "input_y", "code", "output"])
    for a, b, code, output in dataset:
        writer.writerow([a, b, code, output])
        
print("Dataset generated successfully.")
print("Dataset saved at:", dataset_file)
print("\nSample data:\n")

for a, b, code, output in dataset[:50]:
    print("Code:")
    print(code)
    print("Output:", output)
    print("------")
