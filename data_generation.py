import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Define possible roles
roles = ["Data Scientist", "Software Engineer", "Product Manager", "Sales Executive", 
         "Marketing Specialist", "HR Manager", "Business Analyst", "Project Manager"]

# Define the number of samples
num_samples = 10000

# Generate synthetic data
data = {
    "resume": [fake.text(max_nb_chars=2000) for _ in range(num_samples)],
    "role": [random.choice(roles) for _ in range(num_samples)]
}

# Create DataFrame
df_synthetic = pd.DataFrame(data)

# Save to CSV
df_synthetic.to_csv('resumes_train.csv', index=False)
print("Synthetic data generated and saved to 'resumes_train.csv'")