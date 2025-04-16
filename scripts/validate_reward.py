import torch
import torch.nn as nn

# Define the more complex reward model as before
class RewardModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim1=32, hidden_dim2=16, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x  # Output shape: [batch_size, 1]

# Load the reward model from file
model_path = '/Users/zzlou/Documents/Code/cs3263/CS3263/best_reward_model_2.pt'
reward_model = RewardModel(input_dim=5, hidden_dim1=32, hidden_dim2=16, dropout_prob=0.2)
reward_model.load_state_dict(torch.load(model_path, map_location='cpu'))
reward_model.eval()

# Construct two contrasting states
# Preferred state by a lazy student:
# gpa_level = 3 (high), fatigue_level = 1 (low), day = 1, current_required = 0 (non-required), current_start_level = 3 (late start)
# state_preferred = (3, 1, 1, 0, 3)

# # Less preferred state:
# # gpa_level = 1 (low), fatigue_level = 3 (high), day = 1, current_required = 1 (required), current_start_level = 1 (early start)
# state_less_preferred = (1, 3, 1, 1, 1)

# state_preferred = (3, 1, 1, 0, 2)
# # Less Preferred: Moderate GPA, moderate fatigue, day 1, required, course starts early.
# state_less_preferred = (2, 2, 1, 1, 1)

state_preferred = (3, 1, 2, 0, 3)
# Less Preferred: Low GPA, high fatigue, day 2, required, course starts early.
state_less_preferred = (1, 3, 2, 1, 1)

# Convert states into torch tensors (adding batch dimension)
state_preferred_tensor = torch.tensor(state_preferred, dtype=torch.float32).unsqueeze(0)
state_less_preferred_tensor = torch.tensor(state_less_preferred, dtype=torch.float32).unsqueeze(0)

# Pass the states through the reward model
reward_for_preferred = reward_model(state_preferred_tensor).item()
reward_for_less_preferred = reward_model(state_less_preferred_tensor).item()

print("Reward for preferred state:", reward_for_preferred)
print("Reward for less preferred state:", reward_for_less_preferred)
