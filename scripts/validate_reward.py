import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

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

# state_preferred = (3, 1, 2, 0, 3)
# # Less Preferred: Low GPA, high fatigue, day 2, required, course starts early.
# state_less_preferred = (1, 3, 2, 1, 1)

# # Convert states into torch tensors (adding batch dimension)
# state_preferred_tensor = torch.tensor(state_preferred, dtype=torch.float32).unsqueeze(0)
# state_less_preferred_tensor = torch.tensor(state_less_preferred, dtype=torch.float32).unsqueeze(0)

# # Pass the states through the reward model
# reward_for_preferred = reward_model(state_preferred_tensor).item()
# reward_for_less_preferred = reward_model(state_less_preferred_tensor).item()

# print("Reward for preferred state:", reward_for_preferred)
# print("Reward for less preferred state:", reward_for_less_preferred)
extra_pairs = [
    {"preferred": (3, 1, 1, 0, 3), "less_preferred": (1, 3, 1, 1, 1)},
    {"preferred": (3, 1, 2, 0, 3), "less_preferred": (2, 2, 2, 1, 1)},
    {"preferred": (3, 1, 3, 0, 2), "less_preferred": (2, 3, 3, 1, 1)},
    {"preferred": (3, 1, 0, 0, 2), "less_preferred": (1, 3, 0, 1, 1)},
    {"preferred": (3, 1, 4, 0, 3), "less_preferred": (1, 3, 4, 1, 1)},
    {"preferred": (2, 1, 1, 0, 2), "less_preferred": (1, 2, 1, 1, 1)},
    {"preferred": (3, 1, 2, 0, 3), "less_preferred": (2, 2, 2, 1, 2)},
    {"preferred": (3, 1, 6, 0, 3), "less_preferred": (2, 3, 6, 1, 1)},
    {"preferred": (3, 1, 2, 0, 3), "less_preferred": (3, 3, 2, 1, 1)},
    {"preferred": (3, 2, 2, 0, 2), "less_preferred": (2, 3, 2, 1, 1)},
    {"preferred": (3, 1, 2, 0, 3), "less_preferred": (2, 3, 2, 1, 1)},
    # Pair 2: Comparison on day 3
    {"preferred": (3, 1, 3, 0, 2), "less_preferred": (1, 3, 3, 1, 1)},
    # Pair 3: Comparison on day 0
    {"preferred": (3, 1, 0, 0, 3), "less_preferred": (2, 3, 0, 1, 1)},
    # Pair 4: Comparison on day 4
    {"preferred": (3, 1, 4, 0, 3), "less_preferred": (2, 3, 4, 1, 2)},
    # Pair 5: Comparison on day 5
    {"preferred": (3, 1, 5, 0, 3), "less_preferred": (2, 2, 5, 1, 1)},
    # Pair 6: Comparison on day 1
    {"preferred": (3, 1, 1, 0, 2), "less_preferred": (1, 3, 1, 1, 1)},
    # Pair 7: Comparison on day 6
    {"preferred": (3, 1, 6, 0, 3), "less_preferred": (2, 3, 6, 1, 1)},
    # Pair 8: Comparison on day 2 with a mid start difference
    {"preferred": (3, 1, 2, 0, 2), "less_preferred": (2, 3, 2, 1, 1)},
    # Pair 9: Comparison on day 3, differing in start level
    {"preferred": (3, 1, 3, 0, 3), "less_preferred": (2, 3, 3, 1, 2)},
    # Pair 10: Comparison on day 0, differing in both start and required flag
    {"preferred": (3, 1, 0, 0, 3), "less_preferred": (2, 3, 0, 1, 2)},
    {"preferred": (3, 1, 2, 0, 3), "less_preferred": (2, 3, 2, 1, 1)},
    {"preferred": (3, 1, 4, 0, 3), "less_preferred": (2, 2, 4, 1, 1)},
    # Pair 2: Preference on day 0.
    {"preferred": (3, 1, 0, 0, 3), "less_preferred": (2, 3, 0, 1, 1)},
    # Pair 3: Preference on day 2.
    {"preferred": (3, 1, 2, 0, 2), "less_preferred": (1, 3, 2, 1, 1)},
    # Pair 4: Preference on day 3 with subtle differences.
    {"preferred": (3, 1, 3, 0, 3), "less_preferred": (2, 3, 3, 1, 2)},
    # Pair 5: Preference on day 1.
    {"preferred": (3, 1, 1, 0, 2), "less_preferred": (2, 3, 1, 1, 1)},
]

# 用来计数 preferred 状态得分更高的对数
correct = 0
# 为了计算 AUC，我们将所有的预测结果与对应的标签（preferred 为正例 1，less_preferred 为负例 0）分别存储
predictions = []
labels = []

# 对每一对状态进行评估
for pair in extra_pairs:
    # 将状态数据转换成 tensor，并增加 batch 维度
    state_preferred_tensor = torch.tensor(pair['preferred'], dtype=torch.float32).unsqueeze(0)
    state_less_tensor = torch.tensor(pair['less_preferred'], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        reward_preferred = reward_model(state_preferred_tensor).item()
        reward_less = reward_model(state_less_tensor).item()
    
    # 对比得分
    if reward_preferred > reward_less:
        correct += 1
    
    # 保存每个状态的预测结果和对应标签，用于计算 AUC
    predictions.extend([reward_preferred, reward_less])
    labels.extend([1, 0])

# 计算 Accuracy
accuracy = correct / len(extra_pairs)

# 计算 AUC-ROC
auc = roc_auc_score(labels, predictions)

# 使用 binomtest 进行二项检验，假设在随机预测下成功率应为 50%
binom_result = binomtest(correct, len(extra_pairs), p=0.5, alternative='greater')

# 打印结果
print("Accuracy:", accuracy)
print("AUC-ROC:", auc)
print("Binomial test result:", binom_result)