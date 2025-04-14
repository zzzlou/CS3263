import torch
import torch.nn as nn
import torch.optim as optim
import json
import random

# --------------------------
# 定义 Reward Model
# --------------------------
class RewardModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出单个标量

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # 输出形状 [batch_size, 1]

# --------------------------
# 计算一个轨迹的累计奖励
# --------------------------
def compute_trajectory_reward(trajectory, reward_model, device='cpu'):
    """
    trajectory: 一个 list，每个元素为一个 state，state 为 5 维 (earliest_start, total_attended, gpa, fatigue, day)
    对于每个 state 用 reward_model 计算即时 reward，然后累加
    """
    total_reward = 0.0
    for state in trajectory:
        # 将 state 转换为 1 x 5 的 tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        r = reward_model(state_tensor)
        total_reward += r.item()  # 累加即时 reward
    return total_reward

# --------------------------
# 训练 Reward Model 的流程（含验证、early stopping 和模型保存）
# --------------------------
def train_reward_model(reward_model, optimizer, train_data, val_data, num_epochs=20, device='cpu', patience=5, model_save_path='best_reward_model.pt'):
    """
    train_data, val_data: list，每个元素为 (traj1, traj2, human_feedback)
      - traj1 与 traj2 均为 state 的 list, 每个 state 为 5 维向量
      - human_feedback: 1 表示 traj1 被偏好，0 表示 traj2 被偏好
    使用 BCEWithLogitsLoss 训练 reward_model，使得累计 reward 差 d = R(traj1) - R(traj2)
    与人类反馈一致。
    
    增加 early stopping：如果连续 `patience` 个 epoch 内验证集 loss 没有下降，则提前终止训练。
    同时保存验证集 loss 最低时的模型参数到 model_save_path。
    """
    criterion = nn.BCEWithLogitsLoss()
    reward_model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        reward_model.train()
        total_train_loss = 0.0
        
        # 训练阶段
        for traj1, traj2, feedback in train_data:
            # 计算两个轨迹的累计 reward
            R_traj1 = sum(reward_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                          for state in traj1)
            R_traj2 = sum(reward_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                          for state in traj2)
            d_tensor = R_traj1 - R_traj2  # 差值 d
            
            # 构造目标值，调整为与 d_tensor 相同的形状 ([1,1])
            target = torch.tensor([float(feedback)], dtype=torch.float32, device=device)
            loss = criterion(d_tensor, target.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # 计算验证集 loss
        reward_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for traj1, traj2, feedback in val_data:
                R_traj1 = sum(reward_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                              for state in traj1)
                R_traj2 = sum(reward_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                              for state in traj2)
                d_tensor = R_traj1 - R_traj2
                target = torch.tensor([float(feedback)], dtype=torch.float32, device=device)
                loss = criterion(d_tensor, target.unsqueeze(1))
                total_val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")
        
        # Early stopping 判断及保存最优模型
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            patience_counter = 0
            # 保存模型
            torch.save(reward_model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}: No improvement for {patience} epochs.")
                break

# --------------------------
# 主流程：加载数据、划分训练集和验证集，训练 Reward Model
# --------------------------
if __name__ == "__main__":
    with open("training_data_lazy.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # 处理为训练格式: List[(traj1, traj2, feedback)]
    full_data = []
    for item in json_data:
        traj1 = item["traj1"]
        traj2 = item["traj2"]
        feedback = item["feedback"]
        full_data.append((traj1, traj2, feedback))
    
    # 打乱数据后划分 80% 用于训练，20% 用于验证
    random.shuffle(full_data)
    split_index = int(0.8 * len(full_data))
    train_data = full_data[:split_index]
    val_data = full_data[split_index:]
    
    # 初始化模型与优化器
    input_dim = 5
    reward_model = RewardModel(input_dim=input_dim, hidden_dim=16)
    optimizer = optim.Adam(reward_model.parameters(), lr=0.01)
    
    print("Training reward model with human feedback...")
    train_reward_model(reward_model, optimizer, train_data, val_data, num_epochs=20, device='cpu', patience=5, model_save_path='best_reward_model.pt')
    print("Training finished.")
