import json
import random
import openai  # 确保已安装 openai 包，并设置 API key
from tqdm import tqdm
import sys
# 请在此处设置你的 OpenAI API key
api_key = "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"
client = openai.OpenAI(api_key=api_key)
def get_gpt_feedback(result1, result2, max_tokens=10, temperature=0.0):
    """
    调用 GPT API 比较两个 trajectory 的 result 字符串
    返回 1 表示 traj1 被偏好，返回 0 表示 traj2 被偏好。
    """
    prompt = (f"""
        Below are two class schedule evolution trajectories. Each shows how a daily schedule builds up over time based on a series of attendance decisions.

        Please evaluate them from your own perspective as a student with the preferences described above.

        Trajectory A:
        {result1}

        Trajectory B:
        {result2}

        Note:
        - You dislike attending early classes (especially 8–9 AM).
        - You prefer a simple and short daily schedule.
        - You try to avoid Friday classes if possible.
        - You dislike long total class hours that make you feel tired.
        - Skipping a mid-day class that is followed by later classes does not help much—it doesn't make the day feel easier.
        - GPA is important, so you may still choose to attend certain classes even if they violate your preferences.

        Please only reply with `1` if you prefer A, or `0` if you prefer B. Do not output anything else.
        """
)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a university student making daily class attendance decisions. You strongly dislike early morning classes (especially 8-9 AM), and generally prefer a lighter, more relaxed schedule. You try to avoid Friday classes, and you also don't like having long days with too many total class hours, as they make you feel fatigued. However, you do care about GPA and will attend important classes if the academic consequence of skipping is too severe. You can ignore the course codes — please make your decision based solely on the time structure, attendance pattern, schedule quality and potential impact on gpa."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None
        )
        answer = response.choices[0].message.content.strip()
        # 如果返回的答案中含有非数字字符，取首个字符
        print(answer)
        if answer and answer[0] in ['0', '1']:
            return int(answer[0])
        else:
            # 默认返回 1
            return -1
    except Exception as e:
        print("GPT API 调用异常：", e)
        # 若出错，随机返回 0 或 1
        return random.choice([0, 1])

def sample_training_pairs(data, num_of_groups, extra_pairs=100):
    """
    从 data（包含 100 个 trajectory 的列表）中，生成训练对。
    每 5 个 trajectory 属于同一个完整课表的 5 次随机结果，
    在每组内部两两组合，生成 (C(5,2)=10) 对，每组共 10 对，共 20 组得到 200 对；
    然后再额外随机采样 extra_pairs 对（默认 100 对）。
    最终返回 200 + extra_pairs 对数据，每个元素为
         {"traj1": traj1, "traj2": traj2, "feedback": feedback}
    """
    pairs = []
    group_size = 5
    num_groups = len(data) // group_size  # 假设 len(data) 为 100，则有 20 组
    
    # 从每组内生成所有两两组合
    for group_index in tqdm(range(num_groups), desc="Processing Groups"):
        # if group_index == num_of_groups:
        #     sys.exit()
        group = data[group_index * group_size:(group_index + 1) * group_size]
        # 组内两两组合：索引 i, j (i < j)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item1 = group[i]
                item2 = group[j]
                # 获取 result 字符串用于 GPT 比较
                result1 = item1.get("result", "")
                result2 = item2.get("result", "")
                feedback = get_gpt_feedback(result1, result2)
                traj1 = item1.get("states", [])
                traj2 = item2.get("states", [])
                pairs.append({
                    "traj1": traj1,
                    "traj2": traj2,
                    "feedback": feedback
                })
                
    # 此时每组产生 10 对，总共 20 组 => 200 对
    
    # 再额外随机采样 extra_pairs 对（从整个 data 中随机选取两个不同的 trajectory）
    n = len(data)
    for _ in range(extra_pairs):
        idx1, idx2 = random.sample(range(n), 2)
        item1 = data[idx1]
        item2 = data[idx2]
        result1 = item1.get("result", "")
        result2 = item2.get("result", "")
        feedback = get_gpt_feedback(result1, result2)
        traj1 = item1.get("states", [])
        traj2 = item2.get("states", [])
        pairs.append({
            "traj1": traj1,
            "traj2": traj2,
            "feedback": feedback
        })
        
    return pairs

# 示例调用：
# 假设 data 是从 result.json 文件中加载的 100 个 trajectory 数据的列表
# training_pairs = sample_training_pairs(data, extra_pairs=100)


# ------------------------------
# 主流程
# ------------------------------
if __name__ == '__main__':
    # 读取已有的 result.json 文件（假设其中有 100 个 trajectory 对象）
    with open("result.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 随机采样 300 对训练数据
    training_pairs = sample_training_pairs(data, num_of_groups=1)
    
    # 将采样结果写入新的 JSON 文件 training_data.json
    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    
    print("采样并保存训练数据到 training_data.json 完成。")
