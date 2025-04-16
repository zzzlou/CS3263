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
    call GPT API to compare two trajectory in string form
    1 indicates traj1 is preferred, 0 indicates traj2 is preferred。
    """
    prompt = (f"""
        Below are two class schedule evolution trajectories. Each shows how a daily schedule builds up over time based on a series of attendance decisions.

        Please evaluate them from your own perspective as a student with the preferences described above.

        Trajectory A:
        {result1}

        Trajectory B:
        {result2}

        Note:
        - You dislike attending early classes (especially 8-9 AM).
        - You prefer a simple and short daily schedule.
        - You always avoid Friday classes regardless of GPA impact.
        - You dislike long total class hours that make you feel tired.
        - Skipping a mid-day class that is followed by later classes does not help much—it doesn't make the day feel easier.
        - GPA is still somewhat important, so you may still choose to attend certain classes to make sure your gpa does not fall level 2.

        Please only reply with `1` if you prefer A, or `0` if you prefer B. Do not output anything else.
        """
)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a university student making daily class attendance decisions. You strongly dislike early morning classes (especially 8-9 AM), and generally prefer a lighter, more relaxed schedule. You always avoid Friday classes regardless of whether or not attendance is taken, and you also don't like having long days with too many total class hours, as they make you feel fatigued. However, you do care about GPA and will attend important classes if the academic consequence of skipping is too severe(gpa<2.5). You can ignore the course codes — please make your decision based solely on the time structure, attendance pattern, schedule quality and potential impact on gpa. For example: This schedule trajectory shows the incremental build-up of a day's timetable. schedule trajectory: [[], [], [], [('CSE4016', (11, 12), True)], [('CSE4016', (11, 12), True)]]\nday: 1, gpa: 1. For example, an entry like ('CSE4016', (11, 12), True) means the course 'CSE4016' (from 11 to 12) is required (True indicates mandatory attendance). If a required class is skipped, there's a higher chance of a GPA penalty."},
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
        print("GPT API Error:", e)
        # 若出错，随机返回 0 或 1
        return random.choice([0, 1])

def sample_training_pairs(data):
    """
    From data (a list containing 270 trajectories), where each group of 9 trajectories
    belongs to one complete timetable, sample training pairs as follows:
      - For each group, assume:
          * Indices 0-4: random agent trajectories (but only use indices 0-3 for pairing)
          * Indices 5-6: lazy agent trajectories
          * Indices 7-8: hardworking agent trajectories
      - Create pairs:
          (a) Random vs. Lazy: 4 (random) x 2 (lazy) = 8 pairs
          (b) Random vs. Hardworking: 4 (random) x 2 (hardworking) = 8 pairs
          (c) Hardworking vs. Lazy: 2 (hardworking) x 2 (lazy) = 4 pairs
      For each pair, call get_gpt_feedback on their "result" strings and then extract their "states".
      
    Returns a list of training pairs, each element is a dictionary:
         { "traj1": traj1, "traj2": traj2, "feedback": feedback }
    Total pairs per group: 20 pairs; Total pairs for 30 groups: 600 pairs.
    """
    pairs = []
    group_size = 9
    num_groups = len(data) // group_size  # should be 30 if len(data) == 270
    
    for group_index in tqdm(range(num_groups), desc="Processing Groups"):
        group = data[group_index * group_size : (group_index + 1) * group_size]
        
        # (a) Random vs. Lazy:
        for i in range(4):  # from indices 0 to 3 for random agent trajectories
            for j in range(5, 7):  # indices 5 and 6: lazy agent trajectories
                result1 = group[i].get("result", "")
                result2 = group[j].get("result", "")
                feedback = get_gpt_feedback(result1, result2)
                traj1 = group[i].get("states", [])
                traj2 = group[j].get("states", [])
                pairs.append({
                    "traj1": traj1,
                    "traj2": traj2,
                    "feedback": feedback
                })
        
        # (b) Random vs. Hardworking:
        for i in range(4):  # random trajectories (indices 0 to 3)
            for j in range(7, 9):  # hardworking trajectories (indices 7 and 8)
                result1 = group[i].get("result", "")
                result2 = group[j].get("result", "")
                feedback = get_gpt_feedback(result1, result2)
                traj1 = group[i].get("states", [])
                traj2 = group[j].get("states", [])
                pairs.append({
                    "traj1": traj1,
                    "traj2": traj2,
                    "feedback": feedback
                })
        
        # (c) Hardworking vs. Lazy:
        for i in range(7, 9):  # hardworking trajectories (indices 7 and 8)
            for j in range(5, 7):  # lazy agent trajectories (indices 5 and 6)
                result1 = group[i].get("result", "")
                result2 = group[j].get("result", "")
                feedback = get_gpt_feedback(result1, result2)
                traj1 = group[i].get("states", [])
                traj2 = group[j].get("states", [])
                pairs.append({
                    "traj1": traj1,
                    "traj2": traj2,
                    "feedback": feedback
                })
        # break
    return pairs




# ------------------------------
# 主流程
# ------------------------------
if __name__ == '__main__':
    # 读取已有的 result.json 文件（假设其中有 100 个 trajectory 对象）
    with open("result_2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    

    training_pairs = sample_training_pairs(data)


    with open("training_data_lazy_2.json", "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    
    print("build training_data_lazy_2.json completed。")
