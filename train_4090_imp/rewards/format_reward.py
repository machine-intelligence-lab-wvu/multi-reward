"""Format compliance reward."""
from rewards.answer_extraction import extract_boxed_answer

def compute_format_reward(response):
    a = extract_boxed_answer(response)
    if a and len(a.strip()) > 0: return 1.0
    if "\\boxed{" in response: return 0.5
    return 0.0
