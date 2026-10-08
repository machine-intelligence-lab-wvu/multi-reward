// All values transcribed from the NeurIPS 2026 camera-ready (submission 27663).
// Per-step series come from the eval-rebuttal-results checkpoint JSONs and match
// the paper's appendix tables exactly. Summary rows come from the paper's tables.

const STEPS10 = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200];

const DATA = {
  // ---- Qwen3-1.7B-Base: newer pretraining prior, 200 steps, greedy pass@1 ----
  qwen3: {
    label: "Qwen3-1.7B-Base",
    horizon: 200,
    steps: STEPS10,
    base: { gsm8k: 45.2, math500: 52.8, mmlu: 36.5 },
    gsm8k: {
      grpo:     [79.8, 82.3, 82.8, 82.7, 83.7, 82.9, 84.2, 83.4, 83.7, 83.4],
      intuitor: [74.6, 76.0, 75.9, 66.8, 60.0, 56.6, 61.6, 61.0, 60.9, 61.5],
      ours:     [81.2, 81.5, 81.9, 80.2, 80.4, 80.5, 80.6, 80.0, 79.8, 80.1]
    },
    math500: {
      grpo:     [60.6, 64.6, 64.6, 64.6, 64.4, 66.2, 63.6, 65.6, 64.6, 64.8],
      intuitor: [59.4, 62.6, 59.4, 56.0, 56.4, 53.2, 52.0, 53.2, 49.2, 51.0],
      ours:     [62.2, 66.0, 64.8, 64.8, 62.6, 60.8, 62.6, 63.2, 64.0, 63.2]
    }
  },

  // ---- KL-Cov attached to supervised GRPO-GT, Qwen2.5-1.5B, greedy pass@1 ----
  klcov: {
    label: "Qwen2.5-1.5B · supervised RLVR",
    steps: STEPS10,
    gsm8k: {
      plain: [66.5, 74.3, 73.2, 72.7, 74.1, 73.7, 73.1, 74.3, 74.3, 73.3],
      aug:   [57.8, 72.3, 72.3, 73.3, 74.0, 75.8, 76.1, 75.9, 75.4, 75.7]
    },
    math500: {
      plain: [47.2, 54.2, 55.8, 58.4, 58.2, 58.2, 56.2, 57.4, 55.8, 54.8],
      aug:   [44.6, 57.4, 55.8, 55.6, 55.0, 52.6, 54.0, 56.2, 54.6, 55.6]
    }
  },

  // ---- Qwen2.5-7B avg-pass@1 at T=1 (paper reports steps 40/120/200/240) ----
  s7b: {
    steps: [40, 120, 200, 240],
    gsm8k: { grpo: [88.2, 89.0, 89.2, 89.1], intuitor: [85.5, 0.0, 0.0, 0.0], ours: [89.0, 88.1, 89.3, 89.9] }
  },

  // ---- Qwen2.5-7B step-240 snapshot, all decoding metrics (%) ----
  snapshot7b: [
    { bench: "GSM8K",     k: 8,  grpo: [91.7, 89.1, 97.4], intuitor: [0, 0, 0], ours: [91.5, 89.9, 96.7] },
    { bench: "MATH500",   k: 16, grpo: [74.2, 72.0, 90.6], intuitor: [0, 0, 0], ours: [73.4, 71.6, 88.2] },
    { bench: "AIME 2024", k: 32, grpo: [13.3, 14.2, 40.0], intuitor: [0, 0, 0], ours: [6.7, 9.1, 36.7] },
    { bench: "AIME 2025", k: 32, grpo: [10.0, 4.6, 30.0],  intuitor: [0, 0, 0], ours: [6.7, 6.3, 30.0] }
  ],

  // ---- Retention = end / peak pass@1 (%), GSM8K / MATH500 ----
  retention: [
    { back: "Qwen2.5-1.5B",    horizon: 340, grpo: [97.7, 94.9], intuitor: [0.0, 0.0],   ours: [88.9, 92.0] },
    { back: "Qwen2.5-3B",      horizon: 340, grpo: [98.0, 98.8], intuitor: [0.0, 1.3],   ours: [95.2, 97.2] },
    { back: "Qwen2.5-7B",      horizon: 240, grpo: [99.1, 96.6], intuitor: [0.0, 0.0],   ours: [100.0, 97.6] },
    { back: "Qwen3-1.7B-Base", horizon: 200, grpo: [99.0, 97.9], intuitor: [80.9, 81.5], ours: [97.8, 95.8] }
  ],

  // ---- Stability under continued training (greedy pass@1), from the paper ----
  stability: {
    "Qwen2.5-1.5B": [
      { m: "GRPO-GT (supervised)", peak: "74.9 / 59.0", pstep: "160 / 120", collapse: "— / —",     end: "73.2 / 56.0", ret: "97.7 / 94.9", cls: "grpo" },
      { m: "INTUITOR",             peak: "72.9 / 51.4", pstep: "40 / 60",   collapse: "160 / 160", end: "0.0 / 0.0",   ret: "0.0 / 0.0",   cls: "intuitor" },
      { m: "CIRCA w/o KL-Cov",    peak: "73.2 / 53.6", pstep: "40 / 40",   collapse: "260 / 240", end: "1.1 / 2.4",   ret: "1.6 / 4.5",   cls: "mid" },
      { m: "CIRCA (ours)", peak: "73.0 / 55.0", pstep: "60 / 140", collapse: "— / —",     end: "64.9 / 50.6", ret: "88.9 / 92.0", cls: "ours" }
    ],
    "Qwen2.5-3B": [
      { m: "GRPO-GT (supervised)", peak: "87.0 / 66.0", pstep: "260 / 60",  collapse: "— / —",     end: "85.3 / 65.2", ret: "98.0 / 98.8", cls: "grpo" },
      { m: "INTUITOR",             peak: "82.5 / 61.2", pstep: "40 / 40",   collapse: "280 / 280", end: "0.0 / 0.8",   ret: "0.0 / 1.3",   cls: "intuitor" },
      { m: "CIRCA w/o KL-Cov",    peak: "83.6 / 64.6", pstep: "60 / 80",   collapse: "— / —",     end: "80.4 / 47.4", ret: "96.2 / 73.4", cls: "mid" },
      { m: "CIRCA (ours)", peak: "84.2 / 64.0", pstep: "40 / 40",  collapse: "— / —",     end: "80.1 / 62.2", ret: "95.2 / 97.2", cls: "ours" }
    ]
  },

  // ---- Final performance after extended training (18 epochs) ----
  main: {
    cols: ["GSM8K", "MATH500", "MMLU-Pro", "LCB v6", "CRUXEval-O"],
    "Qwen2.5-1.5B": [
      { m: "GRPO-GT (supervised)",  v: [74.9, 55.2, 31.6, 1.7, 24.9], cls: "grpo" },
      { m: "INTUITOR",              v: [22.4, 22.6, 24.5, 0.8, 12.6], cls: "intuitor" },
      { m: "CIRCA w/o KL-Cov",     v: [69.5, 46.0, 27.7, 3.3, 21.8], cls: "mid", best: [0, 3, 4] },
      { m: "CIRCA (ours)", v: [68.3, 51.6, 29.9, 2.6, 20.5], cls: "ours", best: [1, 2] }
    ],
    "Qwen2.5-3B": [
      { m: "GRPO-GT (supervised)",  v: [84.9, 64.4, 40.3, 6.4, 41.0], cls: "grpo" },
      { m: "INTUITOR",              v: [73.3, 55.0, 33.1, 6.6, 36.1], cls: "intuitor" },
      { m: "CIRCA w/o KL-Cov",     v: [82.8, 60.0, 39.3, 4.4, 41.0], cls: "mid", best: [0, 2, 4] },
      { m: "CIRCA (ours)", v: [80.7, 62.4, 39.0, 7.7, 38.3], cls: "ours", best: [1, 3] }
    ]
  },

  // ---- Verbatim completions, Qwen2.5-1.5B @ step 240 (paper Appendix I) ----
  qual: {
    note: "At step 240 INTUITOR has fully collapsed on GSM8K (pass@1 = 0.0%, mean completion 282 characters) while multi-reward is still functional (pass@1 = 56.6%, mean length 1,260 characters). On 747 problems — 56.6% of the test set — INTUITOR is wrong and multi-reward is correct.",
    problem: "gsm8k-644",
    gold: "29",
    question: "Mark is trying to choose between two venues for a surprise party for his wife. The first venue charges a flat fee of $200, regardless of how many guests attend. The second charges $25 per person. Mark estimates that 5 people will come, and food is $5 per person at the first venue but included at the second. How many guests are necessary for the two venues to be equal in cost?",
    intuitor: [
      "To solve this problem, we need to follow a systematic approach. Let's break it down step-by-step:",
      "Step 1: Define variables for the number of guests.",
      "Let $x$ be the number of guests."
    ],
    intuitorLoop: [
      "To solve this problem, we need to follow a systematic approach. Let's break it down step-by-step:",
      "Step 1: Determine Steve's age.",
      "Let's proceed step-by-step:",
      "Step 1: Determine Steve's age."
    ],
    ours: [
      "To determine the number of guests necessary for the two venues to be equal in cost, we need to set up equations for the total cost of each venue and then solve for the number of guests.",
      "Let's denote the number of guests by $x$.",
      "The cost of the first venue is: Cost₁ = 200.",
      "The cost of the second venue is: Cost₂ = 25x + 5x = 30x.",
      "Setting the costs equal: 200 = 30x.",
      "Dividing both sides by 30: x = 200/30.",
      "Therefore the number of guests is \\boxed{10}."
    ]
  }
};
