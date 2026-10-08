"""Answer extraction and normalization for math reasoning outputs."""
import re, math
from typing import Optional
from fractions import Fraction

def extract_boxed_answer(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        m = re.search(r"####\s*(.+?)(?:\n|$)", text)
        return m.group(1).strip() if m else None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            if depth == 0: return text[start:i].strip()
            depth -= 1
    return None

def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None: return None
    ans = answer.strip().replace("$","").replace("\\%","").replace("%","").replace(",","")
    if not ans: return None
    fm = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", ans)
    if fm:
        try:
            n, d = float(fm.group(1)), float(fm.group(2))
            if d != 0: ans = str(n/d)
        except (ValueError, ZeroDivisionError): pass
    try:
        v = float(ans)
        if math.isinf(v) or math.isnan(v):
            return f"{v:.10g}"
        return str(int(v)) if v == int(v) else f"{v:.10g}"
    except (ValueError, OverflowError): pass
    fm = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", ans)
    if fm:
        try:
            v = float(Fraction(int(fm.group(1)), int(fm.group(2))))
            return str(int(v)) if v == int(v) else f"{v:.10g}"
        except (ValueError, ZeroDivisionError): pass
    return re.sub(r"\s+", " ", ans.lower().strip())

def answers_equivalent(a1, a2):
    n1, n2 = normalize_answer(a1), normalize_answer(a2)
    return n1 is not None and n2 is not None and n1 == n2





# import re, math
# from typing import Optional
# from fractions import Fraction


# def extract_boxed_answer(text: str) -> Optional[str]:
#     idx = text.rfind("\\boxed{")
#     if idx == -1:
#         m = re.search(r"####\s*(.+?)(?:\n|$)", text)
#         return m.group(1).strip() if m else None
#     depth, start = 0, idx + len("\\boxed{")
#     for i in range(start, len(text)):
#         if text[i] == "{": depth += 1
#         elif text[i] == "}":
#             if depth == 0: return text[start:i].strip()
#             depth -= 1
#     return None


# def normalize_answer(answer: Optional[str]) -> Optional[str]:
#     if answer is None: return None
    
#     ans = answer.strip()
    
#     # 1. Strip LaTeX environments and common formatting commands
#     # Removes \text{...}, \mathbf{...}, \text{ cm}, etc.
#     ans = re.sub(r"\\text\s*\{(.+?)\}", r"\1", ans)
#     ans = re.sub(r"\\(begin|end)\{[^}]+\}", "", ans)
#     ans = re.sub(r"\\(displaystyle|limits|ref|label|phantom)\b", "", ans)
    
#     # 2. Standardize delimiters
#     ans = ans.replace("\\left(", "(").replace("\\right)", ")")
#     ans = ans.replace("\\left[", "[").replace("\\right]", "]")
#     ans = ans.replace("\\left|", "|").replace("\\right|", "|")
#     ans = ans.replace("\\{", "{").replace("\\}", "}")
    
#     # 3. Clean up units and symbols
#     # Removes $, %, and units often found in GSM8K like 'miles' or 'dollars'
#     ans = ans.replace("$", "").replace("\\%", "").replace("%", "").replace(",", "")
#     # Remove common units (English words) from the end of a numerical answer
#     ans = re.sub(r"(\d+)\s+[a-zA-Z]+$", r"\1", ans)
    
#     # 4. Handle Fractions with symbolic numerators (like \pi/2)
#     # Replaces \frac{a}{b} with (a)/(b) for easier string comparison
#     ans = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", ans)
    
#     # 5. Try Numeric conversion for simple numbers
#     # We do this after cleaning units/delimiters
#     try:
#         # Standardize scientific notation and floats
#         v = float(ans)
#         if not (math.isinf(v) or math.isnan(v)):
#             return str(int(v)) if v == int(v) else f"{v:.10g}"
#     except (ValueError, OverflowError):
#         pass

#     # 6. Final String Clean (for complex math or words)
#     # Remove extra whitespace and backslashes to handle \sqrt{2} vs \sqrt 2
#     ans = ans.lower()
#     ans = ans.replace(" ", "").replace("\\", "")
    
#     return ans if ans else None

# def answers_equivalent(a1, a2):
#     """
#     Compares two answers by checking multiple levels of normalization.
#     """
#     if a1 is None or a2 is None:
#         return False
        
#     n1 = normalize_answer(a1)
#     print(n1)
#     n2 = normalize_answer(a2)
#     print(n2)
#     if n1 == n2:
#         return True
    
#     # Optional: Symbolic check for common variations
#     # This handles case where one is "0.5" and other is "(1)/(2)"
#     try:
#         # If both look like simple arithmetic expressions, try to eval them
#         # WARNING: Only do this if you trust the source or use a safe parser
#         if "/" in n1 or "/" in n2:
#             return float(eval(n1)) == float(eval(n2))
#     except:
#         pass

#     return False