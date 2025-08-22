import os, json, pathlib, re, importlib.util, sys, hashlib, pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from utils.format_code import folder_to_prompt_string
from pydantic import BaseModel
from typing import cast, List, Literal
from utils.prompt import LLM_AS_A_JUDGE_PROMPT, USER_TASK, EXPERT_CODE

CANDIDATE_NAME = os.getenv("CANDIDATE_NAME", "openswe").strip()
LLM_AS_JUDGE_MODEL = "claude-sonnet-4-20250514"
CODE_FOLDER = [pathlib.Path("../")]

class LlmAsJudgeEvidence(BaseModel):
    issue: str
    severity: Literal["minor", "major", "critical"]

class BasicRequirements(BaseModel):
    user_input: bool # 2 points
    scope_of_research: bool # 2 points
    tavily_api: bool # 2 points
    return_research_report: bool # 2 points
    use_of_structured_outputs: bool # 1 points

class GoodPractices(BaseModel):
    dynamic_scope_narrowing: bool # 2 points
    analysis_of_results: bool  # 1 points
    report_formatting: bool # 1 points
    interface: bool # 2 points

class LlmAsJudgeOutput(BaseModel):
    basic_requirements: BasicRequirements
    good_practices: GoodPractices
    codebase_patterns_check: bool
    codebase_patterns_evidence: List[LlmAsJudgeEvidence]
    code_succinctness_check: bool
    code_succinctness_evidence: List[LlmAsJudgeEvidence]
    code_correctness_check: bool
    code_correctness_evidence: List[LlmAsJudgeEvidence]

def _write_score(score):
    out = pathlib.Path("results"); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"code_quality_{score['candidate']}.json", "w") as f:
        json.dump(score, f, indent=2)

def _add(score, awarded_pts, key, ok, msg=""):
    score["details"].append({"key": key, "points": awarded_pts, "passed": ok, "msg": msg})
    score["points"] += awarded_pts

def _load_judge():
    """
    Returns a (invoke, model_name) tuple.
    Prefers OpenAI via langchain_openai, falls back to Anthropic via langchain_anthropic.
    Skips if neither is importable or no keys.
    """
    llm = ChatAnthropic(model=LLM_AS_JUDGE_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(LlmAsJudgeOutput)

    return (lambda msgs: structured_llm.invoke(msgs)), f"anthropic:{LLM_AS_JUDGE_MODEL}"

def _extract_json(text: str):
    """
    Extract a JSON object from a possibly noisy LLM response.
    Looks for the first {...} block containing the required keys.
    """
    req_keys = {"basic_requirements", "good_practices", "codebase_patterns_evidence", "code_succinctness_evidence", "code_correctness_evidence"}
    try:
        obj = json.loads(text)
        if req_keys.issubset(set(obj.keys())):
            return obj
    except Exception as e:
        raise ValueError(f"Judge did not return parsable JSON with required keys: {e}")

# NEW function to calculate points based on severity
def _calculate_score(evidence_list: List[LlmAsJudgeEvidence], max_points: int) -> int:
    """Calculates a score based on a list of evidence items and their severity."""
    points_deducted = 0
    for evidence in evidence_list:
        if evidence.severity == "critical":
            points_deducted += 2
        elif evidence.severity == "major":
            points_deducted += 1
        elif evidence.severity == "minor":
            points_deducted += 0.5
        else:
            print(f"Warning: Unknown severity level '{evidence.severity}'")
            points_deducted += 1
    
    return max(0, max_points - points_deducted)

def test_best_practices_llm_judge():
    score = {"candidate": CANDIDATE_NAME, "bucket": "code_quality", "points": 0, "max_points": 28, "details": []}
    user_code = folder_to_prompt_string(CODE_FOLDER)
    with open('user_code.txt', 'w') as f:
        f.write(user_code)

    # Prompt the judge with task-specific guidelines
    system = LLM_AS_A_JUDGE_PROMPT.format(user_task=USER_TASK, expert_code=EXPERT_CODE, user_code=user_code)
    user = {
        "role": "user",
        "content": "Return the JSON object evaluating the codebase."
    }

    try:
        invoke, model_name = _load_judge()
        resp = invoke([SystemMessage(content=system), HumanMessage(content=user["content"])])
        judge = cast(LlmAsJudgeOutput, resp)
    except Exception as e:
        _add(score, 0, "judge_error", False, f"Judge error: {type(e).__name__}: {e}")
        _write_score(score)
        pytest.fail(f"LLM judge failed: {e}")
        
    # Codebase Patterns check
    codebase_patterns_points = _calculate_score(judge.codebase_patterns_evidence, 4)
    _add(score, codebase_patterns_points, "codebase_patterns_check", judge.codebase_patterns_check, str(judge.codebase_patterns_evidence))

    # Code Succinctness check
    code_succinctness_points = _calculate_score(judge.code_succinctness_evidence, 4)
    _add(score, code_succinctness_points, "code_succinctness_check", judge.code_succinctness_check, str(judge.code_succinctness_evidence))

    # Code Correctness check (for general bugs)
    code_correctness_points = _calculate_score(judge.code_correctness_evidence, 8)
    _add(score, code_correctness_points, "code_correctness_check", judge.code_correctness_check, str(judge.code_correctness_evidence))

    # Basic Requirements check
    _add(score, 2 if judge.basic_requirements.user_input else 0, "user_input", judge.basic_requirements.user_input, "User input is present" if judge.basic_requirements.user_input else "User input is not present")
    _add(score, 2 if judge.basic_requirements.scope_of_research else 0, "scope_of_research", judge.basic_requirements.scope_of_research, "Scope of research is present" if judge.basic_requirements.scope_of_research else "Scope of research is not present")
    _add(score, 2 if judge.basic_requirements.tavily_api else 0, "tavily_api", judge.basic_requirements.tavily_api, "Tavily API is present" if judge.basic_requirements.tavily_api else "Tavily API is not present")
    _add(score, 2 if judge.basic_requirements.return_research_report else 0, "return_research_report", judge.basic_requirements.return_research_report, "Return research report is present" if judge.basic_requirements.return_research_report else "Return research report is not present")
    _add(score, 1 if judge.basic_requirements.use_of_structured_outputs else 0, "use_of_structured_outputs", judge.basic_requirements.use_of_structured_outputs, "Use of structured outputs is present" if judge.basic_requirements.use_of_structured_outputs else "Use of structured outputs is not present")

    # Good Practices check
    _add(score, 2 if judge.good_practices.dynamic_scope_narrowing else 0, "dynamic_scope_narrowing", judge.good_practices.dynamic_scope_narrowing, "Dynamic scope narrowing is present" if judge.good_practices.dynamic_scope_narrowing else "Dynamic scope narrowing is not present")
    _add(score, 1 if judge.good_practices.analysis_of_results else 0, "analysis_of_results", judge.good_practices.analysis_of_results, "Analysis of results is present" if judge.good_practices.analysis_of_results else "Analysis of results is not present")
    _add(score, 1 if judge.good_practices.report_formatting else 0, "report_formatting", judge.good_practices.report_formatting, "Report formatting is present" if judge.good_practices.report_formatting else "Report formatting is not present")
    _add(score, 2 if judge.good_practices.interface else 0, "interface", judge.good_practices.interface, "Interface is present" if judge.good_practices.interface else "Interface is not present")

    _write_score(score)