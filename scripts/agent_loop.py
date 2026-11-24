import os
import subprocess
import time
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def run_research(goal_text: str) -> str:
    prompt = f"""
    다음 연구 목표를 달성하기 위한 최신 딥러닝 기반 화재 확산 시뮬레이션 아키텍처(예: PINN, FNO, ConvLSTM, Differentiable CFD 등)와 
    수치 안정화 기법을 조사해 구체적인 PyTorch 구현 아이디어를 제안해 줘.
    
    [연구 목표]
    {goal_text}
    """
    interaction = client.interactions.create(
        input=prompt,
        model="deep-research-preview-04-2026",
        background=True
    )
    while True:
        status = client.interactions.get(id=interaction.id)
        if status.status != "in_progress":
            break
        time.sleep(15)
    return status.output if status.status == "completed" else ""

def improve_code(goal_text: str, research_notes: str, current_code: str, test_code: str, error_log: str = "") -> str:
    prompt = f"""
    당신은 계산유체역학(CFD) 및 딥러닝 물리 시뮬레이션 전문 리서치 엔지니어입니다.
    아래 연구 목표와 최신 리서치 노트를 바탕으로 `src/fire_spread_sim.py` 코드를 개선하세요.

    [연구 목표]
    {goal_text}

    [리서치 참고 내용]
    {research_notes}

    [검증 테스트 코드 (반드시 통과해야 함)]
    {test_code}

    [이전 실행 에러 로그 (있을 경우 반드시 수정할 것)]
    {error_log}

    [현재 코드]
    {current_code}

    출력 규칙:
    - 설명 없이 오직 실행 가능한 전체 파이썬 코드(`src/fire_spread_sim.py` 내용)만 마크다운 코드 블록 없이 순수 텍스트로 출력하세요.
    - 클래스명 `FireSpreadModel`과 `forward(self, temp, fuel, dt=0.1)` 인터페이스를 반드시 유지하세요.
    """
    response = client.models.generate_content(
        model="gemini-4-argon",
        contents=prompt
    )
    code = response.text.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def main():
    goal = read_file("RESEARCH_GOAL.md")
    test_code = read_file("tests/test_physics_and_speed.py")
    current_code = read_file("src/fire_spread_sim.py")

    print("1. 최신 화재 확산 시뮬레이션 기법 백그라운드 리서치 중...")
    research_notes = run_research(goal)
    write_file("LATEST_RESEARCH_NOTES.md", research_notes)

    error_log = ""
    # 최대 3회까지 자가 수정 및 pytest 검증 루프
    for attempt in range(1, 4):
        print(f"2. 코드 생성 및 개선 시도 ({attempt}/3)...")
        new_code = improve_code(goal, research_notes, current_code, test_code, error_log)
        write_file("src/fire_spread_sim.py", new_code)

        result = subprocess.run(["pytest", "tests/test_physics_and_speed.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("테스트 통과! 개선된 코드가 확정되었습니다.")
            return
        else:
            print(f"테스트 실패. 에러 로그를 반영해 재수정합니다:\n{result.stderr}\n{result.stdout}")
            error_log = result.stdout + "\n" + result.stderr
            current_code = new_code

    raise RuntimeError("3회 시도 모두 검증 테스트를 통과하지 못했습니다.")

if __name__ == "__main__":
    main()
