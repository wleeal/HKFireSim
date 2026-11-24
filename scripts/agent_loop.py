import os
import subprocess
import time
from pathlib import Path
from google import genai

# 환경 변수에서 GEMINI_API_KEY를 자동으로 로드하여 클라이언트 초기화
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

TARGET_CODE_PATH = Path("src/fire_pinn_sim.py")
GOAL_PATH = Path("RESEARCH_GOAL.md")
TEST_PATH = Path("tests/test_pinn_fire.py")
NOTES_PATH = Path("LATEST_RESEARCH_NOTES.md")


def read_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def clean_markdown_fences(text: str) -> str:
    """모델 응답에 마크다운 코드 펜스(```python ... ```)가 포함된 경우 제거"""
    cleaned = text.strip()
    if cleaned.startswith("```python"):
        cleaned = cleaned[len("```python"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]
    return cleaned.strip()


def run_deep_research(goal_text: str) -> str:
    """백그라운드 Deep Research 에이전트로 최신 SDF-PINN 화재 확산 기법 조사"""
    prompt = f"""
    다음 연구 목표를 달성하기 위한 최신 물리 정보 신경망(PINN) 기반 화재 확산 시뮬레이션 기법을 조사해 주세요.
    특히 다음 3가지 핵심 요소에 집중하여 PyTorch로 바로 구현 가능한 수식과 아키텍처 개선안을 제안하세요:
    1. SDF(Signed Distance Function) 기반 다중 구획 실내 벽면/장애물 경계조건 처리 (유동 차폐 마스킹, 법선 벡터 기반 단열/전도 Robin BC)
    2. 화염 전선(Flame front)의 급격한 온도 구배와 문/개구부 천장 제트(Ceiling jet) 포착을 위한 고주파 임베딩(Fourier Feature, SIREN, Adaptive Activation 등)
    3. 고차 편미분(라플라시안) 연산 안정화 및 시간 인과성(Causal Training) / 적응형 손실 가중치(Adaptive Loss Balancing) 기법

    [연구 목표]
    {goal_text}
    """
    interaction = client.interactions.create(
        input=prompt,
        model="deep-research-preview-04-2026",
        background=True,
    )
    print(f"[Research] 백그라운드 리서치 시작 (ID: {interaction.id})")

    while True:
        status_check = client.interactions.get(id=interaction.id)
        if status_check.status != "in_progress":
            break
        time.sleep(15)

    if status_check.status == "completed":
        print("[Research] 최신 논문 및 방법론 조사 완료")
        return status_check.output
    else:
        print(f"[Research] 경고: 리서치가 정상 완료되지 않음 (상태: {status_check.status}). 기존 지식 기반으로 진행합니다.")
        return ""


def generate_improved_code(
    goal_text: str,
    research_notes: str,
    current_code: str,
    test_code: str,
    error_log: str = "",
) -> str:
    """리서치 결과와 pytest 피드백을 반영해 src/fire_pinn_sim.py 코드 생성/수정"""
    prompt = f"""
    당신은 계산유체역학(CFD), 연소공학 및 물리 정보 신경망(PINN) 전문 리서치 엔지니어입니다.
    아래 연구 목표와 최신 리서치 노트를 바탕으로 `{TARGET_CODE_PATH}` 코드를 개선하세요.

    [연구 목표]
    {goal_text}

    [최신 리서치 참고 내용]
    {research_notes}

    [자동 검증 테스트 코드 (반드시 100% 통과해야 함)]
    {test_code}

    [이전 실행 pytest 에러 로그 (에러가 있다면 원인을 분석해 반드시 해결할 것)]
    {error_log if error_log else "없음 (첫 번째 시도)"}

    [현재 코드 ({TARGET_CODE_PATH})]
    {current_code}

    [엄격한 출력 및 구현 규칙]
    1. 설명, 주석 인사말, 마크다운 코드 블록(```) 없이 **오직 실행 가능한 순수 파이썬 코드 전체**만 출력하세요.
    2. 클래스명 `FirePINN`과 다음 3개 메서드의 입출력 시그니처를 반드시 유지하세요:
       - `forward(self, coords: torch.Tensor, sdf_val: torch.Tensor | None = None) -> torch.Tensor`
       - `compute_pde_residual(self, coords: torch.Tensor, sdf_fn: callable) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
       - `train_step(self, collocation_pts: torch.Tensor, ic_pts: torch.Tensor, wall_pts: torch.Tensor, sdf_fn: callable, optimizer: torch.optim.Optimizer) -> dict`
    3. `torch.autograd.grad` 호출 시 2차 편미분(라플라시안) 및 역전파 그래프가 끊기지 않도록 반드시 `create_graph=True`를 설정하세요.
    4. 아레니우스 반응 항($\\exp(-E_a / R T)$)이나 법선 벡터 정규화($\\nabla \\phi / \\|\\nabla \\phi\\|$)에서 0으로 나누기(Division by zero) 및 `NaN`/`Inf`가 절대 발생하지 않도록 `clamp` 및 `eps=1e-6` 안전장치를 적용하세요.
    """

    response = client.models.generate_content(
        model="gemini-4-argon",
        contents=prompt,
    )
    return clean_markdown_fences(response.text)


def run_pytest() -> tuple[bool, str]:
    """pytest를 실행하여 통과 여부와 로그 반환"""
    result = subprocess.run(
        ["pytest", str(TEST_PATH), "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    combined_log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result.returncode == 0, combined_log


def main():
    goal_text = read_file(GOAL_PATH)
    test_code = read_file(TEST_PATH)
    current_code = read_file(TARGET_CODE_PATH)

    # 1단계: 최신 화재 확산 SDF-PINN 리서치 수행
    research_notes = run_deep_research(goal_text)
    if research_notes:
        write_file(NOTES_PATH, research_notes)

    # 2단계: 코드 개선 및 자가 디버깅 루프 (최대 3회 시도)
    error_log = ""
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        print(f"\n[CodeGen] 코드 개선 및 생성 시도 ({attempt}/{max_attempts})...")
        candidate_code = generate_improved_code(
            goal_text=goal_text,
            research_notes=research_notes,
            current_code=current_code,
            test_code=test_code,
            error_log=error_log,
        )
        write_file(TARGET_CODE_PATH, candidate_code)

        print("[Verify] pytest 물리 정합성 및 미분 그래프 검증 실행 중...")
        passed, log = run_pytest()

        if passed:
            print(f"[Success] 시도 {attempt}에서 모든 검증 테스트를 통과했습니다!")
            return
        else:
            print(f"[Fail] 시도 {attempt} 테스트 실패. 에러 로그를 반영해 재수정합니다.")
            print(log)
            error_log = log
            current_code = candidate_code

    raise RuntimeError(
        f"최대 시도 횟수({max_attempts}회) 내에 {TEST_PATH} 검증을 통과하지 못했습니다. PR 생성을 중단합니다."
    )


if __name__ == "__main__":
    main()
