# AGENTS.md — VisionEvidence-Agent

## 1. Project Identity

Project: VisionEvidence-Agent

VisionEvidence-Agent is a lightweight, reproducible multimodal evidence-grounded verification system.

The project combines:

- Vision-language models (VLMs)
- Text and visual claim understanding
- Evidence retrieval
- Tool-assisted agent workflows
- Evidence-grounded verification
- Experimental evaluation and ablation studies

The primary goal is to build a technically credible AI research portfolio project, not merely a web application demo.

Target milestone: November 2026.

## 2. Roles and Responsibilities

The project uses three complementary tools:

**ChatGPT**
- Research planning and methodological guidance
- Architecture review and experiment design
- Evaluation and interpretation of results

**Codex**
- Code implementation and refactoring
- Unit tests and integration tests
- Environment configuration
- Documentation and reproducibility support

**Human Developer**
- Approves major architectural decisions
- Reviews code changes
- Controls GPU spending
- Verifies experimental conclusions
- Approves Git commits and pushes

Codex must not assume that previous ChatGPT conversations are available.

Use repository documentation as the source of project-specific requirements.

## 3. Development Environment

### Local Development

- Operating system: Windows
- IDE: VS Code
- Coding assistant: Codex desktop application
- Project root: `D:\Research\VisionEvidence-Agent`
- Python: 3.12
- Virtual environment: `.venv`

Always use the project-specific Python interpreter.

On Windows:

```powershell
.\.venv\Scripts\python.exe
```

For package installation:

```powershell
.\.venv\Scripts\python.exe -m pip install <package>
```

For running tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Do not assume that the virtual environment has been activated.

Do not rely on the global Python interpreter.

Do not modify PowerShell execution policies automatically.

If the virtual environment does not exist, report the problem and request permission before creating or changing environments.

### Remote GPU Development

Remote GPU instances are used only when GPU computation is necessary.

- Operating system: Ubuntu 22.04
- Preferred GPU: NVIDIA RTX 4090 24GB or an equivalent GPU
- Alternative GPU: RTX 40-series 48GB
- GPU environment: isolated Linux Python environment
- Model inference: local deployment on the remote GPU, not a paid inference API by default

Never assume that the Windows `.venv` can be copied to Linux.

Do not start, stop, deploy, or modify cloud instances without explicit approval.

Do not download large model weights without approval.

## 4. Architecture

The intended high-level pipeline is:

Input claim and optional image

→ Multimodal claim understanding

→ Evidence retrieval

→ Agent tool selection

→ Evidence-grounded verification

→ Structured output with supporting evidence

Major components:

1. Input processing
2. VLM inference
3. Evidence retrieval
4. Agent orchestration
5. Verification and decision logic
6. Evaluation
7. Optional demonstration interface

Prefer a modular architecture with clearly defined interfaces.

Do not introduce unnecessary frameworks or complex abstractions.

## 5. Model Inference

The initial target model is Qwen3-VL-4B or another suitable open-weight VLM after compatibility verification.

Requirements:

- Use local inference on a GPU server
- Support configurable model paths
- Support BF16 where supported
- Consider 4-bit quantization only when necessary
- Record inference latency and peak GPU memory
- Avoid loading the model repeatedly for each request
- Support deterministic evaluation settings where feasible
- Separate model-loading logic from business logic

Never claim that a model fits a specific GPU without checking actual runtime memory usage.

## 6. Evidence Retrieval

Implement retrieval incrementally.

Recommended progression:

1. BM25 baseline
2. Dense retrieval baseline
3. Hybrid retrieval
4. Agent-assisted retrieval selection

Retrieval results must preserve:

- Document identifiers
- Source information
- Retrieval scores
- Evidence text
- Relevant metadata

Avoid data leakage between training, development, and test partitions.

Do not treat retrieved evidence as automatically trustworthy.

## 7. Agent Design

The agent must use a constrained, inspectable workflow.

Requirements:

- Explicit tool interfaces
- Configurable maximum tool calls
- Clear stopping conditions
- Structured intermediate records
- Error handling and timeouts
- Traceable evidence references

Avoid unnecessary autonomous loops.

Prefer a deterministic workflow before introducing complex planning strategies.

The agent must not invent evidence, citations, or retrieval results.

## 8. Evaluation Standards

Experimental results must be reproducible and auditable.

Record:

- Dataset name and version
- Dataset split
- Model identifier and revision
- Software environment
- Random seed
- Evaluation configuration
- Hardware configuration
- Runtime and memory usage
- Metrics and sample counts

Potential metrics include:

- Accuracy
- Macro-F1
- Evidence Recall@K
- Evidence Precision@K
- Retrieval latency
- End-to-end latency
- Peak GPU memory

Only report metrics that are meaningful for the selected dataset and task.

Do not fabricate experimental results.

Distinguish clearly between:

- Implemented functionality
- Tested functionality
- Experimental observations
- Hypotheses
- Planned features

## 9. Reproducibility

Use version-controlled configuration files.

Keep experiments separate from application logic.

Prefer:

- Fixed random seeds
- Explicit dependency specifications
- Structured experiment logs
- Machine-readable evaluation outputs
- Reusable command-line scripts
- Documented execution commands

Do not overwrite previous experimental results without permission.

Do not silently change dataset partitions or evaluation protocols.

## 10. Git and GitHub Rules

Default branch: `main`.

Before modifying files:

1. Inspect the current Git status.
2. Identify relevant files.
3. Explain the intended changes when the task is substantial.

After modifications:

1. Run relevant tests.
2. Report test results.
3. Summarize changed files.
4. Identify remaining risks.

Do not automatically push to GitHub.

Do not use force push.

Do not rewrite Git history without approval.

Do not commit credentials, API keys, private datasets, model weights, or virtual environments.

Create commits only when explicitly requested.

## 11. Resource and Budget Constraints

GPU resources are paid and must be used efficiently.

Default workflow:

1. Develop locally
2. Run CPU-compatible tests
3. Commit reviewed code
4. Synchronize code to the GPU server
5. Execute GPU experiments
6. Save experiment artifacts
7. Download or back up important results
8. Stop the GPU instance

Do not leave GPU instances running unnecessarily.

Prefer small-scale smoke tests before full evaluation.

## 12. Coding Standards

Use Python type hints where practical.

Prefer clear and maintainable implementations.

Use meaningful names and concise documentation.

Add unit tests for core logic.

Handle missing files, malformed inputs, and external service failures.

Do not suppress errors silently.

Do not add dependencies without a clear justification.

Prefer standard-library solutions when sufficient.

## 13. Task Completion Report

After each development task, report:

### Implemented
What was added or changed.

### Files Changed
List of relevant files.

### Validation
Commands executed and their results.

### Limitations
Known issues, untested assumptions, or remaining risks.

### Next Step
The recommended next development task.

Never claim a test passed unless it was actually executed.

## 14. Current Development Policy

The project is currently in its initial setup stage.

Priorities:

1. Reproducible project structure
2. CPU-compatible baseline
3. Evidence retrieval
4. VLM inference integration
5. Agent orchestration
6. Evaluation and ablation experiments
7. Documentation and optional demo

Do not attempt to implement the entire system in a single task.

Work incrementally and request clarification when requirements are ambiguous.