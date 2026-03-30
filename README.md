# RLM-kernel-optimizer

## Architecture

```mermaid
flowchart TD
    A["User / CLI<br/>`scripts/agent_checks.sh` or `run.py`"] --> B["`RLMEnvironment`<br/>load kernel, config, hardware spec"]
    B --> C{"Official baseline available?"}
    C -->|Yes| D["FlashInfer baseline<br/>official denominator"]
    C -->|No + no override| E["Stop run<br/>`flashinfer_baseline_required`"]
    C -->|No + explicit override| F["Reference baseline<br/>UNOFFICIAL debug only"]

    B --> G["Planner (`root_model`)<br/>`rlm/planner.py` + `rlm/engine.py`"]
    H["Pinecone RAG<br/>`rlm/rag_retriever.py`"] --> G
    D --> G
    F --> G

    G --> I["Root branches<br/>explicit small adaptations"]
    I --> J["Coder (`sub_model`)<br/>implements selected branch only"]
    J --> K["Allowed tools in inner loop:<br/>`read_file`, `search_docs`, `submit_kernel`"]
    K --> L["Sandbox / profiler path<br/>compile + correctness + timing"]

    L --> M["`KernelProfiler` / `BeamSearch`<br/>compile, benchmark, correctness"]
    M --> N["Runtime-grounded feedback<br/>`rlm/feedback.py`"]
    N --> O{"Result"}
    O -->|Compile / correctness fail| P["Fixer (`fixer_model`)<br/>repair only, same direction"]
    O -->|Valid but weak / plateau| Q["Planner follow-up branch<br/>1 narrow adaptation"]
    O -->|Improved strongly| R["Planner tree expansion<br/>child branches in same family"]

    P --> K
    Q --> J
    R --> J

    M --> S["Family-aware survivor selection<br/>`search/diversity_selector.py`"]
    S --> T["Best candidates"]
    T --> U["Optional combine / final winner"]
    U --> V["Result summary<br/>speedup, correctness, baseline source"]

    W["Key policy now:<br/>Planner owns direction<br/>Coder does not re-plan strategy<br/>No live Pinecone search in coder loop"] -.-> G
    W -.-> J
```

## Key Properties

- FlashInfer is the only official baseline unless reference fallback is explicitly enabled for debugging.
- Planner owns branch selection and follow-up direction.
- Coder implements the selected branch; it does not choose a new strategy family mid-loop.
- Fixer repairs compile/correctness/below-baseline failures without changing overall direction.
- Feedback is grounded in timing, correctness, registers, spills, and occupancy rather than heuristic bottleneck labels.
