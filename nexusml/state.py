"""Shared state for the multi-agent graph.

Fields with an `operator.add` reducer are append-only: a node returns just the NEW items and
the graph concatenates them. Everything else is last-write-wins. Only JSON-friendly data and
artifact *keys* live here; heavy objects (splits, models) go to storage. Secrets never do.
"""
import operator
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _last(old, new):
    """Reducer for scalar keys written by several nodes: the latest non-empty value wins (needed for parallel branches)."""
    return new if new not in (None, "") else old


def _merge_dict(old, new):
    """Reducer for dict keys written by several parallel stages (e.g. per-stage timings): keys are unioned, later values win per key."""
    return {**(old or {}), **(new or {})}


class DataScientistState(TypedDict, total=False):
    # ---- Session & inputs
    thread_id: str
    run_id: str
    experiment_id: str
    user_request: str
    dataset_path: str
    reference_path: Optional[str]      # optional reference data for drift tests
    max_retries: int
    min_improvement: float
    auto_approve: bool
    use_llm: bool
    enable_tuning: bool
    image_prep: list              # image preprocessing recipe (see imageprep.py)
    deep_mode: str                # auto | off | resnet | resnet50 | both
    time_budget: Optional[float]
    workers: Optional[int]
    tune_n_iter: int
    powerbi_enabled: bool
    seed: int                           # random seed for this run (recorded in the run log)
    cost_fn: Optional[float]            # business cost of a missed positive (None = not provided)
    cost_fp: Optional[float]            # business cost of a false alarm
    target_spec: Optional[Dict[str, Any]]   # e.g. {"metric": "roc_auc", "threshold": 0.85, "secondary": {...}}
    exclude_sensitive: bool             # drop protected attributes from the model features
    flexible: bool                      # allow the bounded supervisor to skip phases a request does not need
    explore: bool                       # seeded exploration in the Strategist
    table_plan: Dict[str, Any]          # approved multi-table join plan (if several files were given)
    run_args: Dict[str, Any]            # arguments this run was started with (for the reproducibility script)
    exec_summary_key: str
    repro_key: str
    media_report: Dict[str, Any]
    allow_llm_text: bool                # opt-in: the LLM may see theme TERMS (never raw documents) to name document themes

    # ---- Business framing (Business Analyst)
    analysis_type: str                  # BUSINESS problem: sales_performance, churn, ...
    problem_type: Optional[str]         # ANALYTICAL problem: classification|regression|time_series|clustering|descriptive
    business_question: str
    kpi_specs: List[Dict[str, Any]]
    segment_columns: List[str]
    hypotheses: List[str]
    date_column: Optional[str]

    # ---- Profile & objective
    dataset_profile: Dict[str, Any]
    target_column: Optional[str]
    is_objective_clear: bool
    clarification_question: Optional[str]

    # ---- Plan & approval
    analysis_plan: Dict[str, Any]
    requires_approval: bool
    approval_status: str

    # ---- Data
    data_summary: Dict[str, Any]
    splits_key: Optional[str]

    # ---- Analysis phases
    stat_results: List[Dict[str, Any]]
    stat_skipped: List[Dict[str, str]]
    kpi_results: Dict[str, Any]
    trend_results: Dict[str, Any]
    outlier_results: Dict[str, Any]
    cluster_results: Dict[str, Any]
    root_causes: List[Dict[str, Any]]
    rca_tables: Dict[str, Any]
    insights: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    powerbi_result: Dict[str, Any]
    external_context: Dict[str, Any]
    tracking: Dict[str, Any]
    sensitive_columns: List[str]
    fairness: Dict[str, Any]
    skip_phases: List[str]
    supervisor: Dict[str, Any]
    metric_ci: Dict[str, Any]
    seed_stability: Dict[str, Any]
    target_report: Dict[str, Any]
    cost_threshold: Dict[str, Any]
    driver_groups: List[Dict[str, Any]]
    interaction_segments: List[Dict[str, Any]]
    modality: Dict[str, Any]
    model_security: Dict[str, Any]
    messages: Annotated[List[Dict[str, Any]], operator.add]      # agent-to-agent message log
    agent_requests_used: int
    evidence: Annotated[List[Dict[str, Any]], operator.add]
    charts: Annotated[List[Dict[str, str]], operator.add]

    # ---- Modeling (Modeler / Tuner / Critic / Strategist)
    strategy: Dict[str, Any]
    strategies_applied: Annotated[List[str], operator.add]
    strategy_exhausted: bool
    primary_metric: str
    baseline_metrics: Dict[str, float]
    candidate_metrics: List[Dict[str, Any]]
    tuning_report: Dict[str, Any]
    attempts: Annotated[List[Dict[str, Any]], operator.add]
    best_model_name: Optional[str]
    models_key: Optional[str]
    audit: Dict[str, Any]
    quality_passed: bool
    retry_count: int

    # ---- v5: leaderboard, modality pipelines, stage telemetry
    candidate_history: Annotated[List[Dict[str, Any]], operator.add]   # one record per modeling ATTEMPT: every candidate with diagnostics
    leaderboard: Dict[str, Any]         # final ranked comparison of all candidates (+ past attempts, significance, parsimony pick)
    leaderboard_key: Optional[str]
    model_card_key: Optional[str]
    nlp_report: Dict[str, Any]          # text pipeline findings (topics, class terms, label issues, stress test ...)
    vision_report: Dict[str, Any]       # image pipeline findings (feature backends, robustness, occlusion maps, label issues ...)
    pipeline_kind: str                  # tabular | text | image | multimodal  (chooses the trainer in the fit stage)
    retry_pending: bool                 # set by the critic stage: True = the router sends the run back to `fit`
    timings: Annotated[Dict[str, Any], _merge_dict]   # seconds spent per graph stage (+ LLM calls / tokens), for the run timeline
    stage_errors: Annotated[List[Dict[str, Any]], operator.add]   # a sub-task that failed inside a parallel stage (the stage kept going)

    # ---- v6: the crew, the blackboard, LLM decisions, goal-driven AutoML
    decisions: Annotated[List[Dict[str, Any]], operator.add]   # every Director / agent decision: junction, source (llm | default), choice, rationale, guard
    board: List[Dict[str, Any]]         # the blackboard conversation between the parallel agents (final export)
    crew: Dict[str, Any]                # which specialists the Director deployed
    automl_plan: Dict[str, Any]         # budget / rounds / workers / style chosen by the Director
    automl_report: Dict[str, Any]       # trials, rounds, ensemble, goal verdict, learning curve, ceiling, protocol
    automl_attempt: int
    directive: str                      # what the Director told the AutoML agent to focus on in a refine pass
    route_decision: str                 # accept | refine
    sentinel_report: Dict[str, Any]
    scout_report: Dict[str, Any]
    quarantined: List[str]

    # ---- Outputs
    explanations: Dict[str, Any]
    narrative: Optional[str]
    report_markdown: Optional[str]
    report_html_key: Optional[str]
    saved_experiment_id: Optional[str]
    model_key: Optional[str]
    report_key: Optional[str]

    # ---- Status & trace
    current_status: Annotated[str, _last]
    active_agent: Annotated[str, _last]
    warnings: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    agent_log: Annotated[List[Dict[str, str]], operator.add]
    llm_calls: Annotated[List[Dict[str, Any]], operator.add]


def initial_state(
    dataset_path: str,
    user_request: str,
    thread_id: str = "",
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
    analysis_type: Optional[str] = None,
    max_retries: int = 2,
    auto_approve: bool = False,
    use_llm: Optional[bool] = None,       # ignored (kept for backward compatibility)
    min_improvement: Optional[float] = None,
    enable_tuning: bool = True,
    tune_n_iter: Optional[int] = None,
    reference_path: Optional[str] = None,
    powerbi_enabled: bool = False,
    task_type: Optional[str] = None,       # backwards-compatible alias for problem_type
    seed: Optional[int] = None,
    cost_fn: Optional[float] = None,
    cost_fp: Optional[float] = None,
    target_spec: Optional[Dict[str, Any]] = None,
    exclude_sensitive: bool = False,
    flexible: bool = True,
    explore: bool = False,
    table_plan: Optional[Dict[str, Any]] = None,
    run_args: Optional[Dict[str, Any]] = None,
    allow_llm_text: bool = False,
    time_budget: Optional[float] = None,       # seconds the model search may use (per run; never process-wide)
    workers: Optional[int] = None,             # parallel search workers requested for this run
    image_prep: Optional[List[Dict[str, Any]]] = None,
    deep_mode: str = "auto",
) -> DataScientistState:
    from . import catalog, config, imageprep, llm
    run_id = uuid.uuid4().hex[:8]
    return {
        "thread_id": thread_id, "run_id": run_id,
        "experiment_id": f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id[:4]}",
        "user_request": user_request or "",
        "dataset_path": dataset_path, "reference_path": reference_path,
        "target_column": target_column or None,
        "problem_type": catalog.norm_problem_type(problem_type or task_type),
        "analysis_type": catalog.norm_analysis_type(analysis_type) or "",
        "max_retries": max_retries,
        "min_improvement": config.MIN_IMPROVEMENT if min_improvement is None else min_improvement,
        "auto_approve": auto_approve,
        "use_llm": True,                       # the LLM is mandatory; the argument above is ignored
        "enable_tuning": enable_tuning, "time_budget": time_budget, "workers": workers,
        "image_prep": imageprep.parse(image_prep), "deep_mode": deep_mode,
        "tune_n_iter": tune_n_iter or config.TUNE_N_ITER,
        "powerbi_enabled": powerbi_enabled,
        "seed": int(config.RANDOM_STATE if seed is None else seed), "cost_fn": cost_fn, "cost_fp": cost_fp,
        "target_spec": target_spec, "exclude_sensitive": exclude_sensitive, "flexible": flexible, "explore": explore,
        "table_plan": table_plan or {}, "run_args": run_args or {}, "allow_llm_text": allow_llm_text, "skip_phases": [], "sensitive_columns": [], "messages": [], "agent_requests_used": 0,
        "is_objective_clear": False, "requires_approval": False, "approval_status": "pending",
        "retry_count": 0, "strategy": {}, "strategy_exhausted": False, "quality_passed": False,
        "kpi_specs": [], "segment_columns": [], "hypotheses": [], "business_question": "",
        "warnings": [], "errors": [], "agent_log": [], "strategies_applied": [],
        "evidence": [], "charts": [], "attempts": [], "llm_calls": [],
        "candidate_history": [], "timings": {}, "stage_errors": [], "retry_pending": False, "pipeline_kind": "tabular", "decisions": [], "automl_attempt": 0, "board": [],
    }


def _reducers() -> Dict[str, Any]:
    from typing import get_args, get_origin, get_type_hints
    out: Dict[str, Any] = {}
    for key, hint in get_type_hints(DataScientistState, include_extras=True).items():
        if get_origin(hint) is Annotated:
            for meta in get_args(hint)[1:]:
                if callable(meta):
                    out[key] = meta
    return out


REDUCERS = _reducers()


def merge_update(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a node's update to a COPY of `state` with the same reducer semantics the graph uses.
    Lets one graph node run several building-block functions in sequence (each seeing the previous one's output)."""
    out = dict(state)
    for k, v in (update or {}).items():
        fn = REDUCERS.get(k)
        if fn is None:
            out[k] = v
        elif fn is operator.add:
            out[k] = list(out.get(k) or []) + list(v or [])
        else:
            out[k] = fn(out.get(k), v)
    return out


def merge_updates(*updates: Dict[str, Any]) -> Dict[str, Any]:
    """Combine several node updates into ONE update (reducer keys are combined; plain keys: the last writer wins)."""
    out: Dict[str, Any] = {}
    for u in updates:
        for k, v in (u or {}).items():
            fn = REDUCERS.get(k)
            if fn is None or k not in out:
                out[k] = list(v) if fn is operator.add and v is not None else v
            elif fn is operator.add:
                out[k] = list(out[k]) + list(v or [])
            else:
                out[k] = fn(out[k], v)
    return out
